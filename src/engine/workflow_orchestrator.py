"""
AgentFlow Workflow Orchestrator - Manages workflow execution and agent coordination
Based on AgentFlow design specification
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.engine.agent_engine import AgentEngine

class WorkflowNode:
    """Represents a node in the workflow (agent, condition, or action)"""
    
    def __init__(self, node_id: str, node_type: str, config: Dict):
        self.id = node_id
        self.type = node_type  # 'agent', 'condition', 'action', 'input', 'output'
        self.config = config
        self.status = 'pending'  # pending, running, completed, failed, skipped
        self.input_data = None
        self.output_data = None
        self.error_message = None
        self.execution_time_ms = 0
        self.started_at = None
        self.completed_at = None
    
    def to_dict(self):
        return {
            'id': self.id,
            'type': self.type,
            'config': self.config,
            'status': self.status,
            'input_data': self.input_data,
            'output_data': self.output_data,
            'error_message': self.error_message,
            'execution_time_ms': self.execution_time_ms,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }


class WorkflowConnection:
    """Represents a connection between workflow nodes"""
    
    def __init__(self, source_id: str, target_id: str, condition: str = None):
        self.source_id = source_id
        self.target_id = target_id
        self.condition = condition  # Optional condition for conditional execution
    
    def to_dict(self):
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'condition': self.condition
        }


class WorkflowOrchestrator:
    """Core workflow orchestration engine"""
    
    def __init__(self):
        self.agent_engine = AgentEngine()
        self.logger = logging.getLogger(__name__)
        self.max_workers = 4  # For parallel execution
    
    def execute_workflow(self, workflow_definition: Dict, input_data: Any, 
                        execution_id: str = None) -> Dict:
        """Execute a complete workflow"""
        start_time = time.time()
        execution_id = execution_id or f"exec_{int(time.time())}"
        
        try:
            # Parse workflow definition
            nodes = self._parse_nodes(workflow_definition.get('nodes', []))
            connections = self._parse_connections(workflow_definition.get('connections', []))
            config = workflow_definition.get('configuration', {})
            
            # Build execution graph
            execution_graph = self._build_execution_graph(nodes, connections)
            
            # Initialize workflow state
            workflow_state = {
                'execution_id': execution_id,
                'status': 'running',
                'current_data': input_data,
                'node_outputs': {},
                'execution_log': [],
                'start_time': datetime.utcnow(),
                'progress_percentage': 0
            }
            
            # Execute workflow
            result = self._execute_workflow_graph(execution_graph, workflow_state, config)
            
            total_execution_time = int((time.time() - start_time) * 1000)
            
            return {
                'success': result['success'],
                'execution_id': execution_id,
                'output_data': result.get('output_data'),
                'execution_log': workflow_state['execution_log'],
                'node_results': {node.id: node.to_dict() for node in nodes.values()},
                'total_execution_time_ms': total_execution_time,
                'total_tokens_used': result.get('total_tokens_used', 0),
                'total_cost_usd': result.get('total_cost_usd', 0.0),
                'error_message': result.get('error_message')
            }
            
        except Exception as e:
            total_execution_time = int((time.time() - start_time) * 1000)
            self.logger.error(f"Workflow execution failed: {str(e)}")
            
            return {
                'success': False,
                'execution_id': execution_id,
                'error_message': str(e),
                'total_execution_time_ms': total_execution_time,
                'total_tokens_used': 0,
                'total_cost_usd': 0.0
            }
    
    def _parse_nodes(self, nodes_data: List[Dict]) -> Dict[str, WorkflowNode]:
        """Parse node definitions into WorkflowNode objects"""
        nodes = {}
        for node_data in nodes_data:
            node = WorkflowNode(
                node_id=node_data['id'],
                node_type=node_data['type'],
                config=node_data.get('config', {})
            )
            nodes[node.id] = node
        return nodes
    
    def _parse_connections(self, connections_data: List[Dict]) -> List[WorkflowConnection]:
        """Parse connection definitions into WorkflowConnection objects"""
        connections = []
        for conn_data in connections_data:
            connection = WorkflowConnection(
                source_id=conn_data['source_id'],
                target_id=conn_data['target_id'],
                condition=conn_data.get('condition')
            )
            connections.append(connection)
        return connections
    
    def _build_execution_graph(self, nodes: Dict[str, WorkflowNode], 
                             connections: List[WorkflowConnection]) -> Dict:
        """Build execution graph with dependencies"""
        graph = {
            'nodes': nodes,
            'dependencies': {},  # node_id -> list of prerequisite node_ids
            'dependents': {},    # node_id -> list of dependent node_ids
            'entry_points': [],  # nodes with no dependencies
            'exit_points': []    # nodes with no dependents
        }
        
        # Initialize dependency tracking
        for node_id in nodes:
            graph['dependencies'][node_id] = []
            graph['dependents'][node_id] = []
        
        # Build dependency relationships
        for connection in connections:
            source_id = connection.source_id
            target_id = connection.target_id
            
            graph['dependencies'][target_id].append(source_id)
            graph['dependents'][source_id].append(target_id)
        
        # Identify entry and exit points
        for node_id in nodes:
            if not graph['dependencies'][node_id]:
                graph['entry_points'].append(node_id)
            if not graph['dependents'][node_id]:
                graph['exit_points'].append(node_id)
        
        return graph
    
    def _execute_workflow_graph(self, graph: Dict, workflow_state: Dict, 
                               config: Dict) -> Dict:
        """Execute the workflow graph"""
        nodes = graph['nodes']
        dependencies = graph['dependencies']
        entry_points = graph['entry_points']
        
        completed_nodes = set()
        failed_nodes = set()
        total_tokens = 0
        total_cost = 0.0
        
        # Start with entry points
        ready_nodes = set(entry_points)
        
        while ready_nodes and workflow_state['status'] == 'running':
            # Determine which nodes can run in parallel
            parallel_nodes = []
            sequential_nodes = []
            
            for node_id in ready_nodes:
                node = nodes[node_id]
                if node.type == 'agent' and config.get('parallel_execution_enabled', True):
                    parallel_nodes.append(node_id)
                else:
                    sequential_nodes.append(node_id)
            
            # Execute parallel nodes
            if parallel_nodes:
                parallel_results = self._execute_parallel_nodes(
                    parallel_nodes, nodes, workflow_state
                )
                
                for node_id, result in parallel_results.items():
                    if result['success']:
                        completed_nodes.add(node_id)
                        total_tokens += result.get('tokens_used', 0)
                        total_cost += result.get('cost_usd', 0.0)
                    else:
                        failed_nodes.add(node_id)
                        if config.get('error_handling_strategy', 'stop_on_error') == 'stop_on_error':
                            workflow_state['status'] = 'failed'
                            return {
                                'success': False,
                                'error_message': f"Node {node_id} failed: {result.get('error')}",
                                'total_tokens_used': total_tokens,
                                'total_cost_usd': total_cost
                            }
            
            # Execute sequential nodes
            for node_id in sequential_nodes:
                result = self._execute_single_node(nodes[node_id], workflow_state)
                
                if result['success']:
                    completed_nodes.add(node_id)
                    total_tokens += result.get('tokens_used', 0)
                    total_cost += result.get('cost_usd', 0.0)
                else:
                    failed_nodes.add(node_id)
                    if config.get('error_handling_strategy', 'stop_on_error') == 'stop_on_error':
                        workflow_state['status'] = 'failed'
                        return {
                            'success': False,
                            'error_message': f"Node {node_id} failed: {result.get('error')}",
                            'total_tokens_used': total_tokens,
                            'total_cost_usd': total_cost
                        }
            
            # Update ready nodes for next iteration
            ready_nodes = self._get_ready_nodes(
                dependencies, completed_nodes, failed_nodes, nodes
            )
            
            # Update progress
            total_nodes = len(nodes)
            completed_count = len(completed_nodes) + len(failed_nodes)
            workflow_state['progress_percentage'] = int((completed_count / total_nodes) * 100)
        
        # Determine final output
        output_data = self._collect_workflow_output(nodes, graph['exit_points'])
        
        return {
            'success': len(failed_nodes) == 0,
            'output_data': output_data,
            'total_tokens_used': total_tokens,
            'total_cost_usd': total_cost
        }
    
    def _execute_parallel_nodes(self, node_ids: List[str], nodes: Dict[str, WorkflowNode], 
                               workflow_state: Dict) -> Dict:
        """Execute multiple nodes in parallel"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all nodes for execution
            future_to_node = {
                executor.submit(self._execute_single_node, nodes[node_id], workflow_state): node_id
                for node_id in node_ids
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_node):
                node_id = future_to_node[future]
                try:
                    result = future.result()
                    results[node_id] = result
                except Exception as e:
                    results[node_id] = {
                        'success': False,
                        'error': str(e),
                        'tokens_used': 0,
                        'cost_usd': 0.0
                    }
        
        return results
    
    def _execute_single_node(self, node: WorkflowNode, workflow_state: Dict) -> Dict:
        """Execute a single workflow node"""
        node.status = 'running'
        node.started_at = datetime.utcnow()
        
        try:
            if node.type == 'agent':
                return self._execute_agent_node(node, workflow_state)
            elif node.type == 'condition':
                return self._execute_condition_node(node, workflow_state)
            elif node.type == 'action':
                return self._execute_action_node(node, workflow_state)
            elif node.type == 'input':
                return self._execute_input_node(node, workflow_state)
            elif node.type == 'output':
                return self._execute_output_node(node, workflow_state)
            else:
                raise ValueError(f"Unknown node type: {node.type}")
                
        except Exception as e:
            node.status = 'failed'
            node.error_message = str(e)
            node.completed_at = datetime.utcnow()
            
            return {
                'success': False,
                'error': str(e),
                'tokens_used': 0,
                'cost_usd': 0.0
            }
    
    def _execute_agent_node(self, node: WorkflowNode, workflow_state: Dict) -> Dict:
        """Execute an agent node"""
        start_time = time.time()
        
        # Get input data for the agent
        input_data = self._get_node_input_data(node, workflow_state)
        node.input_data = input_data
        
        # Execute the agent
        result = self.agent_engine.execute_agent(
            agent_config=node.config,
            input_data=input_data,
            context=workflow_state
        )
        
        # Update node state
        node.execution_time_ms = int((time.time() - start_time) * 1000)
        node.completed_at = datetime.utcnow()
        
        if result['success']:
            node.status = 'completed'
            node.output_data = result['output']
            workflow_state['node_outputs'][node.id] = result['output']
            
            # Add to execution log
            workflow_state['execution_log'].append({
                'timestamp': datetime.utcnow().isoformat(),
                'level': 'info',
                'message': f"Agent node {node.id} completed successfully",
                'node_id': node.id,
                'data': {
                    'execution_time_ms': node.execution_time_ms,
                    'tokens_used': result.get('tokens_used', 0),
                    'cost_usd': result.get('cost_usd', 0.0)
                }
            })
            
            return {
                'success': True,
                'output': result['output'],
                'tokens_used': result.get('tokens_used', 0),
                'cost_usd': result.get('cost_usd', 0.0)
            }
        else:
            node.status = 'failed'
            node.error_message = result.get('error', 'Unknown error')
            
            workflow_state['execution_log'].append({
                'timestamp': datetime.utcnow().isoformat(),
                'level': 'error',
                'message': f"Agent node {node.id} failed: {node.error_message}",
                'node_id': node.id
            })
            
            return {
                'success': False,
                'error': node.error_message,
                'tokens_used': 0,
                'cost_usd': 0.0
            }
    
    def _execute_condition_node(self, node: WorkflowNode, workflow_state: Dict) -> Dict:
        """Execute a condition node"""
        # Simple condition evaluation (expand for production)
        condition = node.config.get('condition', 'true')
        input_data = self._get_node_input_data(node, workflow_state)
        
        # Mock condition evaluation
        result = True  # Simplified for MVP
        
        node.status = 'completed'
        node.output_data = result
        node.completed_at = datetime.utcnow()
        workflow_state['node_outputs'][node.id] = result
        
        return {
            'success': True,
            'output': result,
            'tokens_used': 0,
            'cost_usd': 0.0
        }
    
    def _execute_action_node(self, node: WorkflowNode, workflow_state: Dict) -> Dict:
        """Execute an action node"""
        action_type = node.config.get('action_type', 'log')
        input_data = self._get_node_input_data(node, workflow_state)
        
        # Mock action execution
        if action_type == 'log':
            result = f"Logged: {input_data}"
        elif action_type == 'transform':
            result = f"Transformed: {input_data}"
        else:
            result = f"Action {action_type} executed with: {input_data}"
        
        node.status = 'completed'
        node.output_data = result
        node.completed_at = datetime.utcnow()
        workflow_state['node_outputs'][node.id] = result
        
        return {
            'success': True,
            'output': result,
            'tokens_used': 0,
            'cost_usd': 0.0
        }
    
    def _execute_input_node(self, node: WorkflowNode, workflow_state: Dict) -> Dict:
        """Execute an input node"""
        output_data = workflow_state['current_data']
        
        node.status = 'completed'
        node.output_data = output_data
        node.completed_at = datetime.utcnow()
        workflow_state['node_outputs'][node.id] = output_data
        
        return {
            'success': True,
            'output': output_data,
            'tokens_used': 0,
            'cost_usd': 0.0
        }
    
    def _execute_output_node(self, node: WorkflowNode, workflow_state: Dict) -> Dict:
        """Execute an output node"""
        input_data = self._get_node_input_data(node, workflow_state)
        
        node.status = 'completed'
        node.output_data = input_data
        node.completed_at = datetime.utcnow()
        workflow_state['node_outputs'][node.id] = input_data
        
        return {
            'success': True,
            'output': input_data,
            'tokens_used': 0,
            'cost_usd': 0.0
        }
    
    def _get_node_input_data(self, node: WorkflowNode, workflow_state: Dict) -> Any:
        """Get input data for a node based on its configuration"""
        input_config = node.config.get('input', {})
        
        if 'source_node_id' in input_config:
            source_node_id = input_config['source_node_id']
            return workflow_state['node_outputs'].get(source_node_id)
        else:
            return workflow_state['current_data']
    
    def _get_ready_nodes(self, dependencies: Dict, completed_nodes: set, 
                        failed_nodes: set, nodes: Dict) -> set:
        """Get nodes that are ready to execute"""
        ready_nodes = set()
        
        for node_id, deps in dependencies.items():
            if node_id in completed_nodes or node_id in failed_nodes:
                continue
            
            # Check if all dependencies are completed
            if all(dep_id in completed_nodes for dep_id in deps):
                ready_nodes.add(node_id)
        
        return ready_nodes
    
    def _collect_workflow_output(self, nodes: Dict[str, WorkflowNode], 
                                exit_points: List[str]) -> Any:
        """Collect final output from workflow exit points"""
        if len(exit_points) == 1:
            return nodes[exit_points[0]].output_data
        else:
            return {
                node_id: nodes[node_id].output_data 
                for node_id in exit_points
            }

