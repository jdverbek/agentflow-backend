from flask import Blueprint, jsonify, request
from src.models.user import db
from src.models.workflow import Workflow, WorkflowExecution, WorkflowTemplate
from src.engine.workflow_orchestrator import WorkflowOrchestrator
from datetime import datetime
import json

workflow_bp = Blueprint('workflow', __name__)
workflow_orchestrator = WorkflowOrchestrator()

@workflow_bp.route('/workflows', methods=['GET'])
def get_workflows():
    """Get all workflows for the current user"""
    # For MVP, we'll use user_id = 1 (in production, get from authentication)
    user_id = request.args.get('user_id', 1, type=int)
    
    workflows = Workflow.query.filter_by(user_id=user_id).all()
    return jsonify([workflow.to_dict() for workflow in workflows])

@workflow_bp.route('/workflows', methods=['POST'])
def create_workflow():
    """Create a new workflow"""
    data = request.json
    
    # Validate required fields
    if not data.get('name'):
        return jsonify({'error': 'Name is required'}), 400
    
    # For MVP, we'll use user_id = 1 (in production, get from authentication)
    user_id = data.get('user_id', 1)
    
    workflow = Workflow(
        name=data['name'],
        description=data.get('description', ''),
        nodes=data.get('nodes', []),
        connections=data.get('connections', []),
        configuration=data.get('configuration', {}),
        is_template=data.get('is_template', False),
        is_public=data.get('is_public', False),
        tags=data.get('tags', []),
        user_id=user_id
    )
    
    # Set default configuration if not provided
    if not workflow.configuration:
        workflow.configuration = workflow.get_default_configuration()
    
    db.session.add(workflow)
    db.session.commit()
    
    return jsonify(workflow.to_dict()), 201

@workflow_bp.route('/workflows/<int:workflow_id>', methods=['GET'])
def get_workflow(workflow_id):
    """Get a specific workflow"""
    workflow = Workflow.query.get_or_404(workflow_id)
    return jsonify(workflow.to_dict())

@workflow_bp.route('/workflows/<int:workflow_id>', methods=['PUT'])
def update_workflow(workflow_id):
    """Update a workflow"""
    workflow = Workflow.query.get_or_404(workflow_id)
    data = request.json
    
    # Update fields
    workflow.name = data.get('name', workflow.name)
    workflow.description = data.get('description', workflow.description)
    workflow.nodes = data.get('nodes', workflow.nodes)
    workflow.connections = data.get('connections', workflow.connections)
    workflow.configuration = data.get('configuration', workflow.configuration)
    workflow.is_template = data.get('is_template', workflow.is_template)
    workflow.is_public = data.get('is_public', workflow.is_public)
    workflow.tags = data.get('tags', workflow.tags)
    workflow.updated_at = datetime.utcnow()
    workflow.version += 1  # Increment version for collaboration
    
    db.session.commit()
    return jsonify(workflow.to_dict())

@workflow_bp.route('/workflows/<int:workflow_id>', methods=['DELETE'])
def delete_workflow(workflow_id):
    """Delete a workflow"""
    workflow = Workflow.query.get_or_404(workflow_id)
    db.session.delete(workflow)
    db.session.commit()
    return '', 204

@workflow_bp.route('/workflows/<int:workflow_id>/execute', methods=['POST'])
def execute_workflow(workflow_id):
    """Execute a workflow"""
    workflow = Workflow.query.get_or_404(workflow_id)
    data = request.json
    
    if not data or 'input' not in data:
        return jsonify({'error': 'Input data is required'}), 400
    
    input_data = data['input']
    user_id = data.get('user_id', 1)
    
    # Create execution record
    execution = WorkflowExecution(
        workflow_id=workflow.id,
        input_data=input_data,
        status='pending',
        user_id=user_id
    )
    
    db.session.add(execution)
    db.session.commit()
    
    # Build workflow definition
    workflow_definition = {
        'nodes': workflow.nodes,
        'connections': workflow.connections,
        'configuration': workflow.configuration
    }
    
    # Execute the workflow
    result = workflow_orchestrator.execute_workflow(
        workflow_definition=workflow_definition,
        input_data=input_data,
        execution_id=str(execution.id)
    )
    
    # Update execution record
    execution.status = 'completed' if result['success'] else 'failed'
    execution.output_data = result.get('output_data')
    execution.execution_log = result.get('execution_log', [])
    execution.error_message = result.get('error_message')
    execution.progress_percentage = 100 if result['success'] else execution.progress_percentage
    execution.total_execution_time_ms = result.get('total_execution_time_ms', 0)
    execution.total_tokens_used = result.get('total_tokens_used', 0)
    execution.total_cost_usd = result.get('total_cost_usd', 0.0)
    execution.completed_at = datetime.utcnow()
    
    db.session.commit()
    
    # Return result with execution details
    response = result.copy()
    response['execution'] = execution.to_dict()
    
    return jsonify(response)

@workflow_bp.route('/workflows/<int:workflow_id>/executions', methods=['GET'])
def get_workflow_executions(workflow_id):
    """Get execution history for a workflow"""
    workflow = Workflow.query.get_or_404(workflow_id)
    
    # Pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    executions = WorkflowExecution.query.filter_by(workflow_id=workflow_id)\
        .order_by(WorkflowExecution.started_at.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'executions': [execution.to_dict() for execution in executions.items],
        'total': executions.total,
        'pages': executions.pages,
        'current_page': page,
        'per_page': per_page
    })

@workflow_bp.route('/workflows/<int:workflow_id>/validate', methods=['POST'])
def validate_workflow(workflow_id):
    """Validate a workflow structure"""
    workflow = Workflow.query.get_or_404(workflow_id)
    
    validation_result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Basic validation
    nodes = workflow.nodes
    connections = workflow.connections
    
    # Check for required nodes
    if not nodes:
        validation_result['valid'] = False
        validation_result['errors'].append('Workflow must have at least one node')
    
    # Check for orphaned nodes
    node_ids = {node['id'] for node in nodes}
    connected_nodes = set()
    
    for connection in connections:
        connected_nodes.add(connection['source_id'])
        connected_nodes.add(connection['target_id'])
    
    orphaned_nodes = node_ids - connected_nodes
    if orphaned_nodes and len(nodes) > 1:
        validation_result['warnings'].append(f'Orphaned nodes found: {list(orphaned_nodes)}')
    
    # Check for invalid connections
    for connection in connections:
        if connection['source_id'] not in node_ids:
            validation_result['valid'] = False
            validation_result['errors'].append(f'Invalid source node: {connection["source_id"]}')
        
        if connection['target_id'] not in node_ids:
            validation_result['valid'] = False
            validation_result['errors'].append(f'Invalid target node: {connection["target_id"]}')
    
    # Check for circular dependencies (simplified)
    # In production, implement proper cycle detection
    
    return jsonify(validation_result)

@workflow_bp.route('/workflows/<int:workflow_id>/duplicate', methods=['POST'])
def duplicate_workflow(workflow_id):
    """Duplicate a workflow"""
    original_workflow = Workflow.query.get_or_404(workflow_id)
    data = request.json
    
    user_id = data.get('user_id', 1)
    new_name = data.get('name', f"{original_workflow.name} (Copy)")
    
    # Create duplicate
    duplicate = Workflow(
        name=new_name,
        description=original_workflow.description,
        nodes=original_workflow.nodes.copy(),
        connections=original_workflow.connections.copy(),
        configuration=original_workflow.configuration.copy(),
        tags=original_workflow.tags.copy(),
        user_id=user_id
    )
    
    db.session.add(duplicate)
    db.session.commit()
    
    return jsonify(duplicate.to_dict()), 201

@workflow_bp.route('/executions/<int:execution_id>', methods=['GET'])
def get_execution(execution_id):
    """Get a specific workflow execution"""
    execution = WorkflowExecution.query.get_or_404(execution_id)
    return jsonify(execution.to_dict())

@workflow_bp.route('/executions/<int:execution_id>/cancel', methods=['POST'])
def cancel_execution(execution_id):
    """Cancel a running workflow execution"""
    execution = WorkflowExecution.query.get_or_404(execution_id)
    
    if execution.status in ['pending', 'running']:
        execution.status = 'cancelled'
        execution.completed_at = datetime.utcnow()
        execution.add_log_entry('info', 'Execution cancelled by user')
        
        db.session.commit()
        
        return jsonify({'message': 'Execution cancelled successfully'})
    else:
        return jsonify({'error': 'Cannot cancel execution in current state'}), 400

@workflow_bp.route('/templates', methods=['GET'])
def get_workflow_templates():
    """Get workflow templates"""
    category = request.args.get('category')
    difficulty = request.args.get('difficulty')
    
    query = WorkflowTemplate.query
    
    if category:
        query = query.filter_by(category=category)
    
    if difficulty:
        query = query.filter_by(difficulty_level=difficulty)
    
    templates = query.order_by(WorkflowTemplate.usage_count.desc()).all()
    
    return jsonify([template.to_dict() for template in templates])

@workflow_bp.route('/templates', methods=['POST'])
def create_workflow_template():
    """Create a workflow template"""
    data = request.json
    
    # Validate required fields
    if not data.get('name') or not data.get('workflow_definition'):
        return jsonify({'error': 'Name and workflow_definition are required'}), 400
    
    user_id = data.get('created_by_user_id', 1)
    
    template = WorkflowTemplate(
        name=data['name'],
        description=data.get('description', ''),
        category=data.get('category', 'general'),
        difficulty_level=data.get('difficulty_level', 'beginner'),
        workflow_definition=data['workflow_definition'],
        preview_image_url=data.get('preview_image_url'),
        tags=data.get('tags', []),
        created_by_user_id=user_id
    )
    
    db.session.add(template)
    db.session.commit()
    
    return jsonify(template.to_dict()), 201

@workflow_bp.route('/templates/<int:template_id>/use', methods=['POST'])
def create_workflow_from_template(template_id):
    """Create a workflow from a template"""
    template = WorkflowTemplate.query.get_or_404(template_id)
    data = request.json
    
    user_id = data.get('user_id', 1)
    workflow_name = data.get('name', template.name)
    
    # Create workflow from template
    workflow = Workflow(
        name=workflow_name,
        description=template.description,
        nodes=template.workflow_definition.get('nodes', []),
        connections=template.workflow_definition.get('connections', []),
        configuration=template.workflow_definition.get('configuration', {}),
        tags=template.tags.copy(),
        user_id=user_id
    )
    
    db.session.add(workflow)
    
    # Increment template usage count
    template.usage_count += 1
    
    db.session.commit()
    
    return jsonify(workflow.to_dict()), 201

@workflow_bp.route('/workflows/sample', methods=['GET'])
def get_sample_workflows():
    """Get sample workflow definitions for testing"""
    samples = [
        {
            'name': 'Simple Research Workflow',
            'description': 'A basic workflow that researches a topic and summarizes findings',
            'nodes': [
                {
                    'id': 'input_1',
                    'type': 'input',
                    'config': {
                        'name': 'Research Topic Input',
                        'description': 'Input node for research topic'
                    }
                },
                {
                    'id': 'agent_1',
                    'type': 'agent',
                    'config': {
                        'name': 'Research Agent',
                        'role': 'Research Assistant specialized in finding information',
                        'goal': 'Find comprehensive information about the given topic',
                        'tools': ['web_search', 'data_analyzer'],
                        'memory_config': {'working_memory_size': 10},
                        'behavioral_params': {'creativity_level': 0.3, 'reasoning_transparency': True}
                    }
                },
                {
                    'id': 'agent_2',
                    'type': 'agent',
                    'config': {
                        'name': 'Summary Agent',
                        'role': 'Content Summarizer specialized in creating concise summaries',
                        'goal': 'Create a clear, concise summary of research findings',
                        'tools': ['text_processor'],
                        'memory_config': {'working_memory_size': 8},
                        'behavioral_params': {'creativity_level': 0.5, 'output_verbosity': 'concise'}
                    }
                },
                {
                    'id': 'output_1',
                    'type': 'output',
                    'config': {
                        'name': 'Final Summary Output',
                        'description': 'Output node for final summary'
                    }
                }
            ],
            'connections': [
                {'source_id': 'input_1', 'target_id': 'agent_1'},
                {'source_id': 'agent_1', 'target_id': 'agent_2'},
                {'source_id': 'agent_2', 'target_id': 'output_1'}
            ],
            'configuration': {
                'max_execution_time_minutes': 15,
                'parallel_execution_enabled': False,
                'error_handling_strategy': 'stop_on_error'
            }
        },
        {
            'name': 'Content Creation Pipeline',
            'description': 'A workflow for creating and reviewing content',
            'nodes': [
                {
                    'id': 'input_1',
                    'type': 'input',
                    'config': {
                        'name': 'Content Brief Input',
                        'description': 'Input node for content requirements'
                    }
                },
                {
                    'id': 'agent_1',
                    'type': 'agent',
                    'config': {
                        'name': 'Content Creator',
                        'role': 'Creative Content Creator specialized in writing',
                        'goal': 'Create engaging content based on the brief',
                        'tools': ['text_processor', 'web_search'],
                        'memory_config': {'working_memory_size': 12},
                        'behavioral_params': {'creativity_level': 0.8, 'output_verbosity': 'detailed'}
                    }
                },
                {
                    'id': 'agent_2',
                    'type': 'agent',
                    'config': {
                        'name': 'Content Reviewer',
                        'role': 'Content Reviewer focused on quality and accuracy',
                        'goal': 'Review and improve content quality',
                        'tools': ['text_processor'],
                        'memory_config': {'working_memory_size': 10},
                        'behavioral_params': {'creativity_level': 0.4, 'reasoning_transparency': True}
                    }
                },
                {
                    'id': 'output_1',
                    'type': 'output',
                    'config': {
                        'name': 'Final Content Output',
                        'description': 'Output node for reviewed content'
                    }
                }
            ],
            'connections': [
                {'source_id': 'input_1', 'target_id': 'agent_1'},
                {'source_id': 'agent_1', 'target_id': 'agent_2'},
                {'source_id': 'agent_2', 'target_id': 'output_1'}
            ],
            'configuration': {
                'max_execution_time_minutes': 20,
                'parallel_execution_enabled': False,
                'error_handling_strategy': 'continue_on_error'
            }
        }
    ]
    
    return jsonify(samples)

