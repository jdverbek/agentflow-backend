"""
Orchestration API Routes - Manager Agent Endpoints
Handles complex task delegation and multi-agent coordination
"""

from flask import Blueprint, request, jsonify
import time
import threading
from src.orchestration.manager_agent import manager_agent, TaskStatus

orchestration_bp = Blueprint('orchestration', __name__)

# Store for async task execution
async_executions = {}

@orchestration_bp.route('/api/orchestration/execute', methods=['POST'])
def execute_complex_task():
    """
    Execute a complex task using multi-agent orchestration
    """
    try:
        data = request.get_json()
        task = data.get('task', '')
        max_iterations = data.get('max_iterations', 5)
        async_execution = data.get('async', False)
        
        if not task:
            return jsonify({'error': 'Task description is required'}), 400
        
        if async_execution:
            # Start async execution
            execution_id = _start_async_execution(task, max_iterations)
            return jsonify({
                'execution_id': execution_id,
                'status': 'started',
                'message': 'Task execution started. Use /status endpoint to monitor progress.',
                'async': True
            })
        else:
            # Synchronous execution
            start_time = time.time()
            result = manager_agent.execute_task(task, max_iterations)
            duration = time.time() - start_time
            
            return jsonify({
                'result': result,
                'duration': round(duration, 2),
                'task': task,
                'max_iterations': max_iterations,
                'async': False
            })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@orchestration_bp.route('/api/orchestration/status/<execution_id>', methods=['GET'])
def get_execution_status(execution_id):
    """
    Get the status of an ongoing task execution
    """
    try:
        # Check async executions first
        if execution_id in async_executions:
            async_result = async_executions[execution_id]
            return jsonify(async_result)
        
        # Check manager agent executions
        status = manager_agent.get_execution_status(execution_id)
        if status:
            return jsonify(status)
        
        return jsonify({'error': 'Execution not found'}), 404
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@orchestration_bp.route('/api/orchestration/capabilities', methods=['GET'])
def get_agent_capabilities():
    """
    Get information about available agents and their capabilities
    """
    try:
        capabilities = {}
        
        for agent_type, config in manager_agent.agent_capabilities.items():
            capabilities[agent_type.value] = {
                'name': agent_type.value.replace('_', ' ').title(),
                'models': config['models'],
                'tools': config['tools'],
                'strengths': config['strengths'],
                'use_cases': config['use_cases']
            }
        
        return jsonify({
            'agents': capabilities,
            'orchestration_features': [
                'Intelligent task delegation',
                'Multi-agent coordination',
                'Quality feedback loops',
                'Iterative improvement',
                'Specialized model selection'
            ]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@orchestration_bp.route('/api/orchestration/examples', methods=['GET'])
def get_task_examples():
    """
    Get example tasks that demonstrate the orchestration system
    """
    try:
        examples = [
            {
                'category': 'Market Research',
                'task': 'Research the AI automation market and create a comprehensive report with data analysis and PowerPoint presentation',
                'expected_agents': ['research_analyst', 'data_analyst', 'execution_agent', 'oversight_manager'],
                'estimated_duration': '5-10 minutes',
                'deliverables': ['Research findings', 'Data analysis', 'PPT presentation', 'PDF report']
            },
            {
                'category': 'Content Strategy',
                'task': 'Analyze competitor content strategies and create a content marketing plan with blog posts and social media content',
                'expected_agents': ['research_analyst', 'content_creator', 'oversight_manager'],
                'estimated_duration': '3-7 minutes',
                'deliverables': ['Competitor analysis', 'Content strategy', 'Blog posts', 'Social media content']
            },
            {
                'category': 'Business Analysis',
                'task': 'Analyze our Q4 performance data and create executive dashboard with recommendations and presentation',
                'expected_agents': ['data_analyst', 'content_creator', 'execution_agent', 'oversight_manager'],
                'estimated_duration': '4-8 minutes',
                'deliverables': ['Data analysis', 'Executive dashboard', 'Recommendations', 'Presentation']
            },
            {
                'category': 'Product Launch',
                'task': 'Research target market, create launch strategy, develop marketing materials, and build launch presentation',
                'expected_agents': ['research_analyst', 'content_creator', 'data_analyst', 'execution_agent', 'oversight_manager'],
                'estimated_duration': '7-12 minutes',
                'deliverables': ['Market research', 'Launch strategy', 'Marketing materials', 'Launch presentation']
            }
        ]
        
        return jsonify({
            'examples': examples,
            'usage_tips': [
                'Be specific about desired deliverables (PPT, PDF, reports)',
                'Mention if you need research, analysis, or content creation',
                'Specify target audience or use case for better results',
                'The system will automatically select the best agents for your task'
            ]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@orchestration_bp.route('/api/orchestration/active', methods=['GET'])
def get_active_executions():
    """
    Get all currently active task executions
    """
    try:
        active_executions = []
        
        # Get manager agent executions
        for execution_id, execution in manager_agent.active_executions.items():
            if execution.status in [TaskStatus.IN_PROGRESS, TaskStatus.PENDING]:
                status_info = manager_agent.get_execution_status(execution_id)
                if status_info:
                    active_executions.append(status_info)
        
        # Get async executions
        for execution_id, async_result in async_executions.items():
            if async_result.get('status') in ['started', 'in_progress']:
                active_executions.append({
                    'id': execution_id,
                    'type': 'async',
                    'status': async_result.get('status'),
                    'task': async_result.get('task', 'Unknown'),
                    'started_at': async_result.get('started_at')
                })
        
        return jsonify({
            'active_executions': active_executions,
            'count': len(active_executions)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _start_async_execution(task: str, max_iterations: int) -> str:
    """
    Start an asynchronous task execution
    """
    execution_id = f"async_{int(time.time() * 1000)}"
    
    # Initialize async execution record
    async_executions[execution_id] = {
        'id': execution_id,
        'task': task,
        'status': 'started',
        'started_at': time.time(),
        'max_iterations': max_iterations,
        'result': None,
        'error': None
    }
    
    # Start execution in background thread
    thread = threading.Thread(
        target=_execute_async_task,
        args=(execution_id, task, max_iterations)
    )
    thread.daemon = True
    thread.start()
    
    return execution_id

def _execute_async_task(execution_id: str, task: str, max_iterations: int):
    """
    Execute a task asynchronously in background thread
    """
    try:
        async_executions[execution_id]['status'] = 'in_progress'
        
        # Execute the task
        result = manager_agent.execute_task(task, max_iterations)
        
        # Update async execution record
        async_executions[execution_id].update({
            'status': 'completed',
            'result': result,
            'completed_at': time.time(),
            'duration': time.time() - async_executions[execution_id]['started_at']
        })
        
    except Exception as e:
        async_executions[execution_id].update({
            'status': 'failed',
            'error': str(e),
            'completed_at': time.time()
        })

# Example usage endpoint for testing
@orchestration_bp.route('/api/orchestration/demo', methods=['POST'])
def demo_orchestration():
    """
    Demo endpoint to showcase the orchestration system
    """
    try:
        demo_task = "Research the latest AI trends, analyze market data, create a comprehensive report, and build a PowerPoint presentation with key insights and recommendations"
        
        start_time = time.time()
        result = manager_agent.execute_task(demo_task, max_iterations=3)
        duration = time.time() - start_time
        
        return jsonify({
            'demo_completed': True,
            'task': demo_task,
            'result': result,
            'duration': round(duration, 2),
            'agents_used': ['research_analyst', 'data_analyst', 'content_creator', 'execution_agent', 'oversight_manager'],
            'message': 'Demo completed successfully! This shows how the Manager Agent coordinates multiple specialized agents to complete complex tasks.'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

