from flask import Blueprint, jsonify, request
from src.models.user import db
from src.models.agent import Agent, AgentExecution
from src.engine.agent_engine import AgentEngine
from datetime import datetime

agent_bp = Blueprint('agent', __name__)
agent_engine = AgentEngine()

@agent_bp.route('/agents', methods=['GET'])
def get_agents():
    """Get all agents for the current user"""
    # For MVP, we'll use user_id = 1 (in production, get from authentication)
    user_id = request.args.get('user_id', 1, type=int)
    
    agents = Agent.query.filter_by(user_id=user_id).all()
    return jsonify([agent.to_dict() for agent in agents])

@agent_bp.route('/agents', methods=['POST'])
def create_agent():
    """Create a new agent"""
    data = request.json
    
    # Validate required fields
    if not data.get('name') or not data.get('role') or not data.get('goal'):
        return jsonify({'error': 'Name, role, and goal are required'}), 400
    
    # For MVP, we'll use user_id = 1 (in production, get from authentication)
    user_id = data.get('user_id', 1)
    
    agent = Agent(
        name=data['name'],
        role=data['role'],
        goal=data['goal'],
        tools=data.get('tools', []),
        memory_config=data.get('memory_config', {}),
        behavioral_params=data.get('behavioral_params', {}),
        llm_provider=data.get('llm_provider', 'openai'),
        llm_model=data.get('llm_model', 'gpt-3.5-turbo'),
        user_id=user_id
    )
    
    # Set default configurations if not provided
    if not agent.memory_config:
        agent.memory_config = agent.get_default_memory_config()
    
    if not agent.behavioral_params:
        agent.behavioral_params = agent.get_default_behavioral_params()
    
    db.session.add(agent)
    db.session.commit()
    
    return jsonify(agent.to_dict()), 201

@agent_bp.route('/agents/<int:agent_id>', methods=['GET'])
def get_agent(agent_id):
    """Get a specific agent"""
    agent = Agent.query.get_or_404(agent_id)
    return jsonify(agent.to_dict())

@agent_bp.route('/agents/<int:agent_id>', methods=['PUT'])
def update_agent(agent_id):
    """Update an agent"""
    agent = Agent.query.get_or_404(agent_id)
    data = request.json
    
    # Update fields
    agent.name = data.get('name', agent.name)
    agent.role = data.get('role', agent.role)
    agent.goal = data.get('goal', agent.goal)
    agent.tools = data.get('tools', agent.tools)
    agent.memory_config = data.get('memory_config', agent.memory_config)
    agent.behavioral_params = data.get('behavioral_params', agent.behavioral_params)
    agent.llm_provider = data.get('llm_provider', agent.llm_provider)
    agent.llm_model = data.get('llm_model', agent.llm_model)
    agent.updated_at = datetime.utcnow()
    
    db.session.commit()
    return jsonify(agent.to_dict())

@agent_bp.route('/agents/<int:agent_id>', methods=['DELETE'])
def delete_agent(agent_id):
    """Delete an agent"""
    agent = Agent.query.get_or_404(agent_id)
    db.session.delete(agent)
    db.session.commit()
    return '', 204

@agent_bp.route('/agents/<int:agent_id>/execute', methods=['POST'])
def execute_agent(agent_id):
    """Execute an agent with given input"""
    agent = Agent.query.get_or_404(agent_id)
    data = request.json
    
    if not data or 'input' not in data:
        return jsonify({'error': 'Input data is required'}), 400
    
    input_data = data['input']
    context = data.get('context', {})
    
    # Build agent configuration
    agent_config = {
        'name': agent.name,
        'role': agent.role,
        'goal': agent.goal,
        'tools': agent.tools,
        'memory_config': agent.memory_config,
        'behavioral_params': agent.behavioral_params,
        'llm_provider': agent.llm_provider,
        'llm_model': agent.llm_model
    }
    
    # Execute the agent
    result = agent_engine.execute_agent(agent_config, input_data, context)
    
    # Create execution record
    execution = AgentExecution(
        agent_id=agent.id,
        workflow_execution_id=data.get('workflow_execution_id'),
        input_data=input_data,
        output_data=result.get('output') if result['success'] else None,
        reasoning_log=result.get('reasoning_log', []),
        tool_calls=result.get('tool_calls', []),
        status='completed' if result['success'] else 'failed',
        error_message=result.get('error') if not result['success'] else None,
        execution_time_ms=result.get('execution_time_ms', 0),
        tokens_used=result.get('tokens_used', 0),
        cost_usd=result.get('cost_usd', 0.0),
        completed_at=datetime.utcnow() if result['success'] else None
    )
    
    db.session.add(execution)
    db.session.commit()
    
    # Return result with execution ID
    response = result.copy()
    response['execution_id'] = execution.id
    
    return jsonify(response)

@agent_bp.route('/agents/<int:agent_id>/executions', methods=['GET'])
def get_agent_executions(agent_id):
    """Get execution history for an agent"""
    agent = Agent.query.get_or_404(agent_id)
    
    # Pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    executions = AgentExecution.query.filter_by(agent_id=agent_id)\
        .order_by(AgentExecution.started_at.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'executions': [execution.to_dict() for execution in executions.items],
        'total': executions.total,
        'pages': executions.pages,
        'current_page': page,
        'per_page': per_page
    })

@agent_bp.route('/agents/<int:agent_id>/test', methods=['POST'])
def test_agent(agent_id):
    """Test an agent with sample input"""
    agent = Agent.query.get_or_404(agent_id)
    data = request.json
    
    # Use default test input if not provided
    test_input = data.get('input', 'Hello, please introduce yourself and explain what you can do.')
    
    # Build agent configuration
    agent_config = {
        'name': agent.name,
        'role': agent.role,
        'goal': agent.goal,
        'tools': agent.tools,
        'memory_config': agent.memory_config,
        'behavioral_params': agent.behavioral_params,
        'llm_provider': agent.llm_provider,
        'llm_model': agent.llm_model
    }
    
    # Execute the agent (don't save to database for tests)
    result = agent_engine.execute_agent(agent_config, test_input, {})
    
    return jsonify({
        'test_input': test_input,
        'result': result,
        'agent_config': agent_config
    })

@agent_bp.route('/agents/templates', methods=['GET'])
def get_agent_templates():
    """Get predefined agent templates"""
    templates = [
        {
            'id': 'research_assistant',
            'name': 'Research Assistant',
            'role': 'Research Assistant specialized in finding and analyzing information',
            'goal': 'Help users find accurate, relevant information and provide comprehensive analysis',
            'tools': ['web_search', 'data_analyzer', 'text_processor'],
            'memory_config': {
                'working_memory_size': 15,
                'session_memory_enabled': True,
                'long_term_memory_enabled': True
            },
            'behavioral_params': {
                'creativity_level': 0.3,
                'risk_tolerance': 0.2,
                'output_verbosity': 'detailed',
                'reasoning_transparency': True
            },
            'description': 'Perfect for research tasks, data gathering, and information analysis',
            'use_cases': ['Academic research', 'Market analysis', 'Fact checking']
        },
        {
            'id': 'content_creator',
            'name': 'Content Creator',
            'role': 'Creative Content Creator specialized in writing and content generation',
            'goal': 'Create engaging, high-quality content tailored to specific audiences and purposes',
            'tools': ['text_processor', 'web_search'],
            'memory_config': {
                'working_memory_size': 12,
                'session_memory_enabled': True,
                'long_term_memory_enabled': False
            },
            'behavioral_params': {
                'creativity_level': 0.8,
                'risk_tolerance': 0.6,
                'output_verbosity': 'medium',
                'reasoning_transparency': False
            },
            'description': 'Ideal for creating blog posts, articles, social media content, and marketing copy',
            'use_cases': ['Blog writing', 'Social media posts', 'Marketing content']
        },
        {
            'id': 'data_analyst',
            'name': 'Data Analyst',
            'role': 'Data Analyst specialized in processing and interpreting data',
            'goal': 'Analyze data to extract meaningful insights and provide actionable recommendations',
            'tools': ['data_analyzer', 'calculator', 'text_processor'],
            'memory_config': {
                'working_memory_size': 20,
                'session_memory_enabled': True,
                'long_term_memory_enabled': True
            },
            'behavioral_params': {
                'creativity_level': 0.4,
                'risk_tolerance': 0.3,
                'output_verbosity': 'detailed',
                'reasoning_transparency': True
            },
            'description': 'Specialized in data processing, statistical analysis, and insight generation',
            'use_cases': ['Data analysis', 'Report generation', 'Trend identification']
        },
        {
            'id': 'customer_support',
            'name': 'Customer Support Agent',
            'role': 'Customer Support Agent focused on helping users solve problems',
            'goal': 'Provide helpful, accurate, and empathetic customer support',
            'tools': ['web_search', 'text_processor'],
            'memory_config': {
                'working_memory_size': 10,
                'session_memory_enabled': True,
                'long_term_memory_enabled': False
            },
            'behavioral_params': {
                'creativity_level': 0.5,
                'risk_tolerance': 0.2,
                'output_verbosity': 'concise',
                'reasoning_transparency': False
            },
            'description': 'Designed to handle customer inquiries with empathy and efficiency',
            'use_cases': ['Customer support', 'FAQ responses', 'Problem solving']
        }
    ]
    
    return jsonify(templates)

@agent_bp.route('/agents/from-template', methods=['POST'])
def create_agent_from_template(template_id=None):
    """Create an agent from a template"""
    data = request.json
    template_id = data.get('template_id')
    
    if not template_id:
        return jsonify({'error': 'Template ID is required'}), 400
    
    # Get templates (in production, this would be from database)
    templates_response = get_agent_templates()
    templates = templates_response.get_json()
    
    template = next((t for t in templates if t['id'] == template_id), None)
    if not template:
        return jsonify({'error': 'Template not found'}), 404
    
    # Create agent from template
    user_id = data.get('user_id', 1)
    agent_name = data.get('name', template['name'])
    
    agent = Agent(
        name=agent_name,
        role=template['role'],
        goal=template['goal'],
        tools=template['tools'],
        memory_config=template['memory_config'],
        behavioral_params=template['behavioral_params'],
        llm_provider=data.get('llm_provider', 'openai'),
        llm_model=data.get('llm_model', 'gpt-3.5-turbo'),
        user_id=user_id
    )
    
    db.session.add(agent)
    db.session.commit()
    
    return jsonify(agent.to_dict()), 201

