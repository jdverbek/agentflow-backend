from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
from src.models.user import db

class Agent(db.Model):
    """
    Agent model representing an AI agent with role, goals, tools, and memory configuration.
    Based on AgentFlow design specification.
    """
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    role = db.Column(db.Text, nullable=False)  # Natural language role description
    goal = db.Column(db.Text, nullable=False)  # Natural language goal specification
    tools = db.Column(db.JSON, default=list)  # List of tool IDs/configurations
    memory_config = db.Column(db.JSON, default=dict)  # Memory configuration settings
    behavioral_params = db.Column(db.JSON, default=dict)  # Creativity, risk tolerance, etc.
    llm_provider = db.Column(db.String(50), default='openai')  # LLM provider
    llm_model = db.Column(db.String(100), default='gpt-3.5-turbo')  # Specific model
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Relationships
    user = db.relationship('User', backref=db.backref('agents', lazy=True))
    
    def __repr__(self):
        return f'<Agent {self.name}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'role': self.role,
            'goal': self.goal,
            'tools': self.tools,
            'memory_config': self.memory_config,
            'behavioral_params': self.behavioral_params,
            'llm_provider': self.llm_provider,
            'llm_model': self.llm_model,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'user_id': self.user_id
        }
    
    def get_default_memory_config(self):
        """Return default memory configuration"""
        return {
            'working_memory_size': 10,
            'session_memory_enabled': True,
            'long_term_memory_enabled': False,
            'memory_retention_days': 30
        }
    
    def get_default_behavioral_params(self):
        """Return default behavioral parameters"""
        return {
            'creativity_level': 0.7,
            'risk_tolerance': 0.5,
            'output_verbosity': 'medium',
            'reasoning_transparency': True
        }


class AgentExecution(db.Model):
    """
    Model to track individual agent executions within workflows
    """
    id = db.Column(db.Integer, primary_key=True)
    agent_id = db.Column(db.Integer, db.ForeignKey('agent.id'), nullable=False)
    workflow_execution_id = db.Column(db.Integer, db.ForeignKey('workflow_execution.id'), nullable=False)
    input_data = db.Column(db.JSON)
    output_data = db.Column(db.JSON)
    reasoning_log = db.Column(db.JSON, default=list)  # Step-by-step reasoning
    tool_calls = db.Column(db.JSON, default=list)  # Tools called during execution
    status = db.Column(db.String(20), default='pending')  # pending, running, completed, failed
    error_message = db.Column(db.Text)
    execution_time_ms = db.Column(db.Integer)
    tokens_used = db.Column(db.Integer, default=0)
    cost_usd = db.Column(db.Float, default=0.0)
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    
    # Relationships
    agent = db.relationship('Agent', backref=db.backref('executions', lazy=True))
    
    def __repr__(self):
        return f'<AgentExecution {self.id} - Agent {self.agent_id}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'agent_id': self.agent_id,
            'workflow_execution_id': self.workflow_execution_id,
            'input_data': self.input_data,
            'output_data': self.output_data,
            'reasoning_log': self.reasoning_log,
            'tool_calls': self.tool_calls,
            'status': self.status,
            'error_message': self.error_message,
            'execution_time_ms': self.execution_time_ms,
            'tokens_used': self.tokens_used,
            'cost_usd': self.cost_usd,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }

