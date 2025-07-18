from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
from src.models.user import db

class Tool(db.Model):
    """
    Tool model representing available tools that agents can use.
    Based on AgentFlow design specification.
    """
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    display_name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50))  # e.g., 'web_search', 'data_processing', 'communication'
    input_schema = db.Column(db.JSON, nullable=False)  # JSON schema for input validation
    output_schema = db.Column(db.JSON, nullable=False)  # JSON schema for output validation
    configuration_schema = db.Column(db.JSON, default=dict)  # Schema for tool configuration
    implementation_type = db.Column(db.String(20), default='python')  # python, api, webhook
    implementation_code = db.Column(db.Text)  # Python code for the tool
    api_endpoint = db.Column(db.String(255))  # API endpoint if implementation_type is 'api'
    authentication_required = db.Column(db.Boolean, default=False)
    authentication_schema = db.Column(db.JSON)  # Schema for authentication parameters
    is_system_tool = db.Column(db.Boolean, default=False)  # Whether this is a built-in system tool
    is_public = db.Column(db.Boolean, default=False)  # Whether this tool is publicly available
    usage_count = db.Column(db.Integer, default=0)
    rating = db.Column(db.Float, default=0.0)
    tags = db.Column(db.JSON, default=list)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by_user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    
    # Relationships
    created_by = db.relationship('User', backref=db.backref('created_tools', lazy=True))
    
    def __repr__(self):
        return f'<Tool {self.name}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'category': self.category,
            'input_schema': self.input_schema,
            'output_schema': self.output_schema,
            'configuration_schema': self.configuration_schema,
            'implementation_type': self.implementation_type,
            'implementation_code': self.implementation_code if not self.is_system_tool else None,
            'api_endpoint': self.api_endpoint,
            'authentication_required': self.authentication_required,
            'authentication_schema': self.authentication_schema,
            'is_system_tool': self.is_system_tool,
            'is_public': self.is_public,
            'usage_count': self.usage_count,
            'rating': self.rating,
            'tags': self.tags,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'created_by_user_id': self.created_by_user_id
        }
    
    def get_basic_info(self):
        """Return basic tool information for agent configuration"""
        return {
            'id': self.id,
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'category': self.category,
            'input_schema': self.input_schema,
            'output_schema': self.output_schema,
            'authentication_required': self.authentication_required
        }


class ToolExecution(db.Model):
    """
    Model to track tool executions and their performance
    """
    id = db.Column(db.Integer, primary_key=True)
    tool_id = db.Column(db.Integer, db.ForeignKey('tool.id'), nullable=False)
    agent_execution_id = db.Column(db.Integer, db.ForeignKey('agent_execution.id'), nullable=False)
    input_data = db.Column(db.JSON)
    output_data = db.Column(db.JSON)
    status = db.Column(db.String(20), default='pending')  # pending, running, completed, failed
    error_message = db.Column(db.Text)
    execution_time_ms = db.Column(db.Integer)
    cost_usd = db.Column(db.Float, default=0.0)
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    
    # Relationships
    tool = db.relationship('Tool', backref=db.backref('executions', lazy=True))
    
    def __repr__(self):
        return f'<ToolExecution {self.id} - Tool {self.tool_id}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'tool_id': self.tool_id,
            'agent_execution_id': self.agent_execution_id,
            'input_data': self.input_data,
            'output_data': self.output_data,
            'status': self.status,
            'error_message': self.error_message,
            'execution_time_ms': self.execution_time_ms,
            'cost_usd': self.cost_usd,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }


class UserToolConfiguration(db.Model):
    """
    Model to store user-specific tool configurations and authentication
    """
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    tool_id = db.Column(db.Integer, db.ForeignKey('tool.id'), nullable=False)
    configuration = db.Column(db.JSON, default=dict)  # Tool-specific configuration
    authentication_data = db.Column(db.JSON)  # Encrypted authentication data
    is_enabled = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = db.relationship('User', backref=db.backref('tool_configurations', lazy=True))
    tool = db.relationship('Tool', backref=db.backref('user_configurations', lazy=True))
    
    # Unique constraint to prevent duplicate configurations
    __table_args__ = (db.UniqueConstraint('user_id', 'tool_id', name='unique_user_tool_config'),)
    
    def __repr__(self):
        return f'<UserToolConfiguration User {self.user_id} - Tool {self.tool_id}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'tool_id': self.tool_id,
            'configuration': self.configuration,
            'is_enabled': self.is_enabled,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

