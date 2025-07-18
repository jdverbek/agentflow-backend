from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
from src.models.user import db

class Workflow(db.Model):
    """
    Workflow model representing a complete agent workflow with nodes and connections.
    Based on AgentFlow design specification.
    """
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    nodes = db.Column(db.JSON, default=list)  # List of workflow nodes (agents, conditions, etc.)
    connections = db.Column(db.JSON, default=list)  # List of connections between nodes
    configuration = db.Column(db.JSON, default=dict)  # Workflow-level configuration
    version = db.Column(db.Integer, default=1)  # Version for collaboration features
    is_template = db.Column(db.Boolean, default=False)  # Whether this is a template
    is_public = db.Column(db.Boolean, default=False)  # Whether this is publicly shared
    tags = db.Column(db.JSON, default=list)  # Tags for categorization
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Relationships
    user = db.relationship('User', backref=db.backref('workflows', lazy=True))
    
    def __repr__(self):
        return f'<Workflow {self.name}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'nodes': self.nodes,
            'connections': self.connections,
            'configuration': self.configuration,
            'version': self.version,
            'is_template': self.is_template,
            'is_public': self.is_public,
            'tags': self.tags,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'user_id': self.user_id
        }
    
    def get_default_configuration(self):
        """Return default workflow configuration"""
        return {
            'max_execution_time_minutes': 30,
            'retry_failed_nodes': True,
            'max_retries': 3,
            'parallel_execution_enabled': True,
            'error_handling_strategy': 'stop_on_error'
        }


class WorkflowExecution(db.Model):
    """
    Model to track workflow executions and their results
    """
    id = db.Column(db.Integer, primary_key=True)
    workflow_id = db.Column(db.Integer, db.ForeignKey('workflow.id'), nullable=False)
    input_data = db.Column(db.JSON)  # Initial input to the workflow
    output_data = db.Column(db.JSON)  # Final output from the workflow
    execution_log = db.Column(db.JSON, default=list)  # Detailed execution log
    status = db.Column(db.String(20), default='pending')  # pending, running, completed, failed, cancelled
    error_message = db.Column(db.Text)
    progress_percentage = db.Column(db.Integer, default=0)
    current_node_id = db.Column(db.String(50))  # Currently executing node
    total_execution_time_ms = db.Column(db.Integer)
    total_tokens_used = db.Column(db.Integer, default=0)
    total_cost_usd = db.Column(db.Float, default=0.0)
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Relationships
    workflow = db.relationship('Workflow', backref=db.backref('executions', lazy=True))
    user = db.relationship('User', backref=db.backref('workflow_executions', lazy=True))
    
    def __repr__(self):
        return f'<WorkflowExecution {self.id} - Workflow {self.workflow_id}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'workflow_id': self.workflow_id,
            'input_data': self.input_data,
            'output_data': self.output_data,
            'execution_log': self.execution_log,
            'status': self.status,
            'error_message': self.error_message,
            'progress_percentage': self.progress_percentage,
            'current_node_id': self.current_node_id,
            'total_execution_time_ms': self.total_execution_time_ms,
            'total_tokens_used': self.total_tokens_used,
            'total_cost_usd': self.total_cost_usd,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'user_id': self.user_id
        }
    
    def add_log_entry(self, level, message, node_id=None, data=None):
        """Add an entry to the execution log"""
        if not self.execution_log:
            self.execution_log = []
        
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,  # info, warning, error
            'message': message,
            'node_id': node_id,
            'data': data
        }
        
        self.execution_log.append(log_entry)


class WorkflowTemplate(db.Model):
    """
    Model for workflow templates that users can use as starting points
    """
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    category = db.Column(db.String(50))  # e.g., 'content_creation', 'data_analysis'
    difficulty_level = db.Column(db.String(20), default='beginner')  # beginner, intermediate, advanced
    workflow_definition = db.Column(db.JSON, nullable=False)  # Complete workflow structure
    preview_image_url = db.Column(db.String(255))
    usage_count = db.Column(db.Integer, default=0)
    rating = db.Column(db.Float, default=0.0)
    tags = db.Column(db.JSON, default=list)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by_user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    
    # Relationships
    created_by = db.relationship('User', backref=db.backref('created_templates', lazy=True))
    
    def __repr__(self):
        return f'<WorkflowTemplate {self.name}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'category': self.category,
            'difficulty_level': self.difficulty_level,
            'workflow_definition': self.workflow_definition,
            'preview_image_url': self.preview_image_url,
            'usage_count': self.usage_count,
            'rating': self.rating,
            'tags': self.tags,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'created_by_user_id': self.created_by_user_id
        }

