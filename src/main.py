import os
import sys
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')

# Enable CORS for all routes and origins
CORS(app, origins="*")

# Initialize database only if DATABASE_URL is provided
database_url = os.environ.get('DATABASE_URL')
if database_url:
    try:
        from flask_sqlalchemy import SQLAlchemy
        app.config['SQLALCHEMY_DATABASE_URI'] = database_url
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        db = SQLAlchemy(app)
        
        # Import models and routes after db initialization
        from src.models.user import User
        from src.models.agent import Agent
        from src.models.workflow import Workflow
        
        # Import and register blueprints
        from src.routes.user import user_bp
        from src.routes.agent import agent_bp
        from src.routes.workflow import workflow_bp
        from src.routes.llm import llm_bp
        from src.routes.tools import tools_bp
        from src.routes.agent_execute import agent_execute_bp
        from src.routes.orchestration import orchestration_bp
        from src.routes.progress import progress_bp
        
        app.register_blueprint(user_bp, url_prefix='/api')
        app.register_blueprint(agent_bp, url_prefix='/api')
        app.register_blueprint(workflow_bp, url_prefix='/api')
        app.register_blueprint(llm_bp, url_prefix='/api/llm')
        app.register_blueprint(tools_bp, url_prefix='/api/tools')
        app.register_blueprint(agent_execute_bp)
        app.register_blueprint(orchestration_bp, url_prefix='/api/orchestration')
        app.register_blueprint(progress_bp)
        
        # Create tables
        with app.app_context():
            db.create_all()
            
    except Exception as e:
        print(f"Database initialization failed: {e}")
        # Fall back to in-memory storage
        database_url = None

# If no database, use simple in-memory storage
if not database_url:
    from src.routes.llm import llm_bp
    from src.routes.tools import tools_bp
    from src.routes.agent_execute import agent_execute_bp
    from src.routes.orchestration import orchestration_bp
    from src.routes.progress import progress_bp
    
    app.register_blueprint(llm_bp, url_prefix='/api/llm')
    app.register_blueprint(tools_bp, url_prefix='/api/tools')
    app.register_blueprint(agent_execute_bp)
    app.register_blueprint(orchestration_bp, url_prefix='/api/orchestration')
    app.register_blueprint(progress_bp)
    
    agents = []
    workflows = []
    executions = []
    
    @app.route('/api/agents', methods=['GET'])
    def get_agents():
        return jsonify({"agents": agents, "count": len(agents)})
    
    @app.route('/api/workflows', methods=['GET'])
    def get_workflows():
        return jsonify({"workflows": workflows, "count": len(workflows)})
    
    @app.route('/api/executions', methods=['GET'])
    def get_executions():
        return jsonify({"executions": executions, "count": len(executions)})

@app.route('/')
def home():
    return jsonify({
        "message": "AgentFlow API is running!",
        "version": "1.0.0",
        "status": "healthy",
        "database": "connected" if database_url else "in-memory"
    })

@app.route('/api/health')
def health():
    return jsonify({
        "status": "healthy",
        "service": "agentflow-backend",
        "database": "connected" if database_url else "in-memory"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host='0.0.0.0', port=port, debug=debug)

