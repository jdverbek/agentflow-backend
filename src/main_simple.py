from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# In-memory storage for demo purposes
agents = []
workflows = []
executions = []

@app.route('/')
def home():
    return jsonify({
        "message": "AgentFlow API is running!",
        "version": "1.0.0",
        "status": "healthy"
    })

@app.route('/api/health')
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "agentflow-backend"
    })

@app.route('/api/agents', methods=['GET'])
def get_agents():
    return jsonify({
        "agents": agents,
        "count": len(agents)
    })

@app.route('/api/agents', methods=['POST'])
def create_agent():
    data = request.get_json()
    agent = {
        "id": len(agents) + 1,
        "name": data.get('name', 'Unnamed Agent'),
        "description": data.get('description', ''),
        "model": data.get('model', 'gpt-3.5-turbo'),
        "system_prompt": data.get('system_prompt', ''),
        "tools": data.get('tools', []),
        "created_at": datetime.now().isoformat()
    }
    agents.append(agent)
    return jsonify(agent), 201

@app.route('/api/workflows', methods=['GET'])
def get_workflows():
    return jsonify({
        "workflows": workflows,
        "count": len(workflows)
    })

@app.route('/api/workflows', methods=['POST'])
def create_workflow():
    data = request.get_json()
    workflow = {
        "id": len(workflows) + 1,
        "name": data.get('name', 'Unnamed Workflow'),
        "description": data.get('description', ''),
        "agents": data.get('agents', []),
        "steps": data.get('steps', []),
        "created_at": datetime.now().isoformat()
    }
    workflows.append(workflow)
    return jsonify(workflow), 201

@app.route('/api/executions', methods=['GET'])
def get_executions():
    return jsonify({
        "executions": executions,
        "count": len(executions)
    })

@app.route('/api/executions', methods=['POST'])
def create_execution():
    data = request.get_json()
    execution = {
        "id": len(executions) + 1,
        "workflow_id": data.get('workflow_id'),
        "status": "running",
        "input": data.get('input', {}),
        "output": {},
        "logs": [],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    executions.append(execution)
    
    # Simulate execution completion
    execution["status"] = "completed"
    execution["output"] = {"message": "Workflow executed successfully (demo)"}
    execution["logs"] = ["Started execution", "Processing...", "Completed"]
    execution["updated_at"] = datetime.now().isoformat()
    
    return jsonify(execution), 201

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify({
        "total_agents": len(agents),
        "total_workflows": len(workflows),
        "total_executions": len(executions),
        "successful_executions": len([e for e in executions if e.get('status') == 'completed']),
        "last_updated": datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host='0.0.0.0', port=port, debug=debug)

