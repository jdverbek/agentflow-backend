"""
Orchestration API routes for AgentFlow Manager Agent
Uses Bulletproof Manager Agent with direct environment access
"""
from flask import Blueprint, request, jsonify
import time
import threading
import logging

# Import the bulletproof manager agent
from src.orchestration.bulletproof_manager_agent import BulletproofManagerAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
orchestration_bp = Blueprint('orchestration', __name__)

# Global manager agent instance
manager_agent = None

def get_manager_agent():
    """Get or create manager agent instance"""
    global manager_agent
    if manager_agent is None:
        logger.info("🔧 Initializing Bulletproof Manager Agent...")
        manager_agent = BulletproofManagerAgent()
        logger.info("✅ Bulletproof Manager Agent initialized")
    return manager_agent

@orchestration_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Manager Agent"""
    try:
        agent = get_manager_agent()
        health_status = agent.get_health_status()
        return jsonify(health_status), 200
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return jsonify({
            "status": "HEALTH_CHECK_FAILED",
            "error": str(e),
            "ai_clients": {"openai": "❌ Error", "grok": "❌ Error"}
        }), 500

@orchestration_bp.route('/execute', methods=['POST'])
def execute_task():
    """Execute task with Manager Agent orchestration"""
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        task = data.get('task', '').strip()
        max_iterations = data.get('max_iterations', 3)
        
        if not task:
            return jsonify({"error": "Task is required"}), 400
            
        logger.info(f"🚀 Received task execution request: {task[:100]}...")
        
        # Get manager agent and execute task
        agent = get_manager_agent()
        result = agent.execute_task(task, max_iterations)
        
        logger.info(f"✅ Task execution completed: {result.get('success', False)}")
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"❌ Task execution endpoint failed: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "validation": "ENDPOINT_EXECUTION_FAILED"
        }), 500

