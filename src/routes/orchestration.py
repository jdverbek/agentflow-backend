"""
Orchestration API routes for AgentFlow Manager Agent
Uses Logged Manager Agent with comprehensive logging and debugging
"""
from flask import Blueprint, request, jsonify
import time
import threading
import logging
import traceback

# Import the logged manager agent with comprehensive logging
from orchestration.logged_manager_agent import LoggedManagerAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
orchestration_bp = Blueprint('orchestration', __name__)

# Global manager agent instance
manager_agent = None

def get_manager_agent():
    """Get or create manager agent instance with comprehensive logging"""
    global manager_agent
    if manager_agent is None:
        logger.info("🔧 Initializing Logged Manager Agent with comprehensive logging...")
        try:
            manager_agent = LoggedManagerAgent()
            logger.info("✅ Logged Manager Agent initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Logged Manager Agent: {e}")
            logger.error(f"   Traceback: {traceback.format_exc()}")
            raise
    return manager_agent

@orchestration_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with comprehensive logging"""
    try:
        logger.info("🏥 Health check request received")
        agent = get_manager_agent()
        health_status = agent.get_health_status()
        
        logger.info(f"✅ Health check completed: {health_status.get('system_status', 'UNKNOWN')}")
        return jsonify(health_status), 200
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        
        return jsonify({
            "status": "HEALTH_CHECK_FAILED",
            "error": str(e),
            "ai_clients": {"openai": "❌ Error", "grok": "❌ Error"},
            "traceback": traceback.format_exc()
        }), 500

@orchestration_bp.route('/execute', methods=['POST'])
def execute_task():
    """Execute task with comprehensive logging and debugging"""
    try:
        logger.info("🚀 Task execution request received")
        
        # Get request data
        data = request.get_json()
        if not data:
            logger.error("❌ No JSON data provided in request")
            return jsonify({"error": "No JSON data provided"}), 400
            
        task = data.get('task', '').strip()
        max_iterations = data.get('max_iterations', 3)
        
        if not task:
            logger.error("❌ Task is required but not provided")
            return jsonify({"error": "Task is required"}), 400
            
        logger.info(f"📝 Task received: {task[:100]}{'...' if len(task) > 100 else ''}")
        logger.info(f"🔄 Max iterations: {max_iterations}")
        
        # Get manager agent and execute task
        agent = get_manager_agent()
        
        logger.info("⚡ Starting task execution with comprehensive logging...")
        start_time = time.time()
        
        result = agent.execute_task(task, max_iterations)
        
        execution_time = time.time() - start_time
        logger.info(f"🏁 Task execution completed in {execution_time:.2f}s")
        logger.info(f"   Success: {result.get('success', False)}")
        logger.info(f"   API calls: {result.get('api_calls', 0)}")
        logger.info(f"   Tokens used: {result.get('tokens_used', 0)}")
        logger.info(f"   Validation: {result.get('validation', 'UNKNOWN')}")
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"❌ Task execution endpoint failed: {e}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        
        return jsonify({
            "success": False,
            "error": str(e),
            "validation": "ENDPOINT_EXECUTION_FAILED",
            "traceback": traceback.format_exc()
        }), 500

@orchestration_bp.route('/logs', methods=['GET'])
def get_logs():
    """Get comprehensive execution logs"""
    try:
        logger.info("📋 Logs request received")
        
        execution_id = request.args.get('execution_id')
        
        agent = get_manager_agent()
        logs = agent.get_execution_logs(execution_id)
        
        logger.info(f"✅ Logs retrieved for execution_id: {execution_id or 'ALL'}")
        
        return jsonify(logs), 200
        
    except Exception as e:
        logger.error(f"❌ Failed to retrieve logs: {e}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@orchestration_bp.route('/logs/export', methods=['POST'])
def export_logs():
    """Export comprehensive logs to file"""
    try:
        logger.info("📁 Log export request received")
        
        data = request.get_json() or {}
        filepath = data.get('filepath')
        
        agent = get_manager_agent()
        exported_filepath = agent.export_logs(filepath)
        
        if exported_filepath:
            logger.info(f"✅ Logs exported to: {exported_filepath}")
            return jsonify({
                "success": True,
                "filepath": exported_filepath,
                "message": "Logs exported successfully"
            }), 200
        else:
            logger.error("❌ Failed to export logs")
            return jsonify({
                "success": False,
                "error": "Failed to export logs"
            }), 500
        
    except Exception as e:
        logger.error(f"❌ Log export failed: {e}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@orchestration_bp.route('/debug', methods=['GET'])
def debug_info():
    """Get comprehensive debug information"""
    try:
        logger.info("🐛 Debug info request received")
        
        agent = get_manager_agent()
        
        # Get comprehensive debug information
        debug_info = {
            "timestamp": time.time(),
            "health_status": agent.get_health_status(),
            "recent_logs": agent.get_execution_logs(),
            "system_info": {
                "python_version": str(sys.version),
                "working_directory": os.getcwd(),
                "environment_variables": {
                    "OPENAI_API_KEY": "✓" if os.environ.get('OPENAI_API_KEY') else "✗",
                    "GROK_API_KEY": "✓" if os.environ.get('GROK_API_KEY') else "✗",
                    "FLASK_ENV": os.environ.get('FLASK_ENV', 'not_set'),
                    "RENDER": "✓" if os.environ.get('RENDER') else "✗"
                }
            }
        }
        
        logger.info("✅ Debug info compiled successfully")
        
        return jsonify(debug_info), 200
        
    except Exception as e:
        logger.error(f"❌ Debug info failed: {e}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

# Import required modules for debug endpoint
import sys
import os

