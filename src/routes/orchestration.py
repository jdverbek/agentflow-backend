"""
Orchestration API routes for AgentFlow Manager Agent
Uses Bulletproof Manager Agent with direct environment access
"""
from flask import Blueprint, request, jsonify
import time
import threading
import logging

# Import the fast manager agent
from src.orchestration.fast_manager_agent import FastManagerAgent

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
        logger.info("⚡ Initializing Fast Manager Agent...")
        manager_agent = FastManagerAgent()
        logger.info("✅ Fast Manager Agent initialized")
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



@orchestration_bp.route('/deliverables', methods=['GET'])
def list_deliverables():
    """List all created deliverable files"""
    try:
        agent = get_manager_agent()
        files_info = agent.list_created_files()
        
        return jsonify({
            "success": True,
            "files": files_info,
            "total_files": len(files_info),
            "deliverables_directory": agent.get_deliverables_directory()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Failed to list deliverables: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@orchestration_bp.route('/deliverables/info', methods=['GET'])
def deliverables_info():
    """Get information about the deliverables system"""
    try:
        agent = get_manager_agent()
        info = agent.deliverable_creator.get_deliverables_info()
        
        return jsonify({
            "success": True,
            "deliverables_info": info
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Failed to get deliverables info: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@orchestration_bp.route('/execution-history', methods=['GET'])
def execution_history():
    """Get execution history with deliverable creation details"""
    try:
        agent = get_manager_agent()
        history = agent.get_execution_history()
        
        return jsonify({
            "success": True,
            "execution_history": history,
            "total_executions": len(history)
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Failed to get execution history: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@orchestration_bp.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    """Download a specific deliverable file"""
    try:
        agent = get_manager_agent()
        base_dir = agent.get_deliverables_directory()
        
        # Security check: ensure filename is within deliverables directory
        import os
        from pathlib import Path
        
        # Find the file in any subdirectory
        file_path = None
        for subdir in ["presentations", "documents", "reports", "data"]:
            potential_path = os.path.join(base_dir, subdir, filename)
            if os.path.exists(potential_path):
                file_path = potential_path
                break
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({
                "success": False,
                "error": "File not found"
            }), 404
        
        # Determine MIME type based on file extension
        import mimetypes
        mimetype, _ = mimetypes.guess_type(file_path)
        if not mimetype:
            mimetype = 'application/octet-stream'
        
        # Send file
        from flask import send_file
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype=mimetype
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to download file {filename}: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@orchestration_bp.route('/deliverables/download-all/<execution_id>', methods=['GET'])
def download_all_files(execution_id):
    """Download all files for a specific execution as a ZIP"""
    try:
        agent = get_manager_agent()
        base_dir = agent.get_deliverables_directory()
        
        import zipfile
        import tempfile
        import os
        from flask import send_file
        
        # Create temporary ZIP file
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        
        with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            files_added = 0
            
            # Search for files with the execution ID
            for subdir in ["presentations", "documents", "reports", "data"]:
                subdir_path = os.path.join(base_dir, subdir)
                if os.path.exists(subdir_path):
                    for filename in os.listdir(subdir_path):
                        if execution_id in filename:
                            file_path = os.path.join(subdir_path, filename)
                            if os.path.isfile(file_path):
                                # Add file to ZIP with a clean name
                                arcname = f"{subdir}/{filename}"
                                zipf.write(file_path, arcname)
                                files_added += 1
            
            if files_added == 0:
                temp_zip.close()
                os.unlink(temp_zip.name)
                return jsonify({
                    "success": False,
                    "error": "No files found for this execution"
                }), 404
        
        temp_zip.close()
        
        # Send ZIP file
        return send_file(
            temp_zip.name,
            as_attachment=True,
            download_name=f"agentflow_deliverables_{execution_id}.zip",
            mimetype='application/zip'
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to create ZIP for execution {execution_id}: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

