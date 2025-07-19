"""
Improved orchestration routes that integrate CrewAI functionality with existing AgentFlow.
Provides enhanced endpoints while maintaining compatibility with existing routes.
"""
from flask import Blueprint, request, jsonify
import logging
from ..orchestration.improved_crew_manager import improved_crew_manager

logger = logging.getLogger(__name__)

# Create blueprint for improved orchestration
improved_orchestration_bp = Blueprint('improved_orchestration', __name__, url_prefix='/api/improved')

@improved_orchestration_bp.route('/health', methods=['GET'])
def improved_health():
    """Health check for improved orchestration system."""
    try:
        return jsonify({
            "status": "IMPROVED_ORCHESTRATION_ACTIVE",
            "crew_manager": "✅ Active",
            "chatxai_integration": "✅ Available" if improved_crew_manager.grok_llm else "❌ Not Available",
            "openai_integration": "✅ Available" if improved_crew_manager.openai_llm else "❌ Not Available",
            "sandbox_tools": "✅ Available",
            "version": "1.0.0-improved"
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "ERROR", "error": str(e)}), 500

@improved_orchestration_bp.route('/execute', methods=['POST'])
def improved_execute():
    """
    Execute tasks using the improved CrewAI-based orchestration.
    Provides enhanced multi-agent coordination with ChatXAI integration.
    """
    try:
        data = request.get_json()
        if not data or 'task' not in data:
            return jsonify({"error": "Task description required"}), 400
        
        task_description = data['task']
        max_iterations = data.get('max_iterations', 3)
        
        logger.info(f"Executing improved task: {task_description}")
        
        # Execute using improved crew manager
        result = improved_crew_manager.execute_task(task_description)
        
        if result.get('success'):
            return jsonify({
                "status": "completed",
                "task": task_description,
                "result": result['deliverable'],
                "verification": result['verification'],
                "execution_method": "ImprovedCrewAI",
                "agents_used": result.get('agents_used', []),
                "success": True
            })
        else:
            return jsonify({
                "status": "failed",
                "task": task_description,
                "error": result.get('error', 'Unknown error'),
                "execution_method": "ImprovedCrewAI",
                "success": False
            }), 500
            
    except Exception as e:
        error_msg = f"Improved execution failed: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            "status": "error",
            "error": error_msg,
            "execution_method": "ImprovedCrewAI",
            "success": False
        }), 500

@improved_orchestration_bp.route('/agents', methods=['GET'])
def list_improved_agents():
    """List available improved agents and their capabilities."""
    try:
        agents_info = {
            "manager_agent": {
                "role": "Strategic Manager",
                "model": "grok-4" if improved_crew_manager.grok_llm else "gpt-4o",
                "capabilities": ["task_planning", "delegation", "coordination"]
            },
            "research_agent": {
                "role": "Research Specialist", 
                "model": "gpt-4o" if improved_crew_manager.openai_llm else "grok-4",
                "capabilities": ["information_gathering", "data_collection", "analysis"]
            },
            "thinking_agent": {
                "role": "Strategic Thinker",
                "model": "grok-4" if improved_crew_manager.grok_llm else "gpt-4o", 
                "capabilities": ["reasoning", "analysis", "strategic_planning"]
            },
            "creator_agent": {
                "role": "Deliverable Creator",
                "model": "gpt-4o" if improved_crew_manager.openai_llm else "grok-4",
                "capabilities": ["code_execution", "deliverable_creation", "tool_usage"],
                "tools": ["improved_sandbox_exec", "improved_manus_create"]
            },
            "controller_agent": {
                "role": "Quality Controller",
                "model": "grok-4" if improved_crew_manager.grok_llm else "gpt-4o",
                "capabilities": ["verification", "quality_assurance", "validation"]
            }
        }
        
        return jsonify({
            "agents": agents_info,
            "total_agents": len(agents_info),
            "integration_status": "active"
        })
        
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        return jsonify({"error": str(e)}), 500

@improved_orchestration_bp.route('/tools', methods=['GET'])
def list_improved_tools():
    """List available improved tools and their capabilities."""
    try:
        tools_info = {
            "improved_sandbox_exec": {
                "description": "Secure code execution with E2B and RestrictedPython fallback",
                "capabilities": ["python_execution", "secure_sandbox", "fallback_support"],
                "status": "active"
            },
            "improved_manus_create": {
                "description": "Enhanced deliverable creation with Manus simulation",
                "capabilities": ["deliverable_generation", "document_creation", "artifact_management"],
                "status": "active"
            }
        }
        
        return jsonify({
            "tools": tools_info,
            "total_tools": len(tools_info),
            "sandbox_status": "e2b_with_fallback"
        })
        
    except Exception as e:
        logger.error(f"Failed to list tools: {e}")
        return jsonify({"error": str(e)}), 500

