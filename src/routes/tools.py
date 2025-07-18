from flask import Blueprint, request, jsonify
from src.services.manus_tools import manus_tools

tools_bp = Blueprint('tools', __name__)

@tools_bp.route('/', methods=['GET'])
def get_tools():
    """Get list of available tools"""
    try:
        tools = manus_tools.get_available_tools()
        return jsonify({
            "success": True,
            "tools": tools,
            "total_tools": len(tools),
            "categories": list(set(tool["category"] for tool in tools))
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@tools_bp.route('/execute', methods=['POST'])
def execute_tool():
    """Execute a tool with given parameters"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400
        
        tool_name = data.get('tool_name')
        parameters = data.get('parameters', {})
        
        if not tool_name:
            return jsonify({
                "success": False,
                "error": "tool_name is required"
            }), 400
        
        result = manus_tools.execute_tool(tool_name, parameters)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@tools_bp.route('/categories', methods=['GET'])
def get_tool_categories():
    """Get tool categories"""
    try:
        tools = manus_tools.get_available_tools()
        categories = {}
        
        for tool in tools:
            category = tool["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append({
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"]
            })
        
        return jsonify({
            "success": True,
            "categories": categories,
            "total_categories": len(categories)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@tools_bp.route('/<tool_name>', methods=['GET'])
def get_tool_info(tool_name):
    """Get detailed information about a specific tool"""
    try:
        tools = manus_tools.get_available_tools()
        tool_info = next((tool for tool in tools if tool["name"] == tool_name), None)
        
        if not tool_info:
            return jsonify({
                "success": False,
                "error": f"Tool '{tool_name}' not found"
            }), 404
        
        return jsonify({
            "success": True,
            "tool": tool_info
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@tools_bp.route('/batch-execute', methods=['POST'])
def batch_execute_tools():
    """Execute multiple tools in sequence"""
    try:
        data = request.get_json()
        
        if not data or 'tools' not in data:
            return jsonify({
                "success": False,
                "error": "tools array is required"
            }), 400
        
        tools_to_execute = data['tools']
        results = []
        
        for tool_config in tools_to_execute:
            tool_name = tool_config.get('tool_name')
            parameters = tool_config.get('parameters', {})
            
            if not tool_name:
                results.append({
                    "success": False,
                    "error": "tool_name is required for each tool"
                })
                continue
            
            result = manus_tools.execute_tool(tool_name, parameters)
            results.append(result)
        
        return jsonify({
            "success": True,
            "results": results,
            "total_executed": len(results)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

