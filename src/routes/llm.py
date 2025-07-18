from flask import Blueprint, request, jsonify
from src.services.llm_service import llm_service

llm_bp = Blueprint('llm', __name__)

@llm_bp.route('/models', methods=['GET'])
def get_models():
    """Get available LLM models from all providers"""
    try:
        models = llm_service.get_available_models()
        return jsonify({
            "success": True,
            "models": models,
            "total_providers": len([p for p in models.values() if p])
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@llm_bp.route('/chat', methods=['POST'])
def chat_completion():
    """Generate chat completion using specified provider and model"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['provider', 'model', 'messages']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "success": False,
                    "error": f"Missing required field: {field}"
                }), 400
        
        provider = data['provider']
        model = data['model']
        messages = data['messages']
        
        # Optional parameters
        kwargs = {}
        optional_params = ['temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty']
        for param in optional_params:
            if param in data:
                kwargs[param] = data[param]
        
        # Generate completion
        result = llm_service.chat_completion(provider, model, messages, **kwargs)
        
        # Estimate cost
        usage = result.get('usage', {})
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        estimated_cost = llm_service.estimate_cost(provider, model, prompt_tokens, completion_tokens)
        
        return jsonify({
            "success": True,
            "result": result,
            "estimated_cost": estimated_cost
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@llm_bp.route('/providers', methods=['GET'])
def get_providers():
    """Get available LLM providers and their status"""
    try:
        providers = {
            "openai": {
                "available": llm_service.openai_client is not None,
                "name": "OpenAI",
                "description": "GPT models from OpenAI"
            },
            "grok": {
                "available": llm_service.grok_api_key is not None,
                "name": "Grok (xAI)",
                "description": "Grok models from xAI"
            }
        }
        
        return jsonify({
            "success": True,
            "providers": providers
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@llm_bp.route('/cost-estimate', methods=['POST'])
def estimate_cost():
    """Estimate cost for a given provider, model, and token usage"""
    try:
        data = request.get_json()
        
        required_fields = ['provider', 'model', 'prompt_tokens', 'completion_tokens']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "success": False,
                    "error": f"Missing required field: {field}"
                }), 400
        
        cost = llm_service.estimate_cost(
            data['provider'],
            data['model'],
            data['prompt_tokens'],
            data['completion_tokens']
        )
        
        return jsonify({
            "success": True,
            "estimated_cost": cost
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

