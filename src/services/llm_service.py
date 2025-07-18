import os
import openai
import requests
from typing import Dict, List, Optional, Any

class LLMService:
    """Service for managing multiple LLM providers (OpenAI, Grok)"""
    
    def __init__(self):
        self.openai_client = None
        self.grok_api_key = None
        self.grok_base_url = "https://api.x.ai/v1"
        
        # Initialize OpenAI
        openai_key = os.environ.get('OPENAI_API_KEY')
        if openai_key:
            self.openai_client = openai.OpenAI(api_key=openai_key)
        
        # Initialize Grok
        self.grok_api_key = os.environ.get('GROK_API_KEY')
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get list of available models from all providers"""
        models = {
            "openai": [],
            "grok": []
        }
        
        # OpenAI models
        if self.openai_client:
            models["openai"] = [
                "gpt-4o",
                "gpt-4o-mini", 
                "gpt-4-turbo",
                "gpt-3.5-turbo"
            ]
        
        # Grok models
        if self.grok_api_key:
            models["grok"] = [
                "grok-beta",
                "grok-vision-beta"
            ]
        
        return models
    
    def chat_completion(self, 
                       provider: str, 
                       model: str, 
                       messages: List[Dict[str, str]], 
                       **kwargs) -> Dict[str, Any]:
        """Generate chat completion using specified provider and model"""
        
        if provider == "openai":
            return self._openai_completion(model, messages, **kwargs)
        elif provider == "grok":
            return self._grok_completion(model, messages, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _openai_completion(self, model: str, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate completion using OpenAI"""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            
            return {
                "provider": "openai",
                "model": model,
                "content": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "finish_reason": response.choices[0].finish_reason
            }
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def _grok_completion(self, model: str, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate completion using Grok (xAI)"""
        if not self.grok_api_key:
            raise ValueError("Grok API key not configured")
        
        headers = {
            "Authorization": f"Bearer {self.grok_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        
        try:
            response = requests.post(
                f"{self.grok_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            
            return {
                "provider": "grok",
                "model": model,
                "content": data["choices"][0]["message"]["content"],
                "usage": data.get("usage", {}),
                "finish_reason": data["choices"][0].get("finish_reason")
            }
        except requests.exceptions.RequestException as e:
            raise Exception(f"Grok API error: {str(e)}")
    
    def estimate_cost(self, provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost for API usage"""
        
        # Cost per 1K tokens (approximate pricing)
        pricing = {
            "openai": {
                "gpt-4o": {"input": 0.005, "output": 0.015},
                "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
                "gpt-4-turbo": {"input": 0.01, "output": 0.03},
                "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}
            },
            "grok": {
                "grok-beta": {"input": 0.005, "output": 0.015},
                "grok-vision-beta": {"input": 0.005, "output": 0.015}
            }
        }
        
        if provider in pricing and model in pricing[provider]:
            rates = pricing[provider][model]
            input_cost = (prompt_tokens / 1000) * rates["input"]
            output_cost = (completion_tokens / 1000) * rates["output"]
            return round(input_cost + output_cost, 6)
        
        return 0.0

# Global LLM service instance
llm_service = LLMService()

