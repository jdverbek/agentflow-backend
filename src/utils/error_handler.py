"""
Comprehensive Error Handling and Fallback System for AgentFlow
Provides robust error recovery and graceful degradation
"""
import logging
import traceback
import time
from typing import Dict, Any, Optional, Callable
from functools import wraps

logger = logging.getLogger(__name__)

class AgentFlowErrorHandler:
    """Comprehensive error handling for AgentFlow operations"""
    
    def __init__(self):
        self.error_counts = {}
        self.fallback_strategies = {}
        self.recovery_attempts = {}
    
    def with_retry(self, max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
        """Decorator for automatic retry with exponential backoff"""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                current_delay = delay
                
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        
                        if attempt < max_retries:
                            logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                            logger.info(f"Retrying in {current_delay}s...")
                            time.sleep(current_delay)
                            current_delay *= backoff
                        else:
                            logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
                
                raise last_exception
            return wrapper
        return decorator
    
    def with_fallback(self, fallback_func: Callable):
        """Decorator for fallback execution"""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Primary function {func.__name__} failed: {e}")
                    logger.info(f"Executing fallback for {func.__name__}")
                    
                    try:
                        return fallback_func(*args, **kwargs)
                    except Exception as fallback_error:
                        logger.error(f"Fallback also failed: {fallback_error}")
                        raise e  # Raise original error
            return wrapper
        return decorator
    
    def safe_execute(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Safely execute function with comprehensive error handling"""
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            return {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "error": None
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_info = self._create_error_info(e, func.__name__)
            
            logger.error(f"Safe execution failed for {func.__name__}: {error_info}")
            
            return {
                "success": False,
                "result": None,
                "execution_time": execution_time,
                "error": error_info
            }
    
    def _create_error_info(self, exception: Exception, function_name: str) -> Dict[str, Any]:
        """Create comprehensive error information"""
        return {
            "type": type(exception).__name__,
            "message": str(exception),
            "function": function_name,
            "timestamp": time.time(),
            "traceback": traceback.format_exc()
        }

class AIClientErrorHandler:
    """Specialized error handling for AI client operations"""
    
    def __init__(self):
        self.error_handler = AgentFlowErrorHandler()
        self.client_status = {}
    
    def handle_openai_error(self, error: Exception) -> Dict[str, Any]:
        """Handle OpenAI-specific errors"""
        error_type = type(error).__name__
        
        if "RateLimitError" in error_type:
            return {
                "error_type": "rate_limit",
                "message": "OpenAI rate limit exceeded",
                "retry_after": 60,
                "fallback_available": True
            }
        elif "AuthenticationError" in error_type:
            return {
                "error_type": "authentication",
                "message": "OpenAI API key invalid or missing",
                "retry_after": None,
                "fallback_available": False
            }
        elif "APIConnectionError" in error_type:
            return {
                "error_type": "connection",
                "message": "Failed to connect to OpenAI API",
                "retry_after": 30,
                "fallback_available": True
            }
        else:
            return {
                "error_type": "unknown",
                "message": str(error),
                "retry_after": 10,
                "fallback_available": True
            }
    
    def create_fallback_response(self, task: str, agent_type: str) -> Dict[str, Any]:
        """Create fallback response when AI is unavailable"""
        logger.info(f"Creating fallback response for {agent_type}")
        
        fallback_responses = {
            "research_analyst": {
                "type": "research",
                "content": f"Research analysis for: {task}. Due to AI service unavailability, this is a structured fallback response with key research areas identified.",
                "sources": ["Fallback Research Database"],
                "key_findings": [
                    "Primary research area identified",
                    "Secondary analysis required",
                    "Further investigation needed"
                ],
                "fallback": True
            },
            "content_creator": {
                "type": "content",
                "content": f"Content creation for: {task}. Structured content outline provided as fallback when AI services are unavailable.",
                "slide_titles": [
                    "Introduction",
                    "Main Content",
                    "Analysis",
                    "Conclusions"
                ],
                "narrative_text": "Fallback narrative structure provided",
                "fallback": True
            },
            "execution_agent": {
                "type": "execution",
                "deliverable": {
                    "type": "Structured Document",
                    "content": f"Deliverable structure for: {task}",
                    "status": "fallback_mode"
                },
                "fallback": True
            }
        }
        
        return fallback_responses.get(agent_type, {
            "type": "generic",
            "content": f"Fallback response for {agent_type}: {task}",
            "fallback": True
        })

class EnvironmentErrorHandler:
    """Handle environment-related errors"""
    
    def __init__(self):
        self.env_checks = {}
    
    def validate_environment(self) -> Dict[str, Any]:
        """Validate environment configuration"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Check required environment variables
        required_vars = ["OPENAI_API_KEY", "FLASK_ENV"]
        optional_vars = ["GROK_API_KEY"]
        
        for var in required_vars:
            if not self._check_env_var(var):
                validation_results["valid"] = False
                validation_results["errors"].append(f"Required environment variable {var} not found")
        
        for var in optional_vars:
            if not self._check_env_var(var):
                validation_results["warnings"].append(f"Optional environment variable {var} not found")
        
        # Check Python dependencies
        try:
            import openai
            validation_results["recommendations"].append("OpenAI library available")
        except ImportError:
            validation_results["valid"] = False
            validation_results["errors"].append("OpenAI library not installed")
        
        return validation_results
    
    def _check_env_var(self, var_name: str) -> bool:
        """Check if environment variable exists"""
        import os
        
        # Check multiple variations
        variations = [
            var_name,
            var_name.upper(),
            var_name.lower(),
            f"RENDER_{var_name}",
            f"APP_{var_name}"
        ]
        
        for variation in variations:
            if os.environ.get(variation):
                return True
        
        return False

class TaskExecutionErrorHandler:
    """Handle task execution errors with recovery strategies"""
    
    def __init__(self):
        self.error_handler = AgentFlowErrorHandler()
        self.ai_error_handler = AIClientErrorHandler()
        self.env_error_handler = EnvironmentErrorHandler()
    
    def execute_with_recovery(self, task_func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Execute task with comprehensive error recovery"""
        
        # Pre-execution validation
        env_validation = self.env_error_handler.validate_environment()
        if not env_validation["valid"]:
            return {
                "success": False,
                "error": "Environment validation failed",
                "details": env_validation,
                "recovery_attempted": False
            }
        
        # Execute with retry and fallback
        @self.error_handler.with_retry(max_retries=2, delay=1.0)
        def execute_task():
            return task_func(*args, **kwargs)
        
        try:
            result = execute_task()
            return {
                "success": True,
                "result": result,
                "recovery_attempted": False
            }
            
        except Exception as e:
            logger.error(f"Task execution failed after retries: {e}")
            
            # Attempt recovery
            recovery_result = self._attempt_recovery(task_func, e, *args, **kwargs)
            
            return {
                "success": recovery_result["success"],
                "result": recovery_result.get("result"),
                "error": str(e),
                "recovery_attempted": True,
                "recovery_details": recovery_result
            }
    
    def _attempt_recovery(self, task_func: Callable, error: Exception, *args, **kwargs) -> Dict[str, Any]:
        """Attempt to recover from task execution failure"""
        logger.info("Attempting task recovery...")
        
        try:
            # If it's an AI-related error, try fallback mode
            if "openai" in str(error).lower() or "api" in str(error).lower():
                logger.info("Attempting AI fallback recovery")
                
                # Create fallback result
                fallback_result = {
                    "agent_results": {
                        "research_analyst": self.ai_error_handler.create_fallback_response(
                            kwargs.get("task", "Unknown task"), "research_analyst"
                        ),
                        "content_creator": self.ai_error_handler.create_fallback_response(
                            kwargs.get("task", "Unknown task"), "content_creator"
                        )
                    },
                    "quality_score": 6.0,  # Lower score for fallback
                    "deliverables": [],
                    "summary": "Task completed in fallback mode due to AI service unavailability",
                    "fallback_mode": True
                }
                
                return {
                    "success": True,
                    "result": fallback_result,
                    "recovery_method": "ai_fallback"
                }
            
            # Other recovery strategies can be added here
            
            return {
                "success": False,
                "recovery_method": "none_available"
            }
            
        except Exception as recovery_error:
            logger.error(f"Recovery attempt failed: {recovery_error}")
            return {
                "success": False,
                "recovery_method": "failed",
                "recovery_error": str(recovery_error)
            }

# Global error handler instances
error_handler = AgentFlowErrorHandler()
ai_error_handler = AIClientErrorHandler()
env_error_handler = EnvironmentErrorHandler()
task_error_handler = TaskExecutionErrorHandler()

