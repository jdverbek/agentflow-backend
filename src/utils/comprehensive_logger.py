"""
Comprehensive Logging System for AgentFlow
Tracks every aspect of execution with detailed debugging information
"""
import os
import time
import json
import logging
import traceback
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
from functools import wraps

class AgentFlowLogger:
    """Comprehensive logging system for AgentFlow"""
    
    def __init__(self, log_level=logging.INFO):
        """Initialize comprehensive logging"""
        self.log_level = log_level
        self.execution_logs = []
        self.error_logs = []
        self.api_logs = []
        self.environment_logs = []
        self.performance_logs = []
        self.lock = threading.Lock()
        
        # Setup logging configuration
        self._setup_logging()
        
        # Log system initialization
        self.log_system_info()
        
    def _setup_logging(self):
        """Setup comprehensive logging configuration"""
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(name)20s | %(funcName)15s:%(lineno)4d | %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        )
        
        # Setup root logger
        self.logger = logging.getLogger('AgentFlow')
        self.logger.setLevel(self.log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler with detailed formatting
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for persistent logging
        try:
            file_handler = logging.FileHandler('/tmp/agentflow_debug.log')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            self.logger.warning(f"Could not create file handler: {e}")
        
        self.logger.info("🔧 Comprehensive logging system initialized")
    
    def log_system_info(self):
        """Log comprehensive system information"""
        self.logger.info("=" * 80)
        self.logger.info("🚀 AGENTFLOW COMPREHENSIVE LOGGING SYSTEM STARTED")
        self.logger.info("=" * 80)
        
        # System information
        self.logger.info(f"📅 Timestamp: {datetime.now().isoformat()}")
        self.logger.info(f"🐍 Python Version: {os.sys.version}")
        self.logger.info(f"💻 Platform: {os.name}")
        self.logger.info(f"📁 Working Directory: {os.getcwd()}")
        self.logger.info(f"🔧 Process ID: {os.getpid()}")
        
        # Environment variables audit
        self.log_environment_audit()
        
    def log_environment_audit(self):
        """Comprehensive environment variable audit"""
        self.logger.info("🔍 ENVIRONMENT VARIABLE AUDIT")
        self.logger.info("-" * 50)
        
        # Critical environment variables
        critical_vars = [
            'OPENAI_API_KEY',
            'GROK_API_KEY', 
            'FLASK_ENV',
            'FLASK_APP',
            'PORT',
            'RENDER',
            'RENDER_SERVICE_ID',
            'RENDER_SERVICE_NAME'
        ]
        
        env_status = {}
        
        for var in critical_vars:
            value = os.environ.get(var)
            if value:
                # Mask sensitive values
                if 'KEY' in var or 'TOKEN' in var:
                    masked_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***MASKED***"
                    self.logger.info(f"✅ {var}: {masked_value} (length: {len(value)})")
                    env_status[var] = {"status": "available", "length": len(value)}
                else:
                    self.logger.info(f"✅ {var}: {value}")
                    env_status[var] = {"status": "available", "value": value}
            else:
                self.logger.warning(f"❌ {var}: NOT SET")
                env_status[var] = {"status": "missing"}
        
        # All environment variables (for debugging)
        all_env_vars = dict(os.environ)
        self.logger.info(f"📊 Total environment variables: {len(all_env_vars)}")
        
        # Log environment status
        self.environment_logs.append({
            "timestamp": datetime.now().isoformat(),
            "critical_vars": env_status,
            "total_vars": len(all_env_vars),
            "render_detected": "RENDER" in all_env_vars
        })
        
        return env_status
    
    def log_api_initialization(self, client_name: str, api_key_available: bool, initialization_result: Dict):
        """Log API client initialization"""
        self.logger.info(f"🔌 API CLIENT INITIALIZATION: {client_name}")
        self.logger.info(f"   API Key Available: {'✅' if api_key_available else '❌'}")
        self.logger.info(f"   Initialization Success: {'✅' if initialization_result.get('success') else '❌'}")
        
        if initialization_result.get('error'):
            self.logger.error(f"   Error: {initialization_result['error']}")
            self.logger.error(f"   Traceback: {initialization_result.get('traceback', 'N/A')}")
        
        # Store API log
        api_log = {
            "timestamp": datetime.now().isoformat(),
            "client_name": client_name,
            "api_key_available": api_key_available,
            "initialization_result": initialization_result
        }
        
        with self.lock:
            self.api_logs.append(api_log)
    
    def log_task_execution_start(self, task: str, execution_id: str, max_iterations: int):
        """Log task execution start"""
        self.logger.info("🚀 TASK EXECUTION STARTED")
        self.logger.info(f"   Execution ID: {execution_id}")
        self.logger.info(f"   Task: {task[:100]}{'...' if len(task) > 100 else ''}")
        self.logger.info(f"   Max Iterations: {max_iterations}")
        self.logger.info(f"   Start Time: {datetime.now().isoformat()}")
        
        # Create execution log entry
        execution_log = {
            "execution_id": execution_id,
            "task": task,
            "max_iterations": max_iterations,
            "start_time": time.time(),
            "start_timestamp": datetime.now().isoformat(),
            "status": "STARTED",
            "steps": [],
            "errors": [],
            "api_calls": [],
            "performance_metrics": {}
        }
        
        with self.lock:
            self.execution_logs.append(execution_log)
        
        return execution_log
    
    def log_execution_step(self, execution_id: str, step_name: str, step_data: Dict):
        """Log individual execution step"""
        self.logger.info(f"📋 EXECUTION STEP: {step_name}")
        self.logger.info(f"   Execution ID: {execution_id}")
        self.logger.info(f"   Step Data: {json.dumps(step_data, default=str, indent=2)}")
        
        # Find and update execution log
        with self.lock:
            for log in self.execution_logs:
                if log["execution_id"] == execution_id:
                    log["steps"].append({
                        "step_name": step_name,
                        "timestamp": datetime.now().isoformat(),
                        "data": step_data
                    })
                    break
    
    def log_api_call(self, execution_id: str, client_name: str, model: str, request_data: Dict, response_data: Dict):
        """Log API call details"""
        self.logger.info(f"📞 API CALL: {client_name}")
        self.logger.info(f"   Model: {model}")
        self.logger.info(f"   Request Tokens: {request_data.get('estimated_tokens', 'N/A')}")
        self.logger.info(f"   Response Tokens: {response_data.get('tokens_used', 'N/A')}")
        self.logger.info(f"   Success: {'✅' if response_data.get('success') else '❌'}")
        
        if response_data.get('error'):
            self.logger.error(f"   API Error: {response_data['error']}")
        
        # Store API call log
        api_call_log = {
            "execution_id": execution_id,
            "timestamp": datetime.now().isoformat(),
            "client_name": client_name,
            "model": model,
            "request_data": request_data,
            "response_data": response_data
        }
        
        with self.lock:
            self.api_logs.append(api_call_log)
            
            # Update execution log
            for log in self.execution_logs:
                if log["execution_id"] == execution_id:
                    log["api_calls"].append(api_call_log)
                    break
    
    def log_error(self, execution_id: str, error_type: str, error_message: str, error_context: Dict):
        """Log error with comprehensive context"""
        self.logger.error(f"❌ ERROR: {error_type}")
        self.logger.error(f"   Execution ID: {execution_id}")
        self.logger.error(f"   Message: {error_message}")
        self.logger.error(f"   Context: {json.dumps(error_context, default=str, indent=2)}")
        self.logger.error(f"   Traceback: {traceback.format_exc()}")
        
        # Store error log
        error_log = {
            "execution_id": execution_id,
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "error_context": error_context,
            "traceback": traceback.format_exc()
        }
        
        with self.lock:
            self.error_logs.append(error_log)
            
            # Update execution log
            for log in self.execution_logs:
                if log["execution_id"] == execution_id:
                    log["errors"].append(error_log)
                    log["status"] = "ERROR"
                    break
    
    def log_performance_metric(self, execution_id: str, metric_name: str, metric_value: Any, metric_context: Dict = None):
        """Log performance metrics"""
        self.logger.info(f"📊 PERFORMANCE: {metric_name} = {metric_value}")
        
        performance_log = {
            "execution_id": execution_id,
            "timestamp": datetime.now().isoformat(),
            "metric_name": metric_name,
            "metric_value": metric_value,
            "metric_context": metric_context or {}
        }
        
        with self.lock:
            self.performance_logs.append(performance_log)
            
            # Update execution log
            for log in self.execution_logs:
                if log["execution_id"] == execution_id:
                    if "performance_metrics" not in log:
                        log["performance_metrics"] = {}
                    log["performance_metrics"][metric_name] = {
                        "value": metric_value,
                        "timestamp": datetime.now().isoformat(),
                        "context": metric_context
                    }
                    break
    
    def log_task_execution_complete(self, execution_id: str, success: bool, result: Dict):
        """Log task execution completion"""
        status = "✅ COMPLETED" if success else "❌ FAILED"
        self.logger.info(f"🏁 TASK EXECUTION {status}")
        self.logger.info(f"   Execution ID: {execution_id}")
        self.logger.info(f"   Success: {success}")
        self.logger.info(f"   Result Summary: {json.dumps(result, default=str, indent=2)[:500]}...")
        
        # Update execution log
        with self.lock:
            for log in self.execution_logs:
                if log["execution_id"] == execution_id:
                    log["status"] = "COMPLETED" if success else "FAILED"
                    log["end_time"] = time.time()
                    log["end_timestamp"] = datetime.now().isoformat()
                    log["execution_time"] = log["end_time"] - log["start_time"]
                    log["success"] = success
                    log["result"] = result
                    break
    
    def get_comprehensive_report(self, execution_id: str = None) -> Dict[str, Any]:
        """Generate comprehensive execution report"""
        with self.lock:
            if execution_id:
                # Get specific execution report
                execution_log = None
                for log in self.execution_logs:
                    if log["execution_id"] == execution_id:
                        execution_log = log
                        break
                
                if not execution_log:
                    return {"error": f"Execution {execution_id} not found"}
                
                # Get related logs
                related_api_calls = [call for call in self.api_logs if call.get("execution_id") == execution_id]
                related_errors = [error for error in self.error_logs if error.get("execution_id") == execution_id]
                related_performance = [perf for perf in self.performance_logs if perf.get("execution_id") == execution_id]
                
                return {
                    "execution_log": execution_log,
                    "api_calls": related_api_calls,
                    "errors": related_errors,
                    "performance_metrics": related_performance,
                    "summary": {
                        "total_api_calls": len(related_api_calls),
                        "total_errors": len(related_errors),
                        "execution_time": execution_log.get("execution_time", 0),
                        "success": execution_log.get("success", False)
                    }
                }
            else:
                # Get overall system report
                return {
                    "system_status": {
                        "total_executions": len(self.execution_logs),
                        "total_api_calls": len(self.api_logs),
                        "total_errors": len(self.error_logs),
                        "total_performance_metrics": len(self.performance_logs)
                    },
                    "recent_executions": self.execution_logs[-5:],
                    "recent_errors": self.error_logs[-5:],
                    "environment_status": self.environment_logs[-1] if self.environment_logs else {},
                    "api_status": self.api_logs[-5:]
                }
    
    def export_logs_to_file(self, filepath: str = None):
        """Export all logs to file for analysis"""
        if not filepath:
            filepath = f"/tmp/agentflow_logs_{int(time.time())}.json"
        
        try:
            with self.lock:
                export_data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "execution_logs": self.execution_logs,
                    "api_logs": self.api_logs,
                    "error_logs": self.error_logs,
                    "performance_logs": self.performance_logs,
                    "environment_logs": self.environment_logs
                }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, default=str, indent=2)
            
            self.logger.info(f"📁 Logs exported to: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export logs: {e}")
            return None

# Global logger instance
comprehensive_logger = AgentFlowLogger()

def log_execution_step(step_name: str):
    """Decorator to log execution steps"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            execution_id = kwargs.get('execution_id', 'unknown')
            start_time = time.time()
            
            comprehensive_logger.log_execution_step(execution_id, f"{step_name}_START", {
                "function": func.__name__,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys())
            })
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                comprehensive_logger.log_execution_step(execution_id, f"{step_name}_COMPLETE", {
                    "function": func.__name__,
                    "execution_time": execution_time,
                    "success": True,
                    "result_type": type(result).__name__
                })
                
                comprehensive_logger.log_performance_metric(execution_id, f"{step_name}_execution_time", execution_time)
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                comprehensive_logger.log_error(execution_id, f"{step_name}_ERROR", str(e), {
                    "function": func.__name__,
                    "execution_time": execution_time,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                })
                
                raise
        
        return wrapper
    return decorator

def log_api_call(client_name: str, model: str):
    """Decorator to log API calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            execution_id = kwargs.get('execution_id', 'unknown')
            start_time = time.time()
            
            request_data = {
                "function": func.__name__,
                "model": model,
                "estimated_tokens": kwargs.get('max_tokens', 'unknown')
            }
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                response_data = {
                    "success": True,
                    "execution_time": execution_time,
                    "tokens_used": getattr(result, 'usage', {}).get('total_tokens', 'unknown') if hasattr(result, 'usage') else 'unknown',
                    "response_type": type(result).__name__
                }
                
                comprehensive_logger.log_api_call(execution_id, client_name, model, request_data, response_data)
                comprehensive_logger.log_performance_metric(execution_id, f"api_call_{client_name}_time", execution_time)
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                response_data = {
                    "success": False,
                    "execution_time": execution_time,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                
                comprehensive_logger.log_api_call(execution_id, client_name, model, request_data, response_data)
                
                raise
        
        return wrapper
    return decorator

