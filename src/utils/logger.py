"""
Comprehensive Logging System for AgentFlow
Provides detailed logging, debugging, and monitoring capabilities
"""
import logging
import json
import time
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List
from functools import wraps
import os

class AgentFlowLogger:
    """
    Advanced logging system for AgentFlow with multiple log levels,
    structured logging, and real-time monitoring capabilities
    """
    
    def __init__(self, name: str = "AgentFlow"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Create formatters
        self.detailed_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
        )
        
        self.json_formatter = logging.Formatter(
            '%(asctime)s | JSON | %(message)s'
        )
        
        # Console handler for immediate feedback
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.detailed_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for persistent logging
        log_dir = "/tmp/agentflow_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(f"{log_dir}/agentflow.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(self.detailed_formatter)
        self.logger.addHandler(file_handler)
        
        # JSON handler for structured logs
        json_handler = logging.FileHandler(f"{log_dir}/agentflow_structured.log")
        json_handler.setLevel(logging.INFO)
        json_handler.setFormatter(self.json_formatter)
        self.logger.addHandler(json_handler)
        
        # Execution tracking
        self.execution_logs: Dict[str, List[Dict]] = {}
        self.performance_metrics: Dict[str, Dict] = {}
        self.error_counts: Dict[str, int] = {}
        
    def log_execution_start(self, execution_id: str, task: str, agent_type: str = "manager"):
        """Log the start of an execution with full context"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "execution_start",
            "execution_id": execution_id,
            "task": task,
            "agent_type": agent_type,
            "status": "started"
        }
        
        if execution_id not in self.execution_logs:
            self.execution_logs[execution_id] = []
        
        self.execution_logs[execution_id].append(log_entry)
        
        self.logger.info(f"🚀 EXECUTION START | {execution_id} | {agent_type} | Task: {task[:100]}...")
        self.logger.info(json.dumps(log_entry))
        
    def log_agent_call(self, execution_id: str, agent_name: str, input_data: Any, 
                      api_provider: str = "openai", model: str = "unknown"):
        """Log individual agent API calls with full details"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "agent_api_call",
            "execution_id": execution_id,
            "agent_name": agent_name,
            "api_provider": api_provider,
            "model": model,
            "input_length": len(str(input_data)) if input_data else 0,
            "input_preview": str(input_data)[:200] if input_data else "None"
        }
        
        if execution_id in self.execution_logs:
            self.execution_logs[execution_id].append(log_entry)
        
        self.logger.info(f"🤖 API CALL | {execution_id} | {agent_name} | {api_provider}:{model}")
        self.logger.debug(json.dumps(log_entry))
        
    def log_agent_response(self, execution_id: str, agent_name: str, response: Any,
                          duration: float, success: bool = True, error: str = None):
        """Log agent responses with performance metrics"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "agent_response",
            "execution_id": execution_id,
            "agent_name": agent_name,
            "success": success,
            "duration_seconds": round(duration, 3),
            "response_length": len(str(response)) if response else 0,
            "response_preview": str(response)[:200] if response else "None",
            "error": error
        }
        
        if execution_id in self.execution_logs:
            self.execution_logs[execution_id].append(log_entry)
        
        # Update performance metrics
        if agent_name not in self.performance_metrics:
            self.performance_metrics[agent_name] = {
                "total_calls": 0,
                "successful_calls": 0,
                "total_duration": 0,
                "average_duration": 0,
                "error_count": 0
            }
        
        metrics = self.performance_metrics[agent_name]
        metrics["total_calls"] += 1
        metrics["total_duration"] += duration
        metrics["average_duration"] = metrics["total_duration"] / metrics["total_calls"]
        
        if success:
            metrics["successful_calls"] += 1
            self.logger.info(f"✅ RESPONSE | {execution_id} | {agent_name} | {duration:.3f}s | Success")
        else:
            metrics["error_count"] += 1
            self.error_counts[agent_name] = self.error_counts.get(agent_name, 0) + 1
            self.logger.error(f"❌ RESPONSE | {execution_id} | {agent_name} | {duration:.3f}s | Error: {error}")
        
        self.logger.debug(json.dumps(log_entry))
        
    def log_execution_complete(self, execution_id: str, result: Any, total_duration: float,
                             agents_used: List[str], success: bool = True):
        """Log execution completion with comprehensive results"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "execution_complete",
            "execution_id": execution_id,
            "success": success,
            "total_duration_seconds": round(total_duration, 3),
            "agents_used": agents_used,
            "result_size": len(str(result)) if result else 0,
            "result_preview": str(result)[:300] if result else "None"
        }
        
        if execution_id in self.execution_logs:
            self.execution_logs[execution_id].append(log_entry)
        
        if success:
            self.logger.info(f"🎉 EXECUTION COMPLETE | {execution_id} | {total_duration:.3f}s | Agents: {', '.join(agents_used)}")
        else:
            self.logger.error(f"💥 EXECUTION FAILED | {execution_id} | {total_duration:.3f}s")
        
        self.logger.info(json.dumps(log_entry))
        
    def log_error(self, execution_id: str, error: Exception, context: str = ""):
        """Log errors with full stack trace and context"""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "error",
            "execution_id": execution_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "stack_trace": traceback.format_exc()
        }
        
        if execution_id in self.execution_logs:
            self.execution_logs[execution_id].append(error_entry)
        
        self.logger.error(f"💥 ERROR | {execution_id} | {type(error).__name__}: {str(error)}")
        self.logger.error(f"Context: {context}")
        self.logger.debug(json.dumps(error_entry))
        
    def get_execution_logs(self, execution_id: str) -> List[Dict]:
        """Get all logs for a specific execution"""
        return self.execution_logs.get(execution_id, [])
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        total_executions = len(self.execution_logs)
        total_errors = sum(self.error_counts.values())
        
        return {
            "total_executions": total_executions,
            "total_errors": total_errors,
            "success_rate": ((total_executions - total_errors) / total_executions * 100) if total_executions > 0 else 0,
            "agent_performance": self.performance_metrics,
            "error_breakdown": self.error_counts,
            "timestamp": datetime.now().isoformat()
        }
        
    def log_health_check(self, component: str, status: str, details: Dict = None):
        """Log health check results"""
        health_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "health_check",
            "component": component,
            "status": status,
            "details": details or {}
        }
        
        if status == "healthy":
            self.logger.info(f"💚 HEALTH | {component} | {status}")
        else:
            self.logger.warning(f"💛 HEALTH | {component} | {status}")
        
        self.logger.debug(json.dumps(health_entry))

# Global logger instance
agentflow_logger = AgentFlowLogger()

def log_execution(func):
    """Decorator to automatically log function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        execution_id = kwargs.get('execution_id', f"exec_{int(time.time() * 1000)}")
        func_name = func.__name__
        
        agentflow_logger.logger.debug(f"🔧 FUNCTION START | {execution_id} | {func_name}")
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            agentflow_logger.logger.debug(f"✅ FUNCTION END | {execution_id} | {func_name} | {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            agentflow_logger.log_error(execution_id, e, f"Function: {func_name}")
            agentflow_logger.logger.error(f"❌ FUNCTION ERROR | {execution_id} | {func_name} | {duration:.3f}s")
            raise
    
    return wrapper

def log_api_call(provider: str, model: str = "unknown"):
    """Decorator to log API calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            execution_id = kwargs.get('execution_id', f"api_{int(time.time() * 1000)}")
            
            agentflow_logger.log_agent_call(
                execution_id=execution_id,
                agent_name=func.__name__,
                input_data=kwargs.get('prompt', args[0] if args else None),
                api_provider=provider,
                model=model
            )
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                agentflow_logger.log_agent_response(
                    execution_id=execution_id,
                    agent_name=func.__name__,
                    response=result,
                    duration=duration,
                    success=True
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                agentflow_logger.log_agent_response(
                    execution_id=execution_id,
                    agent_name=func.__name__,
                    response=None,
                    duration=duration,
                    success=False,
                    error=str(e)
                )
                raise
        
        return wrapper
    return decorator

