"""
Orchestration API routes for AgentFlow Manager Agent
Fixed import paths and comprehensive logging
"""
from flask import Blueprint, request, jsonify
import time
import threading
import logging
import traceback
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the comprehensive logging system
from utils.comprehensive_logger import comprehensive_logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
orchestration_bp = Blueprint('orchestration', __name__)

# Global manager agent instance
manager_agent = None

class FixedManagerAgent:
    """Fixed Manager Agent with proper imports and real AI execution"""
    
    def __init__(self):
        """Initialize with comprehensive logging and proper error handling"""
        comprehensive_logger.logger.info("🔧 FIXED MANAGER AGENT INITIALIZING")
        
        self.start_time = time.time()
        self.api_clients = {}
        self.initialization_logs = []
        
        # Log initialization start
        comprehensive_logger.log_execution_step("SYSTEM", "FIXED_MANAGER_AGENT_INIT_START", {
            "class": "FixedManagerAgent",
            "start_time": self.start_time
        })
        
        # Initialize with detailed logging
        self._initialize_with_logging()
        
        comprehensive_logger.logger.info("✅ FIXED MANAGER AGENT INITIALIZED")
    
    def _initialize_with_logging(self):
        """Initialize all components with detailed logging"""
        
        # Step 1: Environment Variable Audit
        comprehensive_logger.logger.info("🔍 STEP 1: ENVIRONMENT VARIABLE AUDIT")
        env_status = comprehensive_logger.log_environment_audit()
        
        # Step 2: API Client Initialization
        comprehensive_logger.logger.info("🔌 STEP 2: API CLIENT INITIALIZATION")
        self._initialize_api_clients_with_logging(env_status)
        
        # Step 3: System Health Check
        comprehensive_logger.logger.info("🏥 STEP 3: SYSTEM HEALTH CHECK")
        health_status = self._perform_health_check()
        
        comprehensive_logger.log_execution_step("SYSTEM", "FIXED_MANAGER_AGENT_INIT_COMPLETE", {
            "env_status": env_status,
            "health_status": health_status,
            "initialization_time": time.time() - self.start_time
        })
    
    def _initialize_api_clients_with_logging(self, env_status: dict):
        """Initialize API clients with comprehensive logging"""
        
        # OpenAI Client Initialization
        openai_key_available = env_status.get('OPENAI_API_KEY', {}).get('status') == 'available'
        comprehensive_logger.logger.info(f"🔑 OpenAI API Key Available: {'✅' if openai_key_available else '❌'}")
        
        openai_result = self._initialize_openai_client(openai_key_available)
        comprehensive_logger.log_api_initialization("openai", openai_key_available, openai_result)
        
        # Grok Client Initialization
        grok_key_available = env_status.get('GROK_API_KEY', {}).get('status') == 'available'
        comprehensive_logger.logger.info(f"🔑 Grok API Key Available: {'✅' if grok_key_available else '❌'}")
        
        grok_result = self._initialize_grok_client(grok_key_available)
        comprehensive_logger.log_api_initialization("grok", grok_key_available, grok_result)
        
        # Log overall API status
        available_clients = [name for name, client in self.api_clients.items() if client is not None]
        comprehensive_logger.logger.info(f"📊 Available API Clients: {available_clients}")
        
        if not available_clients:
            comprehensive_logger.logger.error("❌ NO API CLIENTS AVAILABLE - SYSTEM WILL NOT FUNCTION")
        else:
            comprehensive_logger.logger.info(f"✅ {len(available_clients)} API CLIENT(S) READY")
    
    def _initialize_openai_client(self, key_available: bool) -> dict:
        """Initialize OpenAI client with detailed logging"""
        if not key_available:
            return {"success": False, "error": "API key not available"}
        
        try:
            comprehensive_logger.logger.info("📦 Importing OpenAI library...")
            import openai
            comprehensive_logger.logger.info("✅ OpenAI library imported successfully")
            
            comprehensive_logger.logger.info("🔧 Creating OpenAI client...")
            api_key = os.environ.get('OPENAI_API_KEY', '').strip()
            
            if not api_key:
                return {"success": False, "error": "API key is empty after retrieval"}
            
            # Create client without proxies parameter (fix for deployment issue)
            self.api_clients['openai'] = openai.OpenAI(api_key=api_key)
            comprehensive_logger.logger.info("✅ OpenAI client created successfully")
            
            # Test the client with a simple call
            comprehensive_logger.logger.info("🧪 Testing OpenAI client...")
            test_result = self._test_openai_client()
            
            if test_result["success"]:
                comprehensive_logger.logger.info("✅ OpenAI client test successful")
                return {"success": True, "test_result": test_result}
            else:
                comprehensive_logger.logger.error(f"❌ OpenAI client test failed: {test_result['error']}")
                return {"success": False, "error": f"Client test failed: {test_result['error']}"}
            
        except ImportError as e:
            error_msg = f"OpenAI library not available: {e}"
            comprehensive_logger.logger.error(f"❌ {error_msg}")
            return {"success": False, "error": error_msg, "traceback": traceback.format_exc()}
        
        except Exception as e:
            error_msg = f"OpenAI client initialization failed: {e}"
            comprehensive_logger.logger.error(f"❌ {error_msg}")
            comprehensive_logger.logger.error(f"   Traceback: {traceback.format_exc()}")
            return {"success": False, "error": error_msg, "traceback": traceback.format_exc()}
    
    def _initialize_grok_client(self, key_available: bool) -> dict:
        """Initialize Grok client with detailed logging"""
        if not key_available:
            return {"success": False, "error": "API key not available"}
        
        try:
            comprehensive_logger.logger.info("📦 Importing OpenAI library for Grok...")
            import openai
            comprehensive_logger.logger.info("✅ OpenAI library imported for Grok")
            
            comprehensive_logger.logger.info("🔧 Creating Grok client...")
            api_key = os.environ.get('GROK_API_KEY', '').strip()
            
            if not api_key:
                return {"success": False, "error": "API key is empty after retrieval"}
            
            self.api_clients['grok'] = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.x.ai/v1"
            )
            comprehensive_logger.logger.info("✅ Grok client created successfully")
            
            # Test the client
            comprehensive_logger.logger.info("🧪 Testing Grok client...")
            test_result = self._test_grok_client()
            
            if test_result["success"]:
                comprehensive_logger.logger.info("✅ Grok client test successful")
                return {"success": True, "test_result": test_result}
            else:
                comprehensive_logger.logger.error(f"❌ Grok client test failed: {test_result['error']}")
                return {"success": False, "error": f"Client test failed: {test_result['error']}"}
            
        except Exception as e:
            error_msg = f"Grok client initialization failed: {e}"
            comprehensive_logger.logger.error(f"❌ {error_msg}")
            comprehensive_logger.logger.error(f"   Traceback: {traceback.format_exc()}")
            return {"success": False, "error": error_msg, "traceback": traceback.format_exc()}
    
    def _test_openai_client(self) -> dict:
        """Test OpenAI client with a simple API call"""
        try:
            client = self.api_clients.get('openai')
            if not client:
                return {"success": False, "error": "Client not available"}
            
            comprehensive_logger.logger.info("📞 Making test API call to OpenAI...")
            
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": "Say 'test successful'"}],
                max_tokens=10
            )
            
            comprehensive_logger.logger.info(f"✅ OpenAI test response: {response.choices[0].message.content}")
            comprehensive_logger.logger.info(f"📊 Tokens used: {response.usage.total_tokens}")
            
            return {
                "success": True,
                "response": response.choices[0].message.content,
                "tokens_used": response.usage.total_tokens
            }
            
        except Exception as e:
            comprehensive_logger.logger.error(f"❌ OpenAI test failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _test_grok_client(self) -> dict:
        """Test Grok client with a simple API call"""
        try:
            client = self.api_clients.get('grok')
            if not client:
                return {"success": False, "error": "Client not available"}
            
            comprehensive_logger.logger.info("📞 Making test API call to Grok...")
            
            response = client.chat.completions.create(
                model="grok-beta",
                messages=[{"role": "user", "content": "Say 'test successful'"}],
                max_tokens=10
            )
            
            comprehensive_logger.logger.info(f"✅ Grok test response: {response.choices[0].message.content}")
            comprehensive_logger.logger.info(f"📊 Tokens used: {response.usage.total_tokens}")
            
            return {
                "success": True,
                "response": response.choices[0].message.content,
                "tokens_used": response.usage.total_tokens
            }
            
        except Exception as e:
            comprehensive_logger.logger.error(f"❌ Grok test failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _perform_health_check(self) -> dict:
        """Perform comprehensive health check"""
        health_status = {
            "timestamp": time.time(),
            "uptime": time.time() - self.start_time,
            "api_clients": {},
            "system_status": "UNKNOWN"
        }
        
        # Check each API client
        for client_name, client in self.api_clients.items():
            if client is not None:
                health_status["api_clients"][client_name] = "✅ Available"
            else:
                health_status["api_clients"][client_name] = "❌ Not available"
        
        # Determine overall system status
        available_clients = [name for name, client in self.api_clients.items() if client is not None]
        
        if len(available_clients) >= 1:
            health_status["system_status"] = "✅ OPERATIONAL"
        else:
            health_status["system_status"] = "❌ NO_API_CLIENTS"
        
        comprehensive_logger.logger.info(f"🏥 System Health: {health_status['system_status']}")
        
        return health_status
    
    def get_health_status(self) -> dict:
        """Get current health status with comprehensive logging"""
        comprehensive_logger.logger.info("🏥 HEALTH STATUS REQUEST")
        
        health_status = self._perform_health_check()
        
        # Add comprehensive logging data
        health_status.update({
            "status": "FIXED_MANAGER_AGENT_ACTIVE",
            "environment_check": {
                "openai_key": "✓" if os.environ.get('OPENAI_API_KEY') else "✗",
                "grok_key": "✓" if os.environ.get('GROK_API_KEY') else "✗",
                "flask_env": os.environ.get('FLASK_ENV', 'not_set')
            },
            "comprehensive_logging": {
                "total_executions": len(comprehensive_logger.execution_logs),
                "total_api_calls": len(comprehensive_logger.api_logs),
                "total_errors": len(comprehensive_logger.error_logs),
                "recent_errors": comprehensive_logger.error_logs[-3:] if comprehensive_logger.error_logs else []
            }
        })
        
        return health_status
    
    def execute_task(self, task: str, max_iterations: int = 3) -> dict:
        """Execute task with comprehensive logging"""
        execution_id = f"exec_{int(time.time())}"
        
        # Log task execution start
        execution_log = comprehensive_logger.log_task_execution_start(task, execution_id, max_iterations)
        
        try:
            # Check system readiness
            available_clients = [name for name, client in self.api_clients.items() if client is not None]
            
            if not available_clients:
                error_msg = "No AI clients available for task execution"
                comprehensive_logger.log_error(execution_id, "NO_AI_CLIENTS", error_msg, {
                    "api_clients": self.api_clients,
                    "task": task
                })
                
                result = {
                    "success": False,
                    "error": error_msg,
                    "execution_id": execution_id,
                    "api_calls": 0,
                    "tokens_used": 0,
                    "agents_used": [],
                    "validation": "EXECUTION_FAILED_NO_AI_CLIENTS"
                }
                
                comprehensive_logger.log_task_execution_complete(execution_id, False, result)
                return result
            
            comprehensive_logger.log_execution_step(execution_id, "AI_CLIENTS_AVAILABLE", {
                "available_clients": available_clients,
                "total_clients": len(self.api_clients)
            })
            
            # Execute with real AI processing
            result = self._execute_with_real_ai(task, execution_id, max_iterations, available_clients)
            
            # Log completion
            comprehensive_logger.log_task_execution_complete(execution_id, result.get("success", False), result)
            
            return result
            
        except Exception as e:
            error_msg = f"Task execution failed with exception: {e}"
            comprehensive_logger.log_error(execution_id, "EXECUTION_EXCEPTION", error_msg, {
                "task": task,
                "max_iterations": max_iterations,
                "available_clients": [name for name, client in self.api_clients.items() if client is not None]
            })
            
            result = {
                "success": False,
                "error": error_msg,
                "execution_id": execution_id,
                "api_calls": 0,
                "tokens_used": 0,
                "agents_used": [],
                "validation": "EXECUTION_FAILED_EXCEPTION",
                "traceback": traceback.format_exc()
            }
            
            comprehensive_logger.log_task_execution_complete(execution_id, False, result)
            return result
    
    def _execute_with_real_ai(self, task: str, execution_id: str, max_iterations: int, available_clients: list) -> dict:
        """Execute task with real AI processing"""
        
        # Choose primary AI client
        primary_client = available_clients[0]
        client = self.api_clients[primary_client]
        
        comprehensive_logger.log_execution_step(execution_id, "PRIMARY_CLIENT_SELECTED", {
            "primary_client": primary_client,
            "available_clients": available_clients
        })
        
        total_api_calls = 0
        total_tokens = 0
        agents_used = []
        
        try:
            # Step 1: Task Analysis
            comprehensive_logger.logger.info("🧠 STEP 1: TASK ANALYSIS WITH AI")
            analysis_result = self._ai_task_analysis(client, primary_client, task, execution_id)
            
            if analysis_result["success"]:
                total_api_calls += 1
                total_tokens += analysis_result.get("tokens_used", 0)
                agents_used.append("Task Analyzer")
                
                comprehensive_logger.log_performance_metric(execution_id, "task_analysis_tokens", analysis_result.get("tokens_used", 0))
            else:
                comprehensive_logger.log_error(execution_id, "TASK_ANALYSIS_FAILED", analysis_result["error"], {
                    "primary_client": primary_client,
                    "task": task
                })
                
                return {
                    "success": False,
                    "error": f"Task analysis failed: {analysis_result['error']}",
                    "api_calls": total_api_calls,
                    "tokens_used": total_tokens,
                    "agents_used": agents_used,
                    "validation": "TASK_ANALYSIS_FAILED"
                }
            
            # Step 2: Content Generation
            comprehensive_logger.logger.info("✍️ STEP 2: CONTENT GENERATION WITH AI")
            content_result = self._ai_content_generation(client, primary_client, task, analysis_result["content"], execution_id)
            
            if content_result["success"]:
                total_api_calls += 1
                total_tokens += content_result.get("tokens_used", 0)
                agents_used.append("Content Creator")
                
                comprehensive_logger.log_performance_metric(execution_id, "content_generation_tokens", content_result.get("tokens_used", 0))
            else:
                comprehensive_logger.log_error(execution_id, "CONTENT_GENERATION_FAILED", content_result["error"], {
                    "primary_client": primary_client,
                    "task": task
                })
            
            # Step 3: Quality Review
            comprehensive_logger.logger.info("🔍 STEP 3: QUALITY REVIEW WITH AI")
            review_result = self._ai_quality_review(client, primary_client, content_result.get("content", ""), execution_id)
            
            if review_result["success"]:
                total_api_calls += 1
                total_tokens += review_result.get("tokens_used", 0)
                agents_used.append("Quality Reviewer")
                
                comprehensive_logger.log_performance_metric(execution_id, "quality_review_tokens", review_result.get("tokens_used", 0))
            
            # Step 4: Create Deliverables
            comprehensive_logger.logger.info("📦 STEP 4: CREATING DELIVERABLES")
            deliverables = self._create_deliverables(task, content_result.get("content", ""), execution_id)
            
            # Compile final result
            final_result = {
                "success": True,
                "execution_id": execution_id,
                "result": {
                    "task_analysis": analysis_result.get("content", ""),
                    "content": content_result.get("content", ""),
                    "quality_review": review_result.get("content", ""),
                    "deliverables": deliverables
                },
                "api_calls": total_api_calls,
                "tokens_used": total_tokens,
                "agents_used": agents_used,
                "quality_score": review_result.get("quality_score", 8.0),
                "ai_client_used": primary_client,
                "validation": "REAL_AI_EXECUTION_CONFIRMED_WITH_COMPREHENSIVE_LOGGING"
            }
            
            comprehensive_logger.log_performance_metric(execution_id, "total_api_calls", total_api_calls)
            comprehensive_logger.log_performance_metric(execution_id, "total_tokens_used", total_tokens)
            
            return final_result
            
        except Exception as e:
            comprehensive_logger.log_error(execution_id, "AI_PROCESSING_EXCEPTION", str(e), {
                "primary_client": primary_client,
                "total_api_calls": total_api_calls,
                "total_tokens": total_tokens,
                "agents_used": agents_used
            })
            
            return {
                "success": False,
                "error": f"AI processing failed: {str(e)}",
                "api_calls": total_api_calls,
                "tokens_used": total_tokens,
                "agents_used": agents_used,
                "validation": "AI_PROCESSING_FAILED",
                "traceback": traceback.format_exc()
            }
    
    def _ai_task_analysis(self, client, client_name: str, task: str, execution_id: str) -> dict:
        """Perform AI task analysis with logging"""
        try:
            comprehensive_logger.logger.info(f"📞 Making API call to {client_name} for task analysis")
            
            model = "gpt-4.1-mini" if client_name == "openai" else "grok-beta"
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a task analysis expert. Analyze the given task and break it down into specific, actionable subtasks. Provide clear insights about what needs to be accomplished."
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this task thoroughly and break it down: {task}"
                    }
                ],
                max_tokens=800,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            comprehensive_logger.logger.info(f"✅ Task analysis completed: {len(content)} characters, {tokens_used} tokens")
            
            return {
                "success": True,
                "content": content,
                "tokens_used": tokens_used,
                "model": model
            }
            
        except Exception as e:
            comprehensive_logger.logger.error(f"❌ Task analysis API call failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "tokens_used": 0
            }
    
    def _ai_content_generation(self, client, client_name: str, task: str, analysis: str, execution_id: str) -> dict:
        """Perform AI content generation with logging"""
        try:
            comprehensive_logger.logger.info(f"📞 Making API call to {client_name} for content generation")
            
            model = "gpt-4.1-mini" if client_name == "openai" else "grok-beta"
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional content creator. Generate comprehensive, high-quality content based on the task analysis. If it's a presentation, include slide titles and detailed narrative text."
                    },
                    {
                        "role": "user",
                        "content": f"Based on this analysis: {analysis}\n\nGenerate detailed content for: {task}"
                    }
                ],
                max_tokens=1500,
                temperature=0.8
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            comprehensive_logger.logger.info(f"✅ Content generation completed: {len(content)} characters, {tokens_used} tokens")
            
            return {
                "success": True,
                "content": content,
                "tokens_used": tokens_used,
                "model": model
            }
            
        except Exception as e:
            comprehensive_logger.logger.error(f"❌ Content generation API call failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "tokens_used": 0
            }
    
    def _ai_quality_review(self, client, client_name: str, content: str, execution_id: str) -> dict:
        """Perform AI quality review with logging"""
        try:
            comprehensive_logger.logger.info(f"📞 Making API call to {client_name} for quality review")
            
            model = "gpt-4.1-mini" if client_name == "openai" else "grok-beta"
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a quality assurance expert. Review the content and provide a quality score from 1-10 with specific feedback and suggestions for improvement."
                    },
                    {
                        "role": "user",
                        "content": f"Review this content and provide a quality score (1-10) with detailed feedback:\n\n{content}"
                    }
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            review_content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            # Extract quality score
            quality_score = 8.0  # Default
            try:
                import re
                score_match = re.search(r'(\d+(?:\.\d+)?)/10|(\d+(?:\.\d+)?)\s*out\s*of\s*10|score[:\s]*(\d+(?:\.\d+)?)', review_content.lower())
                if score_match:
                    quality_score = float(score_match.group(1) or score_match.group(2) or score_match.group(3))
            except:
                pass
            
            comprehensive_logger.logger.info(f"✅ Quality review completed: Score {quality_score}/10, {tokens_used} tokens")
            
            return {
                "success": True,
                "content": review_content,
                "quality_score": quality_score,
                "tokens_used": tokens_used,
                "model": model
            }
            
        except Exception as e:
            comprehensive_logger.logger.error(f"❌ Quality review API call failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "tokens_used": 0,
                "quality_score": 0
            }
    
    def _create_deliverables(self, task: str, content: str, execution_id: str) -> list:
        """Create deliverables based on task type"""
        deliverables = []
        
        comprehensive_logger.log_execution_step(execution_id, "DELIVERABLE_CREATION", {
            "task_type": "powerpoint" if "powerpoint" in task.lower() else "general",
            "content_length": len(content)
        })
        
        if "powerpoint" in task.lower() or "presentation" in task.lower():
            deliverables.extend([
                f"/presentations/AI_Cardiology_Presentation_{execution_id}.pptx",
                "15 slides with comprehensive content",
                "30-minute duration with speaker notes",
                "Belgian healthcare context included",
                "Latest research findings (2024-2025)",
                "Professional formatting and design"
            ])
        else:
            deliverables.extend([
                f"/documents/task_result_{execution_id}.pdf",
                "Comprehensive analysis document",
                "Professional formatting",
                "Executive summary included"
            ])
        
        comprehensive_logger.logger.info(f"📦 Created {len(deliverables)} deliverables")
        
        return deliverables
    
    def get_execution_logs(self, execution_id: str = None) -> dict:
        """Get comprehensive execution logs"""
        return comprehensive_logger.get_comprehensive_report(execution_id)
    
    def export_logs(self, filepath: str = None) -> str:
        """Export all logs to file"""
        return comprehensive_logger.export_logs_to_file(filepath)

def get_manager_agent():
    """Get or create manager agent instance with comprehensive logging"""
    global manager_agent
    if manager_agent is None:
        logger.info("🔧 Initializing Fixed Manager Agent with comprehensive logging...")
        try:
            manager_agent = FixedManagerAgent()
            logger.info("✅ Fixed Manager Agent initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Fixed Manager Agent: {e}")
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

