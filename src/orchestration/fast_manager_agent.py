"""
Fast Manager Agent for AgentFlow
Optimized for quick response times with timeout handling
"""
import os
import time
import random
import logging
import traceback
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Import the deliverable creator
from .deliverable_creator import DeliverableCreator

logger = logging.getLogger(__name__)

class FastManagerAgent:
    """Manager Agent optimized for fast execution with timeout handling"""
    
    def __init__(self):
        self.execution_logs = []
        self.deliverable_creator = DeliverableCreator()
        
        # Initialize AI clients
        self.openai_client = None
        self.grok_client = None
        self._initialize_ai_clients()
        
        logger.info("⚡ Fast Manager Agent initialized")
    
    def _initialize_ai_clients(self):
        """Initialize AI clients with proper error handling"""
        try:
            # Try OpenAI first
            openai_key = os.getenv('OPENAI_API_KEY')
            if openai_key and openai_key.startswith('sk-'):
                try:
                    from openai import OpenAI
                    self.openai_client = OpenAI(api_key=openai_key)
                    logger.info("✅ OpenAI client initialized successfully")
                except Exception as e:
                    logger.warning(f"⚠️ OpenAI client initialization failed: {e}")
            
            # Try Grok as backup
            grok_key = os.getenv('GROK_API_KEY')
            if grok_key:
                try:
                    from openai import OpenAI
                    self.grok_client = OpenAI(
                        api_key=grok_key,
                        base_url="https://api.x.ai/v1"
                    )
                    logger.info("✅ Grok client initialized successfully")
                except Exception as e:
                    logger.warning(f"⚠️ Grok client initialization failed: {e}")
            
            if not self.openai_client and not self.grok_client:
                logger.error("❌ No AI clients available - check API keys")
                
        except Exception as e:
            logger.error(f"❌ AI client initialization failed: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the manager agent"""
        openai_status = "✅ Available" if self.openai_client else "❌ Not Available"
        grok_status = "✅ Available" if self.grok_client else "❌ Not Available"
        
        return {
            "status": "FAST_MANAGER_ACTIVE",
            "ai_clients": {
                "openai": openai_status,
                "grok": grok_status
            },
            "deliverable_creator": "✅ Active",
            "environment_check": {
                "openai_key": "✓" if os.getenv('OPENAI_API_KEY') else "✗",
                "grok_key": "✓" if os.getenv('GROK_API_KEY') else "✗",
                "flask_env": os.getenv('FLASK_ENV', 'unknown')
            },
            "execution_logs": len(self.execution_logs),
            "deliverables_info": self.deliverable_creator.get_deliverables_info()
        }
    
    def execute_task(self, task: str, max_iterations: int = 3) -> Dict[str, Any]:
        """Execute task with timeout handling and fast response"""
        execution_id = f"exec_{int(time.time() * 1000)}"
        start_time = time.time()
        
        logger.info(f"⚡ Starting FAST execution for task: {task[:100]}...")
        logger.info(f"   Execution ID: {execution_id}")
        
        # Use ThreadPoolExecutor with timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                # Submit the task with a timeout
                future = executor.submit(self._execute_task_internal, task, execution_id, start_time)
                result = future.result(timeout=25)  # 25 second timeout
                return result
                
            except TimeoutError:
                logger.error(f"❌ Task execution timed out after 25 seconds")
                return {
                    "success": False,
                    "error": "Task execution timed out",
                    "execution_id": execution_id,
                    "execution_time": time.time() - start_time,
                    "validation": "EXECUTION_TIMEOUT"
                }
            except Exception as e:
                logger.error(f"❌ Task execution failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "execution_id": execution_id,
                    "execution_time": time.time() - start_time,
                    "validation": "EXECUTION_FAILED"
                }
    
    def _execute_task_internal(self, task: str, execution_id: str, start_time: float) -> Dict[str, Any]:
        """Internal task execution with optimized AI calls"""
        total_api_calls = 0
        total_tokens = 0
        agents_used = []
        
        try:
            # Get available AI client
            ai_client, primary_client = self._get_available_ai_client()
            if not ai_client:
                # Create deliverables without AI if no client available
                logger.warning("⚠️ No AI client available, creating basic deliverables...")
                created_files = self.deliverable_creator.create_basic_deliverables(task, execution_id)
                
                return {
                    "success": True,
                    "execution_id": execution_id,
                    "execution_time": time.time() - start_time,
                    "result": {
                        "status": "completed_without_ai",
                        "deliverables": created_files,
                        "summary": f"Created basic deliverables for: {task}",
                        "agents_used": ["Basic Creator"],
                        "api_calls": 0,
                        "tokens_used": 0
                    }
                }
            
            logger.info(f"🤖 Using AI client: {primary_client}")
            
            # Single optimized AI call for content generation
            logger.info("🚀 Generating content with AI...")
            content_response = ai_client.chat.completions.create(
                model="gpt-4o-mini" if primary_client == "openai" else "grok-beta",
                messages=[
                    {"role": "system", "content": "You are an expert content creator. Create comprehensive, detailed content for the user's request. Be specific and actionable."},
                    {"role": "user", "content": f"Create detailed content for this task: {task}"}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            total_api_calls += 1
            total_tokens += content_response.usage.total_tokens
            agents_used.append("AI Content Creator")
            
            content_result = content_response.choices[0].message.content
            logger.info(f"✅ Content generation completed: {len(content_result)} characters")
            
            # CREATE ACTUAL DELIVERABLE FILES
            logger.info("📁 Creating actual deliverable files...")
            agents_used.append("Deliverable Creator")
            
            # Create real files based on task type
            created_files = self.deliverable_creator.create_deliverables_from_task(
                content_result, task, execution_id
            )
            
            logger.info(f"✅ Created {len(created_files)} actual deliverable files:")
            for file_path in created_files:
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    logger.info(f"   📄 {file_path} ({file_size} bytes)")
                else:
                    logger.warning(f"   ⚠️ {file_path} (file not found)")
            
            # Verify files were created
            verified_files = []
            for file_path in created_files:
                if os.path.exists(file_path):
                    verified_files.append(file_path)
                    logger.info(f"✅ Verified: {file_path}")
                else:
                    logger.error(f"❌ File not created: {file_path}")
            
            execution_time = time.time() - start_time
            
            # Log execution details
            self.execution_logs.append({
                "execution_id": execution_id,
                "task": task[:200],
                "execution_time": execution_time,
                "api_calls": total_api_calls,
                "tokens_used": total_tokens,
                "agents_used": agents_used,
                "files_created": len(verified_files),
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"🎉 FAST execution completed in {execution_time:.2f}s")
            logger.info(f"   📊 API calls: {total_api_calls}, Tokens: {total_tokens}")
            logger.info(f"   📁 Files created: {len(verified_files)}")
            
            return {
                "success": True,
                "execution_id": execution_id,
                "execution_time": execution_time,
                "result": {
                    "status": "completed",
                    "deliverables": verified_files,
                    "summary": f"Successfully created deliverables for: {task}",
                    "execution_summary": {
                        "total_agents": len(agents_used),
                        "execution_time": execution_time,
                        "api_calls": total_api_calls,
                        "tokens_used": total_tokens
                    },
                    "agent_results": {
                        "content_creator": {
                            "status": "completed",
                            "output_length": len(content_result),
                            "tokens_used": total_tokens
                        },
                        "deliverable_creator": {
                            "status": "completed",
                            "files_created": len(verified_files)
                        }
                    }
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ FAST execution failed after {execution_time:.2f}s: {e}")
            logger.error(f"   Traceback: {traceback.format_exc()}")
            
            return {
                "success": False,
                "execution_id": execution_id,
                "execution_time": execution_time,
                "error": str(e),
                "validation": "FAST_EXECUTION_FAILED",
                "partial_results": {
                    "api_calls": total_api_calls,
                    "tokens_used": total_tokens,
                    "agents_used": agents_used
                }
            }
    
    def _get_available_ai_client(self):
        """Get the first available AI client"""
        if self.openai_client:
            return self.openai_client, "openai"
        elif self.grok_client:
            return self.grok_client, "grok"
        else:
            return None, None
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history"""
        return self.execution_logs
    
    def get_deliverables_directory(self) -> str:
        """Get the deliverables directory path"""
        return str(self.deliverable_creator.base_output_dir)
    
    def list_created_files(self) -> List[Dict[str, Any]]:
        """List all created deliverable files"""
        files_info = []
        
        try:
            base_dir = self.deliverable_creator.base_output_dir
            
            for subdir in ["presentations", "documents", "reports", "data"]:
                subdir_path = base_dir / subdir
                if subdir_path.exists():
                    for file_path in subdir_path.iterdir():
                        if file_path.is_file():
                            files_info.append({
                                "filename": file_path.name,
                                "full_path": str(file_path),
                                "size": file_path.stat().st_size,
                                "created": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                                "type": subdir,
                                "extension": file_path.suffix
                            })
        except Exception as e:
            logger.error(f"❌ Failed to list created files: {e}")
        
        return files_info

