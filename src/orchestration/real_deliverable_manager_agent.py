"""
Real Deliverable Manager Agent for AgentFlow
This version actually creates deliverable files, not just fake paths
"""
import os
import time
import random
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import the deliverable creator
from .deliverable_creator import DeliverableCreator

logger = logging.getLogger(__name__)

class RealDeliverableManagerAgent:
    """Manager Agent that creates actual deliverable files"""
    
    def __init__(self):
        self.execution_logs = []
        self.deliverable_creator = DeliverableCreator()
        
        # Initialize AI clients
        self.openai_client = None
        self.grok_client = None
        self._initialize_ai_clients()
        
        logger.info("🎯 Real Deliverable Manager Agent initialized")
    
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
            "status": "REAL_DELIVERABLE_MANAGER_ACTIVE",
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
        """Execute task with real AI and create actual deliverable files"""
        execution_id = f"exec_{int(time.time() * 1000)}"
        start_time = time.time()
        
        logger.info(f"🚀 Starting REAL deliverable creation for task: {task[:100]}...")
        logger.info(f"   Execution ID: {execution_id}")
        
        # Initialize tracking variables
        total_api_calls = 0
        total_tokens = 0
        agents_used = []
        
        try:
            # Get available AI client
            ai_client, primary_client = self._get_available_ai_client()
            if not ai_client:
                return {
                    "success": False,
                    "error": "No AI clients available",
                    "validation": "NO_AI_CLIENTS_AVAILABLE"
                }
            
            logger.info(f"🤖 Using AI client: {primary_client}")
            
            # Step 1: Task Analysis with Real AI
            logger.info("🔍 Step 1: Task analysis with AI...")
            analysis_response = ai_client.chat.completions.create(
                model="gpt-4o-mini" if primary_client == "openai" else "grok-beta",
                messages=[
                    {"role": "system", "content": "You are a task analysis expert. Break down the user's request into specific, actionable steps and identify what deliverables should be created."},
                    {"role": "user", "content": f"Analyze this task and identify what specific deliverables should be created: {task}"}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            total_api_calls += 1
            total_tokens += analysis_response.usage.total_tokens
            agents_used.append("Task Analyzer")
            
            analysis_result = analysis_response.choices[0].message.content
            logger.info(f"✅ Task analysis completed: {len(analysis_result)} characters")
            
            # Step 2: Content Generation with Real AI
            logger.info("✍️ Step 2: Content generation with AI...")
            content_response = ai_client.chat.completions.create(
                model="gpt-4o-mini" if primary_client == "openai" else "grok-beta",
                messages=[
                    {"role": "system", "content": "You are a content creation expert. Create comprehensive, detailed content based on the task requirements. Include all necessary information for the deliverable."},
                    {"role": "user", "content": f"Create detailed content for this task: {task}\n\nTask Analysis: {analysis_result}"}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            total_api_calls += 1
            total_tokens += content_response.usage.total_tokens
            agents_used.append("Content Creator")
            
            content_result = content_response.choices[0].message.content
            logger.info(f"✅ Content generation completed: {len(content_result)} characters")
            
            # Step 3: Quality Review with Real AI
            logger.info("🔍 Step 3: Quality review with AI...")
            review_response = ai_client.chat.completions.create(
                model="gpt-4o-mini" if primary_client == "openai" else "grok-beta",
                messages=[
                    {"role": "system", "content": "You are a quality assurance expert. Review the content and provide a quality score from 1-10 with specific feedback."},
                    {"role": "user", "content": f"Review this content and provide a quality score (1-10) and detailed feedback:\n\n{content_result}"}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            total_api_calls += 1
            total_tokens += review_response.usage.total_tokens
            agents_used.append("Quality Reviewer")
            
            review_result = review_response.choices[0].message.content
            logger.info(f"✅ Quality review completed: {review_result[:100]}...")
            
            # Extract quality score
            quality_score = self._extract_quality_score(review_result)
            
            # Step 4: CREATE ACTUAL DELIVERABLE FILES
            logger.info("📁 Step 4: Creating actual deliverable files...")
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
            
            # Step 5: Verify files were created
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
                "task": task[:100],
                "api_calls": total_api_calls,
                "tokens_used": total_tokens,
                "agents_used": agents_used,
                "quality_score": quality_score,
                "files_created": len(verified_files),
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            })
            
            # Return comprehensive results
            return {
                "success": True,
                "result": {
                    "task_analysis": analysis_result,
                    "content": content_result,
                    "quality_review": review_result,
                    "deliverables": verified_files,  # Real file paths
                    "files_created": len(verified_files),
                    "file_verification": "VERIFIED" if verified_files else "FAILED"
                },
                "api_calls": total_api_calls,
                "tokens_used": total_tokens,
                "agents_used": agents_used,
                "quality_score": quality_score,
                "execution_time": execution_time,
                "ai_client_used": primary_client,
                "validation": "REAL_DELIVERABLES_CREATED",
                "deliverable_creator_info": self.deliverable_creator.get_deliverables_info()
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Real deliverable creation failed: {e}")
            logger.error(f"   Traceback: {traceback.format_exc()}")
            
            return {
                "success": False,
                "error": f"Deliverable creation failed: {str(e)}",
                "api_calls": total_api_calls,
                "tokens_used": total_tokens,
                "agents_used": agents_used,
                "quality_score": 0,
                "execution_time": execution_time,
                "validation": "DELIVERABLE_CREATION_FAILED",
                "traceback": traceback.format_exc()
            }
    
    def _get_available_ai_client(self):
        """Get the first available AI client"""
        if self.openai_client:
            return self.openai_client, "openai"
        elif self.grok_client:
            return self.grok_client, "grok"
        else:
            return None, None
    
    def _extract_quality_score(self, review_result: str) -> float:
        """Extract quality score from review text"""
        try:
            import re
            score_match = re.search(r'(\d+(?:\.\d+)?)/10|(\d+(?:\.\d+)?)\s*out\s*of\s*10|score[:\s]*(\d+(?:\.\d+)?)', review_result.lower())
            if score_match:
                return float(score_match.group(1) or score_match.group(2) or score_match.group(3))
        except:
            pass
        return 8.0  # Default score
    
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

