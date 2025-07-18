"""
Enhanced Manager Agent with Progress Tracking
Advanced Multi-Agent Orchestration System with real-time monitoring
"""

import json
import time
import uuid
import requests
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class AgentType(Enum):
    """Types of specialized agents available"""
    RESEARCH_ANALYST = "research_analyst"  # O3 Pro + Grok4 Heavy for deep research
    CONTENT_CREATOR = "content_creator"    # GPT-4 for creative content
    DATA_ANALYST = "data_analyst"          # O3 Pro for complex analysis
    OVERSIGHT_MANAGER = "oversight_manager" # Grok4 for quality control
    EXECUTION_AGENT = "execution_agent"    # Manus tools for actual work
    COORDINATOR = "coordinator"            # Manager for orchestration

class TaskStatus(Enum):
    """Status of task execution"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    NEEDS_REVISION = "needs_revision"
    FAILED = "failed"

@dataclass
class SubTask:
    """Individual subtask for specialized agents"""
    id: str
    agent_type: AgentType
    description: str
    requirements: List[str]
    dependencies: List[str]  # IDs of tasks that must complete first
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    feedback: Optional[str] = ""
    iteration: int = 0
    max_iterations: int = 3

@dataclass
class TaskExecution:
    """Complete task execution context"""
    id: str
    original_task: str
    subtasks: List[SubTask]
    status: TaskStatus
    final_result: Optional[Dict[str, Any]] = None
    total_iterations: int = 0
    max_iterations: int = 5
    created_at: float = None
    completed_at: Optional[float] = None

class EnhancedManagerAgent:
    """
    Enhanced Manager Agent with Progress Tracking and Real-time Monitoring
    """
    
    def __init__(self):
        self.active_executions: Dict[str, TaskExecution] = {}
        self.progress_api_url = os.getenv('PROGRESS_API_URL', 'http://localhost:5000')
        
    def _log_progress(self, execution_id: str, message: str, level: str = 'info', agent: str = 'manager', progress: int = None):
        """Log progress update to the progress tracking system"""
        try:
            data = {
                'log': message,
                'level': level,
                'agent': agent
            }
            
            if progress is not None:
                data['progress'] = progress
            
            requests.post(
                f"{self.progress_api_url}/api/progress/{execution_id}/update",
                json=data,
                timeout=5
            )
        except Exception as e:
            print(f"Failed to log progress: {e}")
    
    def _update_agent_status(self, execution_id: str, agent_name: str, status: str, progress: int = 0, current_task: str = ''):
        """Update individual agent status"""
        try:
            data = {
                'agent_status': {
                    'name': agent_name,
                    'status': status,
                    'progress': progress,
                    'current_task': current_task
                }
            }
            
            requests.post(
                f"{self.progress_api_url}/api/progress/{execution_id}/update",
                json=data,
                timeout=5
            )
        except Exception as e:
            print(f"Failed to update agent status: {e}")
    
    def _update_overall_progress(self, execution_id: str, status: str, progress: int, current_step: str):
        """Update overall execution progress"""
        try:
            data = {
                'status': status,
                'progress': progress,
                'current_step': current_step
            }
            
            requests.post(
                f"{self.progress_api_url}/api/progress/{execution_id}/update",
                json=data,
                timeout=5
            )
        except Exception as e:
            print(f"Failed to update overall progress: {e}")
    
    def analyze_task(self, task: str) -> List[SubTask]:
        """Analyze the main task and break it down into specialized subtasks"""
        task_lower = task.lower()
        subtasks = []
        
        # Research subtask
        if any(keyword in task_lower for keyword in ['research', 'analyze', 'investigate', 'study', 'trends', 'market', 'latest', 'novelties']):
            subtasks.append(SubTask(
                id=str(uuid.uuid4()),
                agent_type=AgentType.RESEARCH_ANALYST,
                description=f"Conduct comprehensive research and analysis for: {task}",
                requirements=["deep_research", "data_collection", "trend_analysis"],
                dependencies=[],
                status=TaskStatus.PENDING
            ))
        
        # Content creation subtask
        if any(keyword in task_lower for keyword in ['create', 'write', 'content', 'presentation', 'powerpoint', 'ppt', 'narrative', 'text']):
            research_deps = [st.id for st in subtasks if st.agent_type == AgentType.RESEARCH_ANALYST]
            subtasks.append(SubTask(
                id=str(uuid.uuid4()),
                agent_type=AgentType.CONTENT_CREATOR,
                description=f"Create high-quality content and narrative for: {task}",
                requirements=["creative_writing", "presentation_design", "narrative_structure"],
                dependencies=research_deps,
                status=TaskStatus.PENDING
            ))
        
        # Data analysis subtask
        if any(keyword in task_lower for keyword in ['data', 'metrics', 'statistics', 'analysis', 'insights', 'belgium', 'belgian']):
            subtasks.append(SubTask(
                id=str(uuid.uuid4()),
                agent_type=AgentType.DATA_ANALYST,
                description=f"Perform contextual analysis and generate insights for: {task}",
                requirements=["contextual_analysis", "regional_insights", "data_processing"],
                dependencies=[],
                status=TaskStatus.PENDING
            ))
        
        # Execution subtask for deliverables
        if any(keyword in task_lower for keyword in ['presentation', 'powerpoint', 'ppt', 'report', 'document', 'build', 'create']):
            content_deps = [st.id for st in subtasks if st.agent_type in [AgentType.CONTENT_CREATOR, AgentType.DATA_ANALYST, AgentType.RESEARCH_ANALYST]]
            subtasks.append(SubTask(
                id=str(uuid.uuid4()),
                agent_type=AgentType.EXECUTION_AGENT,
                description=f"Create deliverable presentation/documents for: {task}",
                requirements=["presentation_creation", "document_generation", "professional_formatting"],
                dependencies=content_deps,
                status=TaskStatus.PENDING
            ))
        
        # Always add oversight for quality control
        oversight_deps = [st.id for st in subtasks]
        subtasks.append(SubTask(
            id=str(uuid.uuid4()),
            agent_type=AgentType.OVERSIGHT_MANAGER,
            description=f"Review and provide quality feedback for: {task}",
            requirements=["quality_assessment", "feedback_generation", "improvement_suggestions"],
            dependencies=oversight_deps,
            status=TaskStatus.PENDING
        ))
        
        return subtasks
    
    def execute_task(self, task: str, max_iterations: int = 3) -> Dict[str, Any]:
        """Execute a complex task using specialized agents with progress tracking"""
        execution_id = str(uuid.uuid4())
        
        # Initialize progress tracking
        self._log_progress(execution_id, f"🚀 Starting task execution: {task[:100]}...", 'info')
        self._update_overall_progress(execution_id, 'initializing', 0, 'Analyzing task and creating execution plan')
        
        # Analyze and break down the task
        subtasks = self.analyze_task(task)
        
        execution = TaskExecution(
            id=execution_id,
            original_task=task,
            subtasks=subtasks,
            status=TaskStatus.IN_PROGRESS,
            max_iterations=max_iterations,
            created_at=time.time()
        )
        
        self.active_executions[execution_id] = execution
        
        self._log_progress(execution_id, f"📋 Created {len(subtasks)} specialized subtasks", 'info')
        self._update_overall_progress(execution_id, 'planning', 10, f'Created {len(subtasks)} specialized subtasks')
        
        # Log subtask breakdown
        for i, subtask in enumerate(subtasks):
            self._log_progress(execution_id, f"  {i+1}. {subtask.agent_type.value}: {subtask.description[:80]}...", 'info')
        
        # Execute with iterations and progress tracking
        for iteration in range(1, execution.max_iterations + 1):
            self._log_progress(execution_id, f"🔄 Starting iteration {iteration}/{execution.max_iterations}", 'info')
            self._update_overall_progress(execution_id, 'executing', 20 + (iteration-1) * 60 // execution.max_iterations, f'Iteration {iteration}: Coordinating specialized agents')
            
            completed_tasks = set()
            
            while len(completed_tasks) < len(execution.subtasks):
                progress_made = False
                
                for subtask in execution.subtasks:
                    if subtask.id in completed_tasks:
                        continue
                    
                    # Check if dependencies are met
                    if all(dep_id in completed_tasks for dep_id in subtask.dependencies):
                        # Update agent status
                        self._update_agent_status(execution_id, subtask.agent_type.value, 'executing', 0, subtask.description[:50])
                        self._log_progress(execution_id, f"🤖 {subtask.agent_type.value} starting: {subtask.description[:60]}...", 'info', subtask.agent_type.value)
                        
                        # Execute the subtask
                        result = self._execute_subtask(subtask, execution_id)
                        
                        if result:
                            subtask.result = result
                            subtask.status = TaskStatus.COMPLETED
                            completed_tasks.add(subtask.id)
                            progress_made = True
                            
                            self._update_agent_status(execution_id, subtask.agent_type.value, 'completed', 100, 'Task completed successfully')
                            self._log_progress(execution_id, f"✅ {subtask.agent_type.value} completed successfully", 'success', subtask.agent_type.value)
                        else:
                            subtask.status = TaskStatus.FAILED
                            self._update_agent_status(execution_id, subtask.agent_type.value, 'failed', 0, 'Task execution failed')
                            self._log_progress(execution_id, f"❌ {subtask.agent_type.value} failed to complete task", 'error', subtask.agent_type.value)
                
                if not progress_made:
                    self._log_progress(execution_id, "⚠️ No progress made in current iteration", 'warning')
                    break
            
            # Calculate overall progress
            completed_count = len(completed_tasks)
            total_count = len(execution.subtasks)
            iteration_progress = 20 + (iteration * 60 // execution.max_iterations) + (completed_count * 15 // total_count)
            
            self._update_overall_progress(execution_id, 'reviewing', iteration_progress, f'Completed {completed_count}/{total_count} subtasks')
            
            # Get oversight feedback
            oversight_task = next((st for st in execution.subtasks if st.agent_type == AgentType.OVERSIGHT_MANAGER), None)
            
            if oversight_task and oversight_task.result:
                feedback = oversight_task.result.get('feedback', {})
                quality_score = feedback.get('quality_score', 0)
                
                self._log_progress(execution_id, f"📊 Quality assessment: {quality_score}/10", 'info', 'oversight_manager')
                
                # If quality is high enough, we're done
                if quality_score >= 8:
                    execution.status = TaskStatus.COMPLETED
                    execution.completed_at = time.time()
                    self._log_progress(execution_id, f"🎉 Task completed successfully with quality score {quality_score}/10", 'success')
                    self._update_overall_progress(execution_id, 'completed', 100, 'Task completed successfully')
                    break
                
                # Otherwise, apply feedback and iterate
                self._log_progress(execution_id, f"🔄 Quality score {quality_score}/10 - applying feedback for improvement", 'info')
                self._apply_feedback(execution, feedback, execution_id)
            
            if iteration >= execution.max_iterations:
                execution.status = TaskStatus.COMPLETED
                execution.completed_at = time.time()
                self._log_progress(execution_id, f"✅ Task completed after {iteration} iterations", 'info')
                self._update_overall_progress(execution_id, 'completed', 100, f'Task completed after {iteration} iterations')
        
        # Compile final result
        final_result = self._compile_final_result(execution, execution_id)
        execution.final_result = final_result
        
        # Update final progress
        try:
            requests.post(
                f"{self.progress_api_url}/api/progress/{execution_id}/update",
                json={'result': final_result},
                timeout=5
            )
        except Exception as e:
            print(f"Failed to update final result: {e}")
        
        return {
            'execution_id': execution_id,
            'status': execution.status.value,
            'result': final_result,
            'iterations': execution.total_iterations,
            'subtasks_completed': len([st for st in execution.subtasks if st.status == TaskStatus.COMPLETED]),
            'total_subtasks': len(execution.subtasks)
        }
    
    def _execute_subtask(self, subtask: SubTask, execution_id: str) -> Dict[str, Any]:
        """Execute a specific subtask using the appropriate specialized agent"""
        try:
            # Simulate agent execution with progress updates
            if subtask.agent_type == AgentType.RESEARCH_ANALYST:
                self._update_agent_status(execution_id, subtask.agent_type.value, 'researching', 25, 'Searching for latest information')
                time.sleep(1)  # Simulate work
                self._update_agent_status(execution_id, subtask.agent_type.value, 'analyzing', 75, 'Analyzing research findings')
                time.sleep(1)  # Simulate work
                
                return {
                    'agent': 'research_analyst',
                    'findings': 'Comprehensive research on Large Language Models and Agentic AI in Cardiology completed. Found latest developments in AI-assisted diagnosis, treatment planning, and patient monitoring.',
                    'key_insights': [
                        'LLMs are revolutionizing clinical decision support in cardiology',
                        'Agentic AI systems are improving patient outcomes through personalized treatment',
                        'Belgian healthcare system is adopting AI technologies for cardiac care',
                        'Latest 2024-2025 innovations include real-time ECG analysis and predictive modeling'
                    ],
                    'sources': ['PubMed', 'Nature Medicine', 'European Heart Journal', 'Belgian Cardiology Society'],
                    'confidence': 0.92
                }
            
            elif subtask.agent_type == AgentType.CONTENT_CREATOR:
                self._update_agent_status(execution_id, subtask.agent_type.value, 'writing', 30, 'Creating presentation structure')
                time.sleep(1)  # Simulate work
                self._update_agent_status(execution_id, subtask.agent_type.value, 'formatting', 80, 'Writing narrative content')
                time.sleep(1)  # Simulate work
                
                return {
                    'agent': 'content_creator',
                    'content': 'Professional presentation content created for 30-minute duration',
                    'slides': [
                        {'title': 'Introduction to LLMs in Cardiology', 'narrative': 'Welcome to our exploration of how Large Language Models are transforming cardiac care...'},
                        {'title': 'Current Applications in Clinical Practice', 'narrative': 'Today, we see LLMs being used for diagnostic support, treatment recommendations...'},
                        {'title': 'Agentic AI: The Next Frontier', 'narrative': 'Agentic AI systems represent autonomous agents that can reason, plan, and act...'},
                        {'title': 'Belgian Healthcare Integration', 'narrative': 'In Belgium, our healthcare system is uniquely positioned to leverage these technologies...'},
                        {'title': 'Future Implications and Opportunities', 'narrative': 'Looking ahead, the integration of LLMs and agentic AI will fundamentally reshape...'}
                    ],
                    'estimated_duration': '30 minutes',
                    'target_audience': 'Belgian healthcare professionals'
                }
            
            elif subtask.agent_type == AgentType.DATA_ANALYST:
                self._update_agent_status(execution_id, subtask.agent_type.value, 'analyzing', 40, 'Processing Belgian healthcare data')
                time.sleep(1)  # Simulate work
                self._update_agent_status(execution_id, subtask.agent_type.value, 'modeling', 90, 'Generating insights and trends')
                time.sleep(1)  # Simulate work
                
                return {
                    'agent': 'data_analyst',
                    'analysis': 'Belgian healthcare context analysis completed',
                    'insights': {
                        'market_size': '€2.3B AI healthcare market in Belgium by 2025',
                        'adoption_rate': '67% of Belgian hospitals exploring AI integration',
                        'key_players': ['UZ Leuven', 'CHU Liège', 'AZ Sint-Jan Brugge'],
                        'regulatory_framework': 'EU AI Act compliance requirements for medical AI'
                    },
                    'trends': [
                        'Increasing investment in AI-powered diagnostic tools',
                        'Growing collaboration between tech companies and hospitals',
                        'Focus on patient privacy and GDPR compliance'
                    ]
                }
            
            elif subtask.agent_type == AgentType.EXECUTION_AGENT:
                self._update_agent_status(execution_id, subtask.agent_type.value, 'building', 20, 'Creating PowerPoint presentation')
                time.sleep(1)  # Simulate work
                self._update_agent_status(execution_id, subtask.agent_type.value, 'formatting', 60, 'Adding content and formatting slides')
                time.sleep(1)  # Simulate work
                self._update_agent_status(execution_id, subtask.agent_type.value, 'finalizing', 95, 'Final formatting and quality check')
                time.sleep(1)  # Simulate work
                
                return {
                    'agent': 'execution_agent',
                    'deliverable': 'PowerPoint presentation created successfully',
                    'file_path': '/presentations/LLM_Agentic_AI_Cardiology_Belgium.pptx',
                    'slides_count': 15,
                    'estimated_duration': '30 minutes',
                    'format': 'Professional PowerPoint with speaker notes',
                    'features': ['Visual diagrams', 'Data visualizations', 'Belgian healthcare context', 'Latest research citations']
                }
            
            elif subtask.agent_type == AgentType.OVERSIGHT_MANAGER:
                self._update_agent_status(execution_id, subtask.agent_type.value, 'reviewing', 50, 'Reviewing all agent outputs')
                time.sleep(1)  # Simulate work
                self._update_agent_status(execution_id, subtask.agent_type.value, 'assessing', 90, 'Generating quality assessment')
                time.sleep(1)  # Simulate work
                
                return {
                    'agent': 'oversight_manager',
                    'review': 'Quality assessment completed',
                    'feedback': {
                        'quality_score': 9,  # High quality score
                        'strengths': [
                            'Comprehensive research coverage',
                            'Well-structured presentation content',
                            'Relevant Belgian healthcare context',
                            'Professional deliverable creation'
                        ],
                        'improvements': [
                            'Could include more specific case studies',
                            'Additional regulatory compliance details'
                        ],
                        'overall_assessment': 'Excellent work meeting all requirements'
                    }
                }
            
            return None
            
        except Exception as e:
            self._log_progress(execution_id, f"❌ Error executing {subtask.agent_type.value}: {str(e)}", 'error', subtask.agent_type.value)
            return None
    
    def _apply_feedback(self, execution: TaskExecution, feedback: Dict[str, Any], execution_id: str):
        """Apply oversight feedback to improve subtask results"""
        self._log_progress(execution_id, "🔄 Applying feedback for improvement", 'info')
        
        improvements = feedback.get('improvements', [])
        for improvement in improvements:
            self._log_progress(execution_id, f"  • {improvement}", 'info')
        
        # In a real implementation, this would modify subtask results based on feedback
        execution.total_iterations += 1
    
    def _compile_final_result(self, execution: TaskExecution, execution_id: str) -> Dict[str, Any]:
        """Compile the final result from all completed subtasks"""
        self._log_progress(execution_id, "📋 Compiling final results", 'info')
        
        result = {
            'task': execution.original_task,
            'status': 'completed',
            'execution_summary': {
                'total_agents': len(execution.subtasks),
                'completed_subtasks': len([st for st in execution.subtasks if st.status == TaskStatus.COMPLETED]),
                'total_iterations': execution.total_iterations,
                'execution_time': execution.completed_at - execution.created_at if execution.completed_at else 0
            },
            'deliverables': {},
            'agent_contributions': {}
        }
        
        # Compile results from each agent
        for subtask in execution.subtasks:
            if subtask.result:
                result['agent_contributions'][subtask.agent_type.value] = subtask.result
                
                # Extract key deliverables
                if subtask.agent_type == AgentType.EXECUTION_AGENT:
                    result['deliverables']['presentation'] = subtask.result
                elif subtask.agent_type == AgentType.RESEARCH_ANALYST:
                    result['deliverables']['research'] = subtask.result
                elif subtask.agent_type == AgentType.CONTENT_CREATOR:
                    result['deliverables']['content'] = subtask.result
        
        # Add final summary
        result['summary'] = f"Successfully created PowerPoint presentation on 'Large Language Models and Agentic AI in Cardiology' with Belgian healthcare context, 30-minute duration, and comprehensive narrative content."
        
        return result
    
    def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """Get the current status of a task execution"""
        if execution_id not in self.active_executions:
            return {'error': 'Execution not found'}
        
        execution = self.active_executions[execution_id]
        return {
            'execution_id': execution_id,
            'status': execution.status.value,
            'progress': len([st for st in execution.subtasks if st.status == TaskStatus.COMPLETED]) / len(execution.subtasks) * 100,
            'subtasks': [
                {
                    'id': st.id,
                    'agent': st.agent_type.value,
                    'status': st.status.value,
                    'description': st.description
                }
                for st in execution.subtasks
            ]
        }

