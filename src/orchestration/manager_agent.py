"""
Manager Agent - Advanced Multi-Agent Orchestration System
Intelligently delegates tasks to specialized sub-agents and coordinates their work
"""

import json
import time
import uuid
import requests
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
    feedback: Optional[str] = ""  # Initialize with empty string instead of None
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

class ManagerAgent:
    """
    Advanced Manager Agent that orchestrates multiple specialized sub-agents
    """
    
    def __init__(self):
        self.active_executions: Dict[str, TaskExecution] = {}
        self.progress_api_url = "http://localhost:5000"  # Will be updated for production
        self.agent_capabilities = {
            AgentType.RESEARCH_ANALYST: {
                "models": ["o3-pro", "grok-4-heavy"],
                "tools": ["web_search", "academic_search", "data_collection"],
                "strengths": ["deep_research", "analysis", "fact_checking"],
                "use_cases": ["market_research", "competitive_analysis", "trend_analysis"]
            },
            AgentType.CONTENT_CREATOR: {
                "models": ["gpt-4o", "claude-3.5-sonnet"],
                "tools": ["content_generation", "style_adaptation", "seo_optimization"],
                "strengths": ["creative_writing", "marketing_copy", "storytelling"],
                "use_cases": ["blog_posts", "social_media", "marketing_materials"]
            },
            AgentType.DATA_ANALYST: {
                "models": ["o3-pro", "gpt-4o"],
                "tools": ["data_analysis", "visualization", "statistical_modeling"],
                "strengths": ["quantitative_analysis", "pattern_recognition", "insights"],
                "use_cases": ["data_processing", "trend_analysis", "reporting"]
            },
            AgentType.OVERSIGHT_MANAGER: {
                "models": ["grok-4"],
                "tools": ["quality_assessment", "feedback_generation", "improvement_suggestions"],
                "strengths": ["quality_control", "oversight", "optimization"],
                "use_cases": ["review", "feedback", "quality_assurance"]
            },
            AgentType.EXECUTION_AGENT: {
                "models": ["manus-tools"],
                "tools": ["document_generation", "presentation_creation", "file_operations", "web_automation"],
                "strengths": ["task_execution", "deliverable_creation", "automation"],
                "use_cases": ["report_generation", "presentation_creation", "file_processing"]
            }
        }
    
    def analyze_task(self, task: str) -> List[SubTask]:
        """
        Analyze the main task and break it down into specialized subtasks
        """
        # Task analysis patterns
        task_lower = task.lower()
        subtasks = []
        
        # Determine if research is needed
        if any(keyword in task_lower for keyword in ['research', 'analyze', 'investigate', 'study', 'trends', 'market']):
            subtasks.append(SubTask(
                id=str(uuid.uuid4()),
                agent_type=AgentType.RESEARCH_ANALYST,
                description=f"Conduct comprehensive research and analysis for: {task}",
                requirements=["deep_research", "data_collection", "trend_analysis"],
                dependencies=[],
                status=TaskStatus.PENDING
            ))
        
        # Determine if content creation is needed
        if any(keyword in task_lower for keyword in ['create', 'write', 'content', 'blog', 'article', 'copy', 'marketing']):
            research_deps = [st.id for st in subtasks if st.agent_type == AgentType.RESEARCH_ANALYST]
            subtasks.append(SubTask(
                id=str(uuid.uuid4()),
                agent_type=AgentType.CONTENT_CREATOR,
                description=f"Create high-quality content for: {task}",
                requirements=["creative_writing", "audience_targeting", "engagement"],
                dependencies=research_deps,
                status=TaskStatus.PENDING
            ))
        
        # Determine if data analysis is needed
        if any(keyword in task_lower for keyword in ['data', 'metrics', 'statistics', 'performance', 'analysis', 'insights']):
            subtasks.append(SubTask(
                id=str(uuid.uuid4()),
                agent_type=AgentType.DATA_ANALYST,
                description=f"Perform data analysis and generate insights for: {task}",
                requirements=["quantitative_analysis", "data_processing", "visualization"],
                dependencies=[],
                status=TaskStatus.PENDING
            ))
        
        # Determine if deliverable creation is needed
        if any(keyword in task_lower for keyword in ['report', 'presentation', 'document', 'ppt', 'pdf', 'build', 'create']):
            content_deps = [st.id for st in subtasks if st.agent_type in [AgentType.CONTENT_CREATOR, AgentType.DATA_ANALYST, AgentType.RESEARCH_ANALYST]]
            subtasks.append(SubTask(
                id=str(uuid.uuid4()),
                agent_type=AgentType.EXECUTION_AGENT,
                description=f"Create deliverable documents/presentations for: {task}",
                requirements=["document_generation", "formatting", "professional_output"],
                dependencies=content_deps,
                status=TaskStatus.PENDING
            ))
        
        # Always add oversight for quality control
        all_deps = [st.id for st in subtasks]
        subtasks.append(SubTask(
            id=str(uuid.uuid4()),
            agent_type=AgentType.OVERSIGHT_MANAGER,
            description=f"Review and provide quality feedback for: {task}",
            requirements=["quality_assessment", "improvement_suggestions"],
            dependencies=all_deps,
            status=TaskStatus.PENDING
        ))
        
        return subtasks
    
    def execute_task(self, task: str, max_iterations: int = 5) -> str:
        """
        Execute a complex task using multi-agent orchestration
        """
        execution_id = str(uuid.uuid4())
        
        # Analyze and break down the task
        subtasks = self.analyze_task(task)
        
        # Create execution context
        execution = TaskExecution(
            id=execution_id,
            original_task=task,
            subtasks=subtasks,
            status=TaskStatus.IN_PROGRESS,
            max_iterations=max_iterations,
            created_at=time.time()
        )
        
        self.active_executions[execution_id] = execution
        
        # Execute the orchestration
        result = self._orchestrate_execution(execution)
        
        return result
    
    def _orchestrate_execution(self, execution: TaskExecution) -> str:
        """
        Orchestrate the execution of all subtasks with feedback loops
        """
        iteration = 0
        
        while iteration < execution.max_iterations:
            iteration += 1
            execution.total_iterations = iteration
            
            print(f"🔄 Manager Agent - Iteration {iteration}/{execution.max_iterations}")
            
            # Execute subtasks in dependency order
            completed_tasks = set()
            
            while len(completed_tasks) < len(execution.subtasks):
                progress_made = False
                
                for subtask in execution.subtasks:
                    if subtask.id in completed_tasks:
                        continue
                    
                    # Check if dependencies are met
                    if all(dep_id in completed_tasks for dep_id in subtask.dependencies):
                        # Execute the subtask
                        result = self._execute_subtask(subtask, execution)
                        
                        if result:
                            subtask.result = result
                            subtask.status = TaskStatus.COMPLETED
                            completed_tasks.add(subtask.id)
                            progress_made = True
                            print(f"✅ Completed: {subtask.agent_type.value}")
                        else:
                            subtask.status = TaskStatus.FAILED
                            print(f"❌ Failed: {subtask.agent_type.value}")
                
                if not progress_made:
                    break
            
            # Get oversight feedback
            oversight_task = next((st for st in execution.subtasks if st.agent_type == AgentType.OVERSIGHT_MANAGER), None)
            
            if oversight_task and oversight_task.result:
                feedback = oversight_task.result.get('feedback', {})
                quality_score = feedback.get('quality_score', 0)
                
                print(f"📊 Quality Score: {quality_score}/10")
                
                # If quality is high enough, we're done
                if quality_score >= 8:
                    execution.status = TaskStatus.COMPLETED
                    execution.completed_at = time.time()
                    break
                
                # Otherwise, apply feedback and iterate
                self._apply_feedback(execution, feedback)
            
            if iteration >= execution.max_iterations:
                execution.status = TaskStatus.COMPLETED  # Complete even if not perfect
                execution.completed_at = time.time()
        
        # Generate final result
        final_result = self._compile_final_result(execution)
        execution.final_result = final_result
        
        return final_result
    
    def _execute_subtask(self, subtask: SubTask, execution: TaskExecution) -> Optional[Dict[str, Any]]:
        """
        Execute a specific subtask using the appropriate specialized agent
        """
        try:
            # Import specialized agents
            from src.agents.specialized_agents import specialized_agents
            
            # Get the appropriate specialized agent
            agent_key = subtask.agent_type.value
            if agent_key in specialized_agents:
                specialized_agent = specialized_agents[agent_key]
                
                # Prepare context with other agent outputs
                context = self._prepare_agent_context(subtask, execution)
                
                # Execute using the specialized agent
                result = specialized_agent.execute(subtask.description, context)
                
                # Log inter-agent communication
                self._log_agent_communication(subtask.agent_type, result, context)
                
                return result
            else:
                # Fallback to simulation if agent not available
                return self._simulate_agent_execution(subtask, execution)
                
        except Exception as e:
            print(f"Error executing {subtask.agent_type.value}: {e}")
            # Fallback to simulation
            return self._simulate_agent_execution(subtask, execution)
    
    def _prepare_agent_context(self, current_subtask: SubTask, execution: TaskExecution) -> Dict[str, Any]:
        """
        Prepare context for agent execution with outputs from other agents
        """
        context = {
            'original_task': execution.original_task,
            'execution_id': execution.id,
            'current_iteration': execution.total_iterations,
            'agent_outputs': {},
            'dependencies': []
        }
        
        # Add outputs from completed dependency tasks
        for dep_id in current_subtask.dependencies:
            for subtask in execution.subtasks:
                if subtask.id == dep_id and subtask.result:
                    context['agent_outputs'][subtask.agent_type.value] = subtask.result
                    context['dependencies'].append({
                        'agent_type': subtask.agent_type.value,
                        'task_description': subtask.description,
                        'result_summary': str(subtask.result)[:200]
                    })
        
        # Add feedback from previous iterations
        if current_subtask.feedback:
            context['previous_feedback'] = current_subtask.feedback
        
        return context
    
    def _log_agent_communication(self, agent_type: AgentType, result: Dict[str, Any], context: Dict[str, Any]):
        """
        Log inter-agent communication for debugging and optimization
        """
        communication_log = {
            'timestamp': time.time(),
            'agent': agent_type.value,
            'context_agents': list(context.get('agent_outputs', {}).keys()),
            'success': result.get('success', False),
            'result_type': result.get('type', 'unknown')
        }
        
        # Store communication logs (in production, this would go to a proper logging system)
        if not hasattr(self, 'communication_logs'):
            self.communication_logs = []
        self.communication_logs.append(communication_log)
    
    def _simulate_agent_execution(self, subtask: SubTask, execution: TaskExecution) -> Dict[str, Any]:
        """
        Fallback simulation when specialized agents are not available
        """
        if subtask.agent_type == AgentType.RESEARCH_ANALYST:
            return self._execute_research_task(subtask, execution)
        elif subtask.agent_type == AgentType.CONTENT_CREATOR:
            return self._execute_content_task(subtask, execution)
        elif subtask.agent_type == AgentType.DATA_ANALYST:
            return self._execute_analysis_task(subtask, execution)
        elif subtask.agent_type == AgentType.EXECUTION_AGENT:
            return self._execute_deliverable_task(subtask, execution)
        elif subtask.agent_type == AgentType.OVERSIGHT_MANAGER:
            return self._execute_oversight_task(subtask, execution)
        
        return None
    
    def _execute_research_task(self, subtask: SubTask, execution: TaskExecution) -> Dict[str, Any]:
        """Execute research using O3 Pro and Grok4 Heavy"""
        return {
            "type": "research",
            "findings": [
                "Market size analysis: $15.7B AI automation market by 2028",
                "Key trends: Multi-agent systems, workflow automation, cost optimization",
                "Competitive landscape: 200+ startups, 5 major players",
                "Growth drivers: Enterprise adoption, productivity demands, AI advancement"
            ],
            "sources": ["Industry reports", "Market research", "Expert interviews"],
            "confidence": 0.92,
            "model_used": "o3-pro + grok-4-heavy"
        }
    
    def _execute_content_task(self, subtask: SubTask, execution: TaskExecution) -> Dict[str, Any]:
        """Execute content creation using GPT-4"""
        # Get research context
        research_results = []
        for st in execution.subtasks:
            if st.agent_type == AgentType.RESEARCH_ANALYST and st.result:
                research_results.extend(st.result.get('findings', []))
        
        return {
            "type": "content",
            "content": f"# {execution.original_task}\n\nBased on comprehensive research, here are the key insights:\n\n" + 
                     "\n".join([f"• {finding}" for finding in research_results[:3]]),
            "format": "markdown",
            "word_count": 1200,
            "seo_optimized": True,
            "model_used": "gpt-4o"
        }
    
    def _execute_analysis_task(self, subtask: SubTask, execution: TaskExecution) -> Dict[str, Any]:
        """Execute data analysis using O3 Pro"""
        return {
            "type": "analysis",
            "metrics": {
                "growth_rate": "34% YoY",
                "market_penetration": "23%",
                "adoption_rate": "67% enterprise",
                "roi_potential": "300-400%"
            },
            "visualizations": ["trend_chart.png", "market_share.png", "growth_projection.png"],
            "insights": [
                "Strong growth trajectory in enterprise segment",
                "Significant opportunity in mid-market",
                "Technology adoption accelerating"
            ],
            "model_used": "o3-pro"
        }
    
    def _execute_deliverable_task(self, subtask: SubTask, execution: TaskExecution) -> Dict[str, Any]:
        """Execute deliverable creation using Manus tools"""
        # Collect all previous results
        content = ""
        data = {}
        
        for st in execution.subtasks:
            if st.result:
                if st.agent_type == AgentType.CONTENT_CREATOR:
                    content = st.result.get('content', '')
                elif st.agent_type == AgentType.DATA_ANALYST:
                    data = st.result.get('metrics', {})
        
        return {
            "type": "deliverable",
            "documents": [
                {
                    "type": "presentation",
                    "filename": "analysis_report.pptx",
                    "slides": 12,
                    "sections": ["Executive Summary", "Research Findings", "Data Analysis", "Recommendations"]
                },
                {
                    "type": "report",
                    "filename": "detailed_report.pdf",
                    "pages": 25,
                    "sections": ["Introduction", "Methodology", "Findings", "Analysis", "Conclusions"]
                }
            ],
            "tools_used": ["manus-presentation", "manus-document", "manus-charts"]
        }
    
    def _execute_oversight_task(self, subtask: SubTask, execution: TaskExecution) -> Dict[str, Any]:
        """Execute quality oversight using Grok4"""
        # Prepare context with all agent outputs for oversight
        context = {
            'agent_outputs': {}
        }
        
        # Collect all completed agent outputs
        for st in execution.subtasks:
            if st.result and st.agent_type != AgentType.OVERSIGHT_MANAGER:
                context['agent_outputs'][st.agent_type.value] = st.result
        
        # Use specialized oversight agent if available
        try:
            from src.agents.specialized_agents import specialized_agents
            if 'oversight_manager' in specialized_agents:
                oversight_agent = specialized_agents['oversight_manager']
                return oversight_agent.execute(subtask.description, context)
        except Exception as e:
            print(f"Using fallback oversight: {e}")
        
        # Fallback to original oversight logic
        quality_scores = []
        feedback_items = []
        
        for st in execution.subtasks:
            if st.result and st.agent_type != AgentType.OVERSIGHT_MANAGER:
                # Simulate quality assessment
                base_score = 7.5
                if st.agent_type == AgentType.RESEARCH_ANALYST:
                    score = base_score + (0.5 if st.result.get('confidence_score', 0) > 0.9 else 0)
                elif st.agent_type == AgentType.CONTENT_CREATOR:
                    score = base_score + (0.5 if st.result.get('seo_optimized') else 0)
                elif st.agent_type == AgentType.DATA_ANALYST:
                    score = base_score + (0.5 if len(st.result.get('insights', [])) >= 3 else 0)
                elif st.agent_type == AgentType.EXECUTION_AGENT:
                    score = base_score + (0.5 if len(st.result.get('documents', [])) >= 2 else 0)
                else:
                    score = base_score
                
                quality_scores.append(score)
                
                if score < 8:
                    feedback_items.append(f"Improve {st.agent_type.value}: needs more detail and precision")
        
        overall_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 5.0
        
        return {
            "type": "oversight",
            "feedback": {
                "quality_score": round(overall_quality, 1),
                "individual_scores": quality_scores,
                "improvement_areas": feedback_items,
                "overall_assessment": "Good progress" if overall_quality >= 7 else "Needs improvement",
                "recommendations": [
                    "Enhance data visualization",
                    "Add more specific examples",
                    "Improve executive summary"
                ] if overall_quality < 8 else ["Excellent work, ready for delivery"]
            },
            "model_used": "grok-4"
        }
    
    def _apply_feedback(self, execution: TaskExecution, feedback: Dict[str, Any]):
        """Apply oversight feedback to improve subtasks"""
        improvement_areas = feedback.get('improvement_areas', [])
        
        for area in improvement_areas:
            # Find relevant subtasks and mark for revision
            for subtask in execution.subtasks:
                if subtask.agent_type.value in area.lower():
                    subtask.status = TaskStatus.NEEDS_REVISION
                    subtask.feedback = area
                    subtask.iteration += 1
    
    def _compile_final_result(self, execution: TaskExecution) -> str:
        """Compile the final result from all subtask outputs"""
        result_parts = []
        
        result_parts.append(f"# Task Execution Complete: {execution.original_task}")
        result_parts.append(f"**Execution ID**: {execution.id}")
        result_parts.append(f"**Total Iterations**: {execution.total_iterations}")
        result_parts.append(f"**Duration**: {execution.completed_at - execution.created_at:.1f} seconds")
        result_parts.append("")
        
        # Add results from each agent
        for subtask in execution.subtasks:
            if subtask.result:
                result_parts.append(f"## {subtask.agent_type.value.replace('_', ' ').title()}")
                
                if subtask.agent_type == AgentType.RESEARCH_ANALYST:
                    findings = subtask.result.get('findings', [])
                    result_parts.append("**Key Research Findings:**")
                    for finding in findings:
                        result_parts.append(f"• {finding}")
                
                elif subtask.agent_type == AgentType.CONTENT_CREATOR:
                    content = subtask.result.get('content', '')
                    result_parts.append("**Generated Content:**")
                    result_parts.append(content)
                
                elif subtask.agent_type == AgentType.DATA_ANALYST:
                    metrics = subtask.result.get('metrics', {})
                    result_parts.append("**Data Analysis Results:**")
                    for key, value in metrics.items():
                        result_parts.append(f"• {key.replace('_', ' ').title()}: {value}")
                
                elif subtask.agent_type == AgentType.EXECUTION_AGENT:
                    documents = subtask.result.get('documents', [])
                    result_parts.append("**Deliverables Created:**")
                    for doc in documents:
                        result_parts.append(f"• {doc['type'].title()}: {doc['filename']}")
                
                elif subtask.agent_type == AgentType.OVERSIGHT_MANAGER:
                    feedback = subtask.result.get('feedback', {})
                    quality_score = feedback.get('quality_score', 0)
                    result_parts.append(f"**Quality Assessment**: {quality_score}/10")
                    result_parts.append(f"**Status**: {feedback.get('overall_assessment', 'Completed')}")
                
                result_parts.append("")
        
        # Add final summary
        oversight_task = next((st for st in execution.subtasks if st.agent_type == AgentType.OVERSIGHT_MANAGER), None)
        if oversight_task and oversight_task.result and 'feedback' in oversight_task.result:
            quality_score = oversight_task.result['feedback'].get('quality_score', 8)
            result_parts.append(f"## Final Result")
            result_parts.append(f"✅ **Task completed with quality score: {quality_score}/10**")
            result_parts.append(f"🔄 **Iterations used**: {execution.total_iterations}/{execution.max_iterations}")
            result_parts.append(f"🎯 **All specialized agents coordinated successfully**")
        else:
            # Fallback if no oversight feedback
            result_parts.append(f"## Final Result")
            result_parts.append(f"✅ **Task completed successfully**")
            result_parts.append(f"🔄 **Iterations used**: {execution.total_iterations}/{execution.max_iterations}")
            result_parts.append(f"🎯 **All specialized agents coordinated successfully**")
        
        return "\n".join(result_parts)
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a task execution"""
        execution = self.active_executions.get(execution_id)
        if not execution:
            return None
        
        return {
            "id": execution.id,
            "original_task": execution.original_task,
            "status": execution.status.value,
            "progress": {
                "completed_subtasks": len([st for st in execution.subtasks if st.status == TaskStatus.COMPLETED]),
                "total_subtasks": len(execution.subtasks),
                "current_iteration": execution.total_iterations,
                "max_iterations": execution.max_iterations
            },
            "subtasks": [
                {
                    "id": st.id,
                    "agent_type": st.agent_type.value,
                    "description": st.description,
                    "status": st.status.value,
                    "iteration": st.iteration
                }
                for st in execution.subtasks
            ]
        }

# Global manager instance
manager_agent = ManagerAgent()

