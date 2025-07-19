"""
Improved CrewAI-based manager agent that integrates with existing AgentFlow orchestration.
Provides enhanced multi-agent coordination with proper ChatXAI integration.
"""
import logging
from typing import Dict, Any, List
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_xai import ChatXAI  # FIXED: Correct import for Grok-4

from ..config.improved_settings import improved_settings
from ..tools.improved_sandbox import improved_sandbox_exec_tool, improved_manus_tool

logger = logging.getLogger(__name__)

class ImprovedCrewManager:
    """
    Enhanced manager agent using CrewAI for hierarchical task delegation.
    Integrates with existing AgentFlow while providing improved AI coordination.
    """
    
    def __init__(self):
        self.setup_llms()
        self.setup_agents()
        logger.info("ImprovedCrewManager initialized with ChatXAI and CrewAI")
    
    def setup_llms(self):
        """Initialize LLM clients with proper error handling."""
        try:
            # OpenAI for research and content creation
            self.openai_llm = ChatOpenAI(
                model=improved_settings.model_openai,
                api_key=improved_settings.openai_api_key,
                temperature=0.2,
            )
            logger.info(f"OpenAI LLM initialized: {improved_settings.model_openai}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI LLM: {e}")
            self.openai_llm = None
        
        try:
            # Grok for strategic thinking and management
            self.grok_llm = ChatXAI(
                model=improved_settings.model_grok,
                api_key=improved_settings.xai_api_key,
                temperature=0.2,
            )
            logger.info(f"Grok LLM initialized: {improved_settings.model_grok}")
        except Exception as e:
            logger.error(f"Failed to initialize Grok LLM: {e}")
            self.grok_llm = None
    
    def setup_agents(self):
        """Create specialized agents with proper tool integration."""
        # Manager Agent (uses Grok-4 for strategic planning)
        self.manager_agent = Agent(
            role="Strategic Manager",
            goal="Break down complex tasks and delegate optimally to specialized agents",
            backstory="An experienced project manager using Grok-4's advanced reasoning for strategic planning and task coordination.",
            llm=self.grok_llm if self.grok_llm else self.openai_llm,
            verbose=improved_settings.crew_verbose,
        )
        
        # Research Agent (uses GPT-4o for information gathering)
        self.research_agent = Agent(
            role="Research Specialist",
            goal="Gather comprehensive and authoritative information",
            backstory="A thorough researcher leveraging GPT-4o's knowledge base and reasoning capabilities.",
            llm=self.openai_llm if self.openai_llm else self.grok_llm,
            verbose=improved_settings.crew_verbose,
        )
        
        # Thinking Agent (uses Grok-4 for deep reasoning)
        self.thinking_agent = Agent(
            role="Strategic Thinker",
            goal="Provide structured reasoning and analysis",
            backstory="A strategic analyst exploiting Grok-4's 256k context window for deep chains of thought.",
            llm=self.grok_llm if self.grok_llm else self.openai_llm,
            verbose=improved_settings.crew_verbose,
        )
        
        # Creator Agent (uses tools for deliverable creation)
        self.creator_agent = Agent(
            role="Deliverable Creator",
            goal="Build concrete deliverables using secure code execution",
            backstory="A technical specialist using sandboxed environments and AI tools for safe deliverable creation.",
            tools=[improved_sandbox_exec_tool, improved_manus_tool],
            llm=self.openai_llm if self.openai_llm else self.grok_llm,
            verbose=improved_settings.crew_verbose,
        )
        
        # Controller Agent (uses Grok-4 for quality assurance)
        self.controller_agent = Agent(
            role="Quality Controller",
            goal="Verify deliverables for correctness and completeness",
            backstory="A meticulous quality assurance specialist using Grok-4 for thorough verification.",
            llm=self.grok_llm if self.grok_llm else self.openai_llm,
            verbose=improved_settings.crew_verbose,
        )
        
        logger.info("All specialized agents initialized successfully")
    
    def execute_task(self, user_task: str) -> Dict[str, Any]:
        """
        Execute a complex task using hierarchical multi-agent coordination.
        
        Args:
            user_task: The task description from the user
            
        Returns:
            Dictionary containing deliverable and verification results
        """
        logger.info(f"Starting improved crew execution for task: {user_task}")
        
        try:
            # Create task breakdown
            plan_task = Task(
                description=f"Create a structured execution plan for: '{user_task}'. "
                           "Allocate research, reasoning, creation and quality control steps. "
                           "Provide clear sub-tasks for each specialized agent.",
                agent=self.manager_agent,
                expected_output="Detailed JSON plan with delegated sub-tasks and execution order."
            )
            
            # Research phase
            research_task = Task(
                description="Perform comprehensive research required by the execution plan. "
                           "Gather all necessary information, data, and context.",
                agent=self.research_agent,
                expected_output="Comprehensive research findings and gathered information."
            )
            
            # Analysis phase
            thinking_task = Task(
                description="Perform deep reasoning and analysis based on research findings. "
                           "Provide strategic insights and structured conclusions.",
                agent=self.thinking_agent,
                expected_output="Strategic analysis and reasoned conclusions."
            )
            
            # Creation phase
            create_task = Task(
                description="Combine research and analysis to build the requested deliverable. "
                           "Use available tools for secure creation. Return the artifact or its URI.",
                agent=self.creator_agent,
                expected_output="Completed deliverable artifact or secure URI reference."
            )
            
            # Create and execute crew
            crew = Crew(
                agents=[self.manager_agent, self.research_agent, self.thinking_agent, self.creator_agent],
                tasks=[plan_task, research_task, thinking_task, create_task],
                process=Process.hierarchical,
                manager_agent=self.manager_agent,
                verbose=2 if improved_settings.crew_verbose else 0,
                max_iter=improved_settings.crew_max_iterations,
            )
            
            # Execute main workflow
            deliverable = crew.kickoff()
            logger.info("Crew execution completed, starting verification")
            
            # Separate verification step
            verify_task = Task(
                description=f"Examine the deliverable '{deliverable}' item-by-item for: "
                           "1. Accuracy and correctness "
                           "2. Completeness against requirements "
                           "3. Quality and consistency "
                           "4. Technical implementation "
                           "Provide detailed pass/fail assessment for each aspect.",
                agent=self.controller_agent,
                expected_output="Comprehensive verification report with itemized pass/fail results."
            )
            
            verification = verify_task.execute(context={"deliverable": deliverable})
            logger.info("Verification completed successfully")
            
            return {
                "success": True,
                "deliverable": deliverable,
                "verification": verification,
                "execution_method": "ImprovedCrewManager",
                "agents_used": ["manager", "research", "thinking", "creator", "controller"]
            }
            
        except Exception as e:
            error_msg = f"Improved crew execution failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "execution_method": "ImprovedCrewManager"
            }

# Global instance for integration with existing routes
improved_crew_manager = ImprovedCrewManager()

