from crewai import Task, Crew, Process
from .agents import (
    manager_agent,
    research_agent,
    thinking_agent,
    creator_agent,
    controller_agent,
)
from .logging import logger  # NEW: Logging

def run_orchestrator(user_task: str) -> dict:
    logger.info(f"Starting orchestration for task: {user_task}")
    
    # Top-level planning
    plan_task = Task(
        description=f"Create a structured execution plan for '{user_task}'. "
                    "Allocate research, reasoning, creation and QC steps.",
        agent=manager_agent,
        expected_output="JSON list of delegated sub-tasks."
    )

    # Delegated work
    research_task = Task(
        description="Perform all research required by the plan.",
        agent=research_agent,
        expected_output="Research findings."
    )
    thinking_task = Task(
        description="Perform deep reasoning per the plan.",
        agent=thinking_agent,
        expected_output="Reasoned conclusions."
    )
    create_task = Task(
        description="Combine inputs and build the deliverable. "
                    "Return the artefact or its storage URI.",
        agent=creator_agent,
        expected_output="Deliverable artefact."
    )

    # Crew for delegation and creation (hierarchical)
    crew = Crew(
        agents=[manager_agent, research_agent, thinking_agent, creator_agent],
        tasks=[plan_task, research_task, thinking_task, create_task],
        process=Process.hierarchical,
        manager_agent=manager_agent,
        verbose=2,
        max_iter=10,  # NEW: Prevent infinite loops
    )
    deliverable = crew.kickoff()
    logger.info("Deliverable created.")

    # Separate verification (ensures it's post-creation)
    verify_task = Task(
        description=f"Examine the deliverable '{deliverable}' item-by-item for flaws, "
                    "accuracy, completeness, and logical consistency. List issues per item.",
        agent=controller_agent,
        expected_output="Itemized verification report with pass/fail per item."
    )
    verification = verify_task.execute(context={"deliverable": deliverable})  # Direct exec
    logger.info("Verification complete.")

    return {"deliverable": deliverable, "verification": verification}

