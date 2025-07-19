from crewai import Agent
from langchain_openai import ChatOpenAI
from langchain_xai import ChatXAI  # FIXED: Correct import (requires langchain-xai)
from .config import settings
from .tools.sandbox import sandbox_exec_tool
from .tools.manus import manus_tool

# --- LLM registry ----------------------------------------------------------- #
openai_llm = ChatOpenAI(
    model=settings.model_openai,
    api_key=settings.openai_api_key,
    temperature=0.2,
)
grok_llm = ChatXAI(  # Supports Grok-4
    model=settings.model_grok,
    api_key=settings.xai_api_key,
    temperature=0.2,
)

# --- Agents ----------------------------------------------------------------- #
manager_agent = Agent(
    role="Manager",
    goal="Break tasks down and delegate optimally",
    backstory="A strategic planner using Grok-4 for meta-reasoning.",
    llm=grok_llm,
    verbose=False,  # CHANGED: Less noisy in prod
)

research_agent = Agent(
    role="Researcher",
    goal="Gather authoritative information",
    backstory="Harnesses GPT-4o's retrieval strength.",
    llm=openai_llm,
    verbose=False,
)

thinking_agent = Agent(
    role="Thinker",
    goal="Provide structured reasoning",
    backstory="Exploits Grok-4's 256k context for deep chains of thought.",
    llm=grok_llm,
    verbose=False,
)

creator_agent = Agent(
    role="Creator",
    goal="Build concrete deliverables via safe code execution",
    backstory="Uses sandboxed Python or Manus when available.",
    tools=[sandbox_exec_tool, manus_tool],
    llm=openai_llm,
    verbose=False,
)

controller_agent = Agent(
    role="Controller",
    goal="Verify each deliverable item for correctness",
    backstory="Applies Grok-4 as an exacting QA auditor.",
    llm=grok_llm,
    verbose=False,
)

__all__ = [
    "manager_agent",
    "research_agent",
    "thinking_agent",
    "creator_agent",
    "controller_agent",
]

