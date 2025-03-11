"""
Agent package for various specialized agents.

This package contains agents responsible for specific tasks such as
cleaning up key facts and key snippets in the database when they
exceed certain thresholds, as well as performing research tasks,
planning implementation, and implementing specific tasks.

Includes agents for:
- Key facts garbage collection
- Key snippets garbage collection
- Implementation tasks
- Planning tasks
- Research tasks
"""

from typing import Optional

from ra_aid.agents.implementation_agent import run_task_implementation_agent
from ra_aid.agents.key_facts_gc_agent import run_key_facts_gc_agent
from ra_aid.agents.key_snippets_gc_agent import run_key_snippets_gc_agent
from ra_aid.agents.planning_agent import run_planning_agent
from ra_aid.agents.research_agent import run_research_agent, run_web_research_agent

__all__ = [
    "run_key_facts_gc_agent", 
    "run_key_snippets_gc_agent",
    "run_planning_agent",
    "run_research_agent",
    "run_task_implementation_agent",
    "run_web_research_agent"
]