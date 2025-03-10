"""
Agent package for various specialized agents.

This package contains agents responsible for specific tasks such as
cleaning up key facts and key snippets in the database when they
exceed certain thresholds, as well as performing research tasks
and planning implementation.

Includes agents for:
- Key facts garbage collection
- Key snippets garbage collection
- Research tasks
- Planning tasks
"""

from typing import Optional

from ra_aid.agents.key_facts_gc_agent import run_key_facts_gc_agent
from ra_aid.agents.key_snippets_gc_agent import run_key_snippets_gc_agent
from ra_aid.agents.research_agent import run_research_agent, run_web_research_agent
from ra_aid.agents.planning_agent import run_planning_agent

__all__ = [
    "run_key_facts_gc_agent", 
    "run_key_snippets_gc_agent",
    "run_research_agent",
    "run_web_research_agent",
    "run_planning_agent"
]