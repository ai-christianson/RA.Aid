"""
RA-Aid Prompts Package

This package organizes the various prompt templates used throughout the RA-Aid application.
Prompts are categorized by their functional purpose (research, planning, implementation, etc.)
and organized into separate modules for better maintainability.

The prompts in this package serve as templates for constructing context-specific instructions 
for various types of AI agents used within the system.
"""

# Re-export all prompts for backward compatibility

# Common prompts
from ra_aid.prompts.common_prompts import NEW_PROJECT_HINTS

# Expert prompts
from ra_aid.prompts.expert_prompts import (
    EXPERT_PROMPT_SECTION_RESEARCH,
    EXPERT_PROMPT_SECTION_PLANNING,
    EXPERT_PROMPT_SECTION_IMPLEMENTATION,
    EXPERT_PROMPT_SECTION_CHAT,
)

# Human prompts
from ra_aid.prompts.human_prompts import (
    HUMAN_PROMPT_SECTION_RESEARCH,
    HUMAN_PROMPT_SECTION_PLANNING,
    HUMAN_PROMPT_SECTION_IMPLEMENTATION,
)

# Web research prompts
from ra_aid.prompts.web_research_prompts import (
    WEB_RESEARCH_PROMPT_SECTION_RESEARCH,
    WEB_RESEARCH_PROMPT_SECTION_PLANNING,
    WEB_RESEARCH_PROMPT_SECTION_IMPLEMENTATION,
    WEB_RESEARCH_PROMPT_SECTION_CHAT,
    WEB_RESEARCH_PROMPT,
)

# Research prompts
from ra_aid.prompts.research_prompts import (
    RESEARCH_COMMON_PROMPT_HEADER,
    RESEARCH_PROMPT,
    RESEARCH_ONLY_PROMPT,
)

# Planning prompts
from ra_aid.prompts.planning_prompts import PLANNING_PROMPT

# Reasoning assist prompts
from ra_aid.prompts.reasoning_assist_prompt import REASONING_ASSIST_PROMPT_PLANNING, REASONING_ASSIST_PROMPT_IMPLEMENTATION, REASONING_ASSIST_PROMPT_RESEARCH

# Implementation prompts
from ra_aid.prompts.implementation_prompts import IMPLEMENTATION_PROMPT

# Chat prompts
from ra_aid.prompts.chat_prompts import CHAT_PROMPT

# CIAYN prompts
from ra_aid.prompts.ciayn_prompts import (
    CIAYN_AGENT_SYSTEM_PROMPT,
    CIAYN_AGENT_HUMAN_PROMPT,
    EXTRACT_TOOL_CALL_PROMPT,
    NO_TOOL_CALL_PROMPT,
)

# Add an __all__ list with all the exported names
__all__ = [
    # Common prompts
    "NEW_PROJECT_HINTS",
    
    # Expert prompts
    "EXPERT_PROMPT_SECTION_RESEARCH",
    "EXPERT_PROMPT_SECTION_PLANNING",
    "EXPERT_PROMPT_SECTION_IMPLEMENTATION",
    "EXPERT_PROMPT_SECTION_CHAT",
    
    # Human prompts
    "HUMAN_PROMPT_SECTION_RESEARCH",
    "HUMAN_PROMPT_SECTION_PLANNING",
    "HUMAN_PROMPT_SECTION_IMPLEMENTATION",
    
    # Web research prompts
    "WEB_RESEARCH_PROMPT_SECTION_RESEARCH",
    "WEB_RESEARCH_PROMPT_SECTION_PLANNING",
    "WEB_RESEARCH_PROMPT_SECTION_IMPLEMENTATION",
    "WEB_RESEARCH_PROMPT_SECTION_CHAT",
    "WEB_RESEARCH_PROMPT",
    
    # Research prompts
    "RESEARCH_COMMON_PROMPT_HEADER",
    "RESEARCH_PROMPT",
    "RESEARCH_ONLY_PROMPT",
    
    # Planning prompts
    "PLANNING_PROMPT",
    
    # Reasoning assist prompts
    "REASONING_ASSIST_PROMPT_PLANNING",
    "REASONING_ASSIST_PROMPT_IMPLEMENTATION",
    "REASONING_ASSIST_PROMPT_RESEARCH",
    
    # Implementation prompts
    "IMPLEMENTATION_PROMPT",
    
    # Chat prompts
    "CHAT_PROMPT",
    
    # CIAYN prompts
    "CIAYN_AGENT_SYSTEM_PROMPT",
    "CIAYN_AGENT_HUMAN_PROMPT",
    "EXTRACT_TOOL_CALL_PROMPT",
    "NO_TOOL_CALL_PROMPT",
]