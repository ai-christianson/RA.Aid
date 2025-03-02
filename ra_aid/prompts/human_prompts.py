"""
Human interaction prompt sections for the AI agent system.

This module contains human-specific prompt sections that provide guidance
for how agents should interact with human operators in different stages
of task execution.
"""

# Human-specific prompt sections
HUMAN_PROMPT_SECTION_RESEARCH = """
Human Interaction:
    If you need clarification from the human operator:
    - Ask clear, specific questions
    - Use the ask_human tool for queries
    - Wait for human response before proceeding
"""

HUMAN_PROMPT_SECTION_PLANNING = """
Human Interaction:
    If you need requirements clarification:
    - Use ask_human for specific planning questions
    - Await human input before finalizing plans
    - Keep questions focused and context-aware
"""

HUMAN_PROMPT_SECTION_IMPLEMENTATION = """
Human Interaction:
    If you need implementation guidance:
    - Ask the human operator using ask_human
    - Keep questions specific to the current task
    - Wait for responses before proceeding
"""