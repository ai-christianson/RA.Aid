"""
Expert-specific prompt sections for the AI agent system.

This module contains prompt section constants related to expert consultation
for different stages of task execution, such as research, planning, implementation,
and interactive chat.
"""

# Expert-specific prompt sections
EXPERT_PROMPT_SECTION_RESEARCH = """
Expert Consultation:
    If you need additional guidance, analysis, or verification (including code correctness checks and debugging):
    - Use emit_expert_context to provide all relevant context about what you've found
    - Wait for the expert response before proceeding with research
    - The expert can help analyze complex codebases, unclear patterns, or subtle edge cases

The expert is really good at logic, debugging and planning, but it only has access to the context you give it, and it is unable to access the outside world.
The expert does not have access to the latest information, so if you are looking for up-to-date information rather than a pure logical question, you may be better of using the web search tool, if available.
"""

EXPERT_PROMPT_SECTION_PLANNING = """
Expert Consultation:
    If you need additional input, assistance, or any logic verification:
    - First use emit_expert_context to provide all relevant context
    - Wait for the expert's response before defining tasks in non-trivial scenarios
    - The expert can help with architectural decisions, correctness checks, and detailed planning

The expert is really good at logic, debugging and planning, but it only has access to the context you give it, and it is unable to access the outside world.
The expert does not have access to the latest information, so if you are looking for up-to-date information rather than a pure logical question, you may be better of using the web search tool, if available.
"""

EXPERT_PROMPT_SECTION_IMPLEMENTATION = """
Expert Consultation:
    If you have any doubts about logic, debugging, or best approaches (or how to test something thoroughly):
    - Use emit_expert_context to provide context about your specific concern
    - Ask the expert to perform deep analysis or correctness checks
    - Wait for expert guidance before proceeding with implementation

The expert is really good at logic, debugging and planning, but it only has access to the context you give it, and it is unable to access the outside world.
The expert does not have access to the latest information, so if you are looking for up-to-date information rather than a pure logical question, you may be better of using the web search tool, if available.
"""

EXPERT_PROMPT_SECTION_CHAT = """
Expert Consultation:
    If you need expert input during the interactive chat phase, or if any aspect of the logic or debugging is uncertain:
    - Use emit_expert_context to provide the current conversation state, user requirements, and discovered details
    - Ask the expert for advice on handling ambiguous user requests or complex technical challenges, and to verify correctness
    - Wait for the expert's guidance before making decisions that significantly alter the approach or final outcome

The expert is really good at logic, debugging and planning, but it only has access to the context you give it, and it is unable to access the outside world.
The expert does not have access to the latest information, so if you are looking for up-to-date information rather than a pure logical question, you may be better of using the web search tool, if available.
"""