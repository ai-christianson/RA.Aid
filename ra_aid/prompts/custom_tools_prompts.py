"""
Contains custom tools specific prompt sections for use in RA-Aid.
"""

DEFAULT_CUSTOM_TOOLS_PROMPT = """
Custom Tools:
    - When custom tools are provided (e.g. compile_firmware, generate_docs), use them directly instead of researching or using other tools
    - Custom tools are pre-configured callables that handle their specific tasks
    - NEVER print messages about custom tool operations - always call the actual tool
    - Do not use request_research or other tools when a custom tool exists for the task
    - Custom tools take precedence over all other tools for their specific tasks
"""