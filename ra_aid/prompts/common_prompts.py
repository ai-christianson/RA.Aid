"""
Common prompt sections shared across different agent types.

This module contains prompt sections that are used by multiple agent types
in the AI agent system. These are shared components to ensure consistency
across different prompt templates.
"""

# New project hints
NEW_PROJECT_HINTS = """
Because this is a new project:
- If the user did not specify a stack, use your best judgment, or make a proposal and ask the human if the human-in-the-loop tool is available.
- If the user did not specify a directory to create the project in, create directly in the current directory.
"""