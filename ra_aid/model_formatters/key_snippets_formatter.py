"""
Key snippets model formatter module.

This module provides utility functions for formatting key snippets from database models
into consistent markdown styling for display or output purposes.
"""

from typing import Dict, Optional


def format_key_snippet(snippet_id: int, filepath: str, line_number: int, snippet: str, description: Optional[str] = None) -> str:
    """
    Format a single key snippet with markdown formatting.

    Args:
        snippet_id: The identifier of the snippet
        filepath: Path to the source file
        line_number: Line number where the snippet starts
        snippet: The source code snippet text
        description: Optional description of the significance

    Returns:
        str: Formatted key snippet as markdown

    Example:
        >>> format_key_snippet(1, "src/main.py", 10, "def hello():\\n    return 'world'", "Main function")
        '## 📝 Code Snippet #1\\n\\n**Source Location**:\\n- File: `src/main.py`\\n- Line: `10`\\n\\n**Code**:\\n```python\\ndef hello():\\n    return \\'world\\'\\n```\\n\\n**Description**:\\nMain function'
    """
    if not snippet:
        return ""

    formatted_snippet = f"## 📝 Code Snippet #{snippet_id}\n\n"
    formatted_snippet += f"**Source Location**:\n"
    formatted_snippet += f"- File: `{filepath}`\n"
    formatted_snippet += f"- Line: `{line_number}`\n\n"
    formatted_snippet += f"**Code**:\n```python\n{snippet}\n```\n"

    if description:
        formatted_snippet += f"\n**Description**:\n{description}"

    return formatted_snippet


def format_key_snippets_dict(snippets_dict: Dict[int, Dict]) -> str:
    """
    Format a dictionary of key snippets with consistent markdown formatting.

    Args:
        snippets_dict: Dictionary mapping snippet IDs to snippet information dictionaries.
                       Each snippet dictionary should contain: filepath, line_number, snippet, 
                       and optionally description.

    Returns:
        str: Formatted key snippets as markdown with proper spacing and headings

    Example:
        >>> snippets = {
        ...     1: {
        ...         "filepath": "src/main.py", 
        ...         "line_number": 10, 
        ...         "snippet": "def hello():\\n    return 'world'", 
        ...         "description": "Main function"
        ...     }
        ... }
        >>> format_key_snippets_dict(snippets)
        '## 📝 Code Snippet #1\\n\\n**Source Location**:\\n- File: `src/main.py`\\n- Line: `10`\\n\\n**Code**:\\n```python\\ndef hello():\\n    return \\'world\\'\\n```\\n\\n**Description**:\\nMain function'
    """
    if not snippets_dict:
        return ""

    # Sort by ID for consistent output and format as markdown sections
    snippets = []
    for snippet_id, snippet_info in sorted(snippets_dict.items()):
        snippets.extend([
            format_key_snippet(
                snippet_id, 
                snippet_info.get("filepath", ""), 
                snippet_info.get("line_number", 0), 
                snippet_info.get("snippet", ""), 
                snippet_info.get("description", None)
            ),
            ""  # Empty line between snippets
        ])

    # Join all snippets and remove trailing newline
    return "\n".join(snippets).rstrip()