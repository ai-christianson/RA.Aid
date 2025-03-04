"""
Research notes model formatter module.

This module provides utility functions for formatting research notes from database models
into consistent markdown styling for display or output purposes.
"""

from typing import Dict, Optional


def format_research_note(note_id: int, content: str) -> str:
    """
    Format a single research note with markdown formatting.

    Args:
        note_id: The identifier of the research note
        content: The text content of the research note

    Returns:
        str: Formatted research note as markdown

    Example:
        >>> format_research_note(1, "This is an important research finding")
        '## 🔍 Research Note #1\n\nThis is an important research finding'
    """
    if not content:
        return ""

    return f"## 🔍 Research Note #{note_id}\n\n{content}"


def format_research_notes_dict(notes_dict: Dict[int, str]) -> str:
    """
    Format a dictionary of research notes with consistent markdown formatting.

    Args:
        notes_dict: Dictionary mapping note IDs to content strings

    Returns:
        str: Formatted research notes as markdown with proper spacing and headings

    Example:
        >>> format_research_notes_dict({1: "First finding", 2: "Second finding"})
        '## 🔍 Research Note #1\n\nFirst finding\n\n## 🔍 Research Note #2\n\nSecond finding'
    """
    if not notes_dict:
        return ""

    # Sort by ID for consistent output and format as markdown sections
    notes = []
    for note_id, content in sorted(notes_dict.items()):
        notes.extend([
            format_research_note(note_id, content),
            ""  # Empty line between notes
        ])

    # Join all notes and remove trailing newline
    return "\n".join(notes).rstrip()