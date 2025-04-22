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
        str: The content of the research note

    Example:
        >>> format_research_note(1, "This is an important research finding")
        'This is an important research finding'
    """
    # The panel title/heading should be handled by the rendering component (UI/CLI).
    return content


def format_research_notes_dict(notes_dict: Dict[int, str]) -> str:
    """
    Format a dictionary of research notes with consistent markdown formatting.

    Args:
        notes_dict: Dictionary mapping note IDs to content strings

    Returns:
        str: Formatted research notes as markdown with proper spacing and headings

    Example:
        >>> format_research_notes_dict({1: "First finding", 2: "Second finding"})
        'First finding\n\nSecond finding'
    """
    if not notes_dict:
        return ""

    # Sort by ID for consistent output and format as markdown sections
    notes = []
    for note_id, content in sorted(notes_dict.items()):
        formatted_note = format_research_note(note_id, content)
        if formatted_note: # Only add non-empty notes
            notes.extend([
                formatted_note,
                ""  # Empty line between notes
            ])

    # Join all notes and remove trailing newline/empty string if present
    return "\n".join(notes).rstrip()
