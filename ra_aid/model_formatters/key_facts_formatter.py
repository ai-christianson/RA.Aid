"""
Key facts model formatter module.

This module provides utility functions for formatting key facts from database models
into consistent markdown styling for display or output purposes.
"""

from typing import Dict, Optional


def format_key_fact(fact_id: int, content: str) -> str:
    """
    Format a single key fact with markdown formatting.

    Args:
        fact_id: The identifier of the fact
        content: The text content of the fact

    Returns:
        str: Formatted key fact as markdown

    Example:
        >>> format_key_fact(1, "This is an important fact")
        '## 🔑 Key Fact #1\n\nThis is an important fact'
    """
    if not content:
        return ""

    return f"## 🔑 Key Fact #{fact_id}\n\n{content}"


def format_key_facts_dict(facts_dict: Dict[int, str]) -> str:
    """
    Format a dictionary of key facts with consistent markdown formatting.

    Args:
        facts_dict: Dictionary mapping fact IDs to content strings

    Returns:
        str: Formatted key facts as markdown with proper spacing and headings

    Example:
        >>> format_key_facts_dict({1: "First fact", 2: "Second fact"})
        '## 🔑 Key Fact #1\n\nFirst fact\n\n## 🔑 Key Fact #2\n\nSecond fact'
    """
    if not facts_dict:
        return ""

    # Sort by ID for consistent output and format as markdown sections
    facts = []
    for fact_id, content in sorted(facts_dict.items()):
        facts.extend([
            format_key_fact(fact_id, content),
            ""  # Empty line between facts
        ])

    # Join all facts and remove trailing newline
    return "\n".join(facts).rstrip()