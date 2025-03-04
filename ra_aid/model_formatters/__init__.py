"""
This module provides formatting functions for model data for display or output.

It includes functions to format key facts, key snippets, and research notes in a consistent, 
readable way for presentation to users and other parts of the system.
"""

from ra_aid.model_formatters.key_facts_formatter import format_key_fact, format_key_facts_dict
from ra_aid.model_formatters.research_notes_formatter import format_research_note, format_research_notes_dict

__all__ = ["format_key_fact", "format_key_facts_dict", "format_research_note", "format_research_notes_dict"]