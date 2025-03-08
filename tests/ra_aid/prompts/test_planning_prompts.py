"""Tests for planning prompts."""

import pytest
from ra_aid.agent_utils import get_config_repository
from ra_aid.prompts.planning_prompts import PLANNING_PROMPT


def test_planning_prompt_expert_guidance_section():
    """Test that the planning prompt includes the expert_guidance_section placeholder."""
    assert "{expert_guidance_section}" in PLANNING_PROMPT


def test_planning_prompt_formatting_with_expert_guidance():
    """Test formatting the planning prompt with expert guidance."""
    # Sample expert guidance
    expert_guidance_section = "<expert guidance>\nThis is test expert guidance\n</expert guidance>"
    
    # Format the prompt
    formatted_prompt = PLANNING_PROMPT.format(
        current_date="2025-03-08",
        working_directory="/test/path",
        expert_section="",
        human_section="",
        web_research_section="",
        base_task="Test task",
        project_info="Test project info",
        research_notes="Test research notes",
        related_files="Test related files",
        key_facts="Test key facts",
        key_snippets="Test key snippets",
        work_log="Test work log",
        env_inv="Test env inventory",
        expert_guidance_section=expert_guidance_section,
    )
    
    # Check that the expert guidance section is included
    assert expert_guidance_section in formatted_prompt


def test_planning_prompt_formatting_without_expert_guidance():
    """Test formatting the planning prompt without expert guidance."""
    # Format the prompt with empty expert guidance
    formatted_prompt = PLANNING_PROMPT.format(
        current_date="2025-03-08",
        working_directory="/test/path",
        expert_section="",
        human_section="",
        web_research_section="",
        base_task="Test task",
        project_info="Test project info",
        research_notes="Test research notes",
        related_files="Test related files",
        key_facts="Test key facts",
        key_snippets="Test key snippets",
        work_log="Test work log",
        env_inv="Test env inventory",
        expert_guidance_section="",
    )
    
    # Check that the expert guidance section placeholder is replaced with empty string
    assert "<expert guidance>" not in formatted_prompt