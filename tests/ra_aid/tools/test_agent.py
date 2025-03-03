"""Tests for the agent.py module to verify KeyFactRepository integration."""

import pytest
from unittest.mock import patch, MagicMock

from ra_aid.tools.agent import (
    request_research,
    request_task_implementation,
    request_implementation,
    request_research_and_implementation,
)
from ra_aid.tools.memory import _global_memory


@pytest.fixture
def reset_memory():
    """Reset global memory before each test"""
    _global_memory["key_facts"] = {}
    _global_memory["key_fact_id_counter"] = 0
    _global_memory["research_notes"] = []
    _global_memory["plans"] = []
    _global_memory["tasks"] = {}
    _global_memory["task_id_counter"] = 0
    _global_memory["related_files"] = {}
    _global_memory["related_file_id_counter"] = 0
    _global_memory["work_log"] = []
    yield
    # Clean up after test
    _global_memory["key_facts"] = {}
    _global_memory["key_fact_id_counter"] = 0
    _global_memory["research_notes"] = []
    _global_memory["plans"] = []
    _global_memory["tasks"] = {}
    _global_memory["task_id_counter"] = 0
    _global_memory["related_files"] = {}
    _global_memory["related_file_id_counter"] = 0
    _global_memory["work_log"] = []


@pytest.fixture
def mock_functions():
    """Mock functions used in agent.py"""
    with patch('ra_aid.tools.agent.key_fact_repository') as mock_fact_repo, \
         patch('ra_aid.tools.agent.format_key_facts_dict') as mock_fact_formatter, \
         patch('ra_aid.tools.agent.key_snippet_repository') as mock_snippet_repo, \
         patch('ra_aid.tools.agent.format_key_snippets_dict') as mock_snippet_formatter, \
         patch('ra_aid.tools.agent.initialize_llm') as mock_llm, \
         patch('ra_aid.tools.agent.get_related_files') as mock_get_files, \
         patch('ra_aid.tools.agent.get_memory_value') as mock_get_memory, \
         patch('ra_aid.tools.agent.get_work_log') as mock_get_work_log, \
         patch('ra_aid.tools.agent.reset_completion_flags') as mock_reset, \
         patch('ra_aid.tools.agent.get_completion_message') as mock_get_completion:

        # Setup mock return values
        mock_fact_repo.get_facts_dict.return_value = {1: "Test fact 1", 2: "Test fact 2"}
        mock_fact_formatter.return_value = "Formatted facts"
        mock_snippet_repo.get_snippets_dict.return_value = {1: {"filepath": "test.py", "line_number": 10, "snippet": "def test():", "description": "Test function"}}
        mock_snippet_formatter.return_value = "Formatted snippets"
        mock_llm.return_value = MagicMock()
        mock_get_files.return_value = ["file1.py", "file2.py"]
        mock_get_memory.return_value = "Test memory value"
        mock_get_work_log.return_value = "Test work log"
        mock_get_completion.return_value = "Task completed"
        
        # Return all mocks as a dictionary
        yield {
            'key_fact_repository': mock_fact_repo,
            'key_snippet_repository': mock_snippet_repo,
            'format_key_facts_dict': mock_fact_formatter,
            'format_key_snippets_dict': mock_snippet_formatter,
            'initialize_llm': mock_llm,
            'get_related_files': mock_get_files,
            'get_memory_value': mock_get_memory,
            'get_work_log': mock_get_work_log,
            'reset_completion_flags': mock_reset,
            'get_completion_message': mock_get_completion
        }


def test_request_research_uses_key_fact_repository(reset_memory, mock_functions):
    """Test that request_research uses KeyFactRepository directly with formatting."""
    # Mock running the research agent
    with patch('ra_aid.agent_utils.run_research_agent'):
        # Call the function
        result = request_research("test query")
        
        # Verify repository was called
        mock_functions['key_fact_repository'].get_facts_dict.assert_called_once()
        
        # Verify formatter was called with repository results
        mock_functions['format_key_facts_dict'].assert_called_once_with(
            mock_functions['key_fact_repository'].get_facts_dict.return_value
        )
        
        # Verify formatted facts are used in response
        assert result["key_facts"] == "Formatted facts"
        
        # Verify get_memory_value is not called with "key_facts"
        for call in mock_functions['get_memory_value'].call_args_list:
            assert call[0][0] != "key_facts"


def test_request_research_max_depth(reset_memory, mock_functions):
    """Test that max recursion depth handling uses KeyFactRepository."""
    # Set recursion depth to max
    _global_memory["agent_depth"] = 3
    
    # Call the function (should hit max depth case)
    result = request_research("test query")
    
    # Verify repository was called
    mock_functions['key_fact_repository'].get_facts_dict.assert_called_once()
    
    # Verify formatter was called with repository results
    mock_functions['format_key_facts_dict'].assert_called_once_with(
        mock_functions['key_fact_repository'].get_facts_dict.return_value
    )
    
    # Verify formatted facts are used in response
    assert result["key_facts"] == "Formatted facts"
    
    # Verify get_memory_value is not called with "key_facts"
    for call in mock_functions['get_memory_value'].call_args_list:
            assert call[0][0] != "key_facts"


def test_request_research_and_implementation_uses_key_fact_repository(reset_memory, mock_functions):
    """Test that request_research_and_implementation uses KeyFactRepository correctly."""
    # Mock running the research agent
    with patch('ra_aid.agent_utils.run_research_agent'):
        # Call the function
        result = request_research_and_implementation("test query")
        
        # Verify repository was called
        mock_functions['key_fact_repository'].get_facts_dict.assert_called_once()
        
        # Verify formatter was called with repository results
        mock_functions['format_key_facts_dict'].assert_called_once_with(
            mock_functions['key_fact_repository'].get_facts_dict.return_value
        )
        
        # Verify formatted facts are used in response
        assert result["key_facts"] == "Formatted facts"
        
        # Verify get_memory_value is not called with "key_facts"
        for call in mock_functions['get_memory_value'].call_args_list:
            assert call[0][0] != "key_facts"


def test_request_implementation_uses_key_fact_repository(reset_memory, mock_functions):
    """Test that request_implementation uses KeyFactRepository correctly."""
    # Mock running the planning agent
    with patch('ra_aid.agent_utils.run_planning_agent'):
        # Call the function
        result = request_implementation("test task")
        
        # Verify repository was called
        mock_functions['key_fact_repository'].get_facts_dict.assert_called_once()
        
        # Verify formatter was called with repository results
        mock_functions['format_key_facts_dict'].assert_called_once_with(
            mock_functions['key_fact_repository'].get_facts_dict.return_value
        )
        
        # Check that the formatted key facts are included in the response
        assert "Formatted facts" in result


def test_request_task_implementation_uses_key_fact_repository(reset_memory, mock_functions):
    """Test that request_task_implementation uses KeyFactRepository correctly."""
    # Set up _global_memory with required values
    _global_memory["tasks"] = {0: "Task 1"}
    _global_memory["base_task"] = "Base task"
    
    # Mock running the implementation agent
    with patch('ra_aid.agent_utils.run_task_implementation_agent'):
        # Call the function
        result = request_task_implementation("test task")
        
        # Verify repository was called
        mock_functions['key_fact_repository'].get_facts_dict.assert_called_once()
        
        # Verify formatter was called with repository results
        mock_functions['format_key_facts_dict'].assert_called_once_with(
            mock_functions['key_fact_repository'].get_facts_dict.return_value
        )
        
        # Check that the formatted key facts are included in the response
        assert "Formatted facts" in result