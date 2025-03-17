"""Tests for the agent.py module to verify KeyFactRepository integration."""

import pytest
from unittest.mock import patch, MagicMock
import os

from ra_aid.tools.agent import (
    request_research,
    request_task_implementation,
    request_implementation,
    request_research_and_implementation,
)
from ra_aid.database.repositories.related_files_repository import get_related_files_repository
from ra_aid.database.repositories.work_log_repository import get_work_log_repository, WorkLogEntry
from ra_aid.database.repositories.config_repository import get_config_repository


@pytest.fixture
def reset_memory():
    """Fixture for test initialization (kept for backward compatibility)"""
    yield


@pytest.fixture(autouse=True)
def mock_related_files_repository():
    """Mock the RelatedFilesRepository to avoid database operations during tests"""
    with patch('ra_aid.database.repositories.related_files_repository.related_files_repo_var') as mock_repo_var:
        # Setup a mock repository
        mock_repo = MagicMock()
        
        # Create a dictionary to simulate stored files
        related_files = {}
        
        # Setup get_all method to return the files dict
        mock_repo.get_all.return_value = related_files
        
        # Setup format_related_files method
        mock_repo.format_related_files.return_value = [f"ID#{file_id} {filepath}" for file_id, filepath in sorted(related_files.items())]
        
        # Make the mock context var return our mock repo
        mock_repo_var.get.return_value = mock_repo
        
        yield mock_repo

@pytest.fixture(autouse=True)
def mock_config_repository():
    """Mock the ConfigRepository to avoid database operations during tests"""
    with patch('ra_aid.database.repositories.config_repository.config_repo_var') as mock_repo_var:
        # Setup a mock repository
        mock_repo = MagicMock()
        
        # Create a dictionary to simulate config
        config = {
            "recursion_limit": 2,
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-20241022",
            "temperature": 0.01
        }
        
        # Setup get method to return config values
        def get_config(key, default=None):
            return config.get(key, default)
        mock_repo.get.side_effect = get_config
        
        # Note: get_all is deprecated, but kept for backward compatibility
        # Setup get_all method to return a reference to the config dict
        mock_repo.get_all.return_value = config
        
        # Setup get method to return config values
        def get_config(key, default=None):
            return config.get(key, default)
        mock_repo.get.side_effect = get_config
        
        # Make the mock context var return our mock repo
        mock_repo_var.get.return_value = mock_repo
        
        yield mock_repo

@pytest.fixture(autouse=True)
def mock_work_log_repository():
    """Mock the WorkLogRepository to avoid database operations during tests"""
    with patch('ra_aid.tools.memory.get_work_log_repository') as mock_repo:
        # Setup the mock repository to behave like the original, but using memory
        entries = []  # Local in-memory storage
        
        # Mock add_entry method
        def mock_add_entry(event):
            from datetime import datetime
            entry = WorkLogEntry(timestamp=datetime.now().isoformat(), event=event)
            entries.append(entry)
        mock_repo.return_value.add_entry.side_effect = mock_add_entry
        
        # Mock get_all method
        def mock_get_all():
            return entries.copy()
        mock_repo.return_value.get_all.side_effect = mock_get_all
        
        # Mock clear method
        def mock_clear():
            entries.clear()
        mock_repo.return_value.clear.side_effect = mock_clear
        
        # Mock format_work_log method
        def mock_format_work_log():
            if not entries:
                return "No work log entries"
                
            formatted_entries = []
            for entry in entries:
                formatted_entries.extend([
                    f"## {entry['timestamp']}",
                    "",
                    entry["event"],
                    "",  # Blank line between entries
                ])
                
            return "\n".join(formatted_entries).rstrip()  # Remove trailing newline
        mock_repo.return_value.format_work_log.side_effect = mock_format_work_log
        
        yield mock_repo

@pytest.fixture(autouse=True)
def mock_trajectory_repository():
    """Mock the TrajectoryRepository to avoid database operations during tests"""
    with patch('ra_aid.database.repositories.trajectory_repository.trajectory_repo_var') as mock_repo_var:
        # Setup a mock repository
        mock_repo = MagicMock()
        
        # Setup create method to return a mock trajectory
        def mock_create(**kwargs):
            mock_trajectory = MagicMock()
            mock_trajectory.id = 1
            return mock_trajectory
        mock_repo.create.side_effect = mock_create
        
        # Make the mock context var return our mock repo
        mock_repo_var.get.return_value = mock_repo
        
        yield mock_repo

@pytest.fixture(autouse=True)
def mock_human_input_repository():
    """Mock the HumanInputRepository to avoid database operations during tests"""
    with patch('ra_aid.database.repositories.human_input_repository.human_input_repo_var') as mock_repo_var:
        # Setup a mock repository
        mock_repo = MagicMock()
        
        # Setup get_most_recent_id method to return a dummy ID
        mock_repo.get_most_recent_id.return_value = 1
        
        # Make the mock context var return our mock repo
        mock_repo_var.get.return_value = mock_repo
        
        yield mock_repo

@pytest.fixture
def mock_functions():
    """Mock functions used in agent.py"""
    mock_fact_repo = MagicMock()
    mock_snippet_repo = MagicMock()
    with patch('ra_aid.tools.agent.get_key_fact_repository', return_value=mock_fact_repo) as mock_get_fact_repo, \
         patch('ra_aid.tools.agent.format_key_facts_dict') as mock_fact_formatter, \
         patch('ra_aid.tools.agent.get_key_snippet_repository', return_value=mock_snippet_repo) as mock_get_snippet_repo, \
         patch('ra_aid.tools.agent.format_key_snippets_dict') as mock_snippet_formatter, \
         patch('ra_aid.tools.agent.initialize_llm') as mock_llm, \
         patch('ra_aid.tools.agent.get_related_files') as mock_get_files, \
         patch('ra_aid.tools.agent.get_work_log') as mock_get_work_log, \
         patch('ra_aid.tools.agent.reset_completion_flags') as mock_reset, \
         patch('ra_aid.tools.agent.get_completion_message') as mock_get_completion, \
         patch('ra_aid.tools.agent.get_trajectory_repository') as mock_get_trajectory_repo, \
         patch('ra_aid.tools.agent.get_human_input_repository') as mock_get_human_input_repo:

        # Setup mock return values
        mock_fact_repo.get_facts_dict.return_value = {1: "Test fact 1", 2: "Test fact 2"}
        mock_fact_formatter.return_value = "Formatted facts"
        mock_snippet_repo.get_snippets_dict.return_value = {1: {"filepath": "test.py", "line_number": 10, "snippet": "def test():", "description": "Test function"}}
        mock_snippet_formatter.return_value = "Formatted snippets"
        mock_llm.return_value = MagicMock()
        mock_get_files.return_value = ["file1.py", "file2.py"]
        mock_get_work_log.return_value = "Test work log"
        mock_get_completion.return_value = "Task completed"
        
        # Setup mock for trajectory repository
        mock_trajectory_repo = MagicMock()
        mock_get_trajectory_repo.return_value = mock_trajectory_repo
        
        # Setup mock for human input repository
        mock_human_input_repo = MagicMock()
        mock_human_input_repo.get_most_recent_id.return_value = 1
        mock_get_human_input_repo.return_value = mock_human_input_repo
        
        # Return all mocks as a dictionary
        yield {
            'get_key_fact_repository': mock_get_fact_repo,
            'get_key_snippet_repository': mock_get_snippet_repo,
            'format_key_facts_dict': mock_fact_formatter,
            'format_key_snippets_dict': mock_snippet_formatter,
            'initialize_llm': mock_llm,
            'get_related_files': mock_get_files,
            'get_work_log': mock_get_work_log,
            'reset_completion_flags': mock_reset,
            'get_completion_message': mock_get_completion,
            'get_trajectory_repository': mock_get_trajectory_repo,
            'get_human_input_repository': mock_get_human_input_repo
        }


def test_request_research_uses_key_fact_repository(reset_memory, mock_functions):
    """Test that request_research uses KeyFactRepository directly with formatting."""
    # Mock running the research agent
    with patch('ra_aid.agents.research_agent.run_research_agent'):
        # Call the function
        result = request_research("test query")
        
        # Verify repository was called
        mock_functions['get_key_fact_repository'].assert_called_once()
        mock_functions['get_key_fact_repository'].return_value.get_facts_dict.assert_called_once()
        
        # Verify formatter was called with repository results
        mock_functions['format_key_facts_dict'].assert_called_once_with(
            mock_functions['get_key_fact_repository'].return_value.get_facts_dict.return_value
        )
        
        # Verify formatted facts are used in response
        assert result["key_facts"] == "Formatted facts"


def test_request_research_max_depth(reset_memory, mock_functions):
    """Test that max recursion depth handling uses KeyFactRepository."""
    # Mock depth using context-based approach
    with patch('ra_aid.tools.agent.get_depth') as mock_get_depth:
        mock_get_depth.return_value = 3
    
    # Call the function (should hit max depth case)
    result = request_research("test query")
    
    # Verify repository was called
    mock_functions['get_key_fact_repository'].assert_called_once()
    mock_functions['get_key_fact_repository'].return_value.get_facts_dict.assert_called_once()
    
    # Verify formatter was called with repository results
    mock_functions['format_key_facts_dict'].assert_called_once_with(
        mock_functions['get_key_fact_repository'].return_value.get_facts_dict.return_value
    )
    
    # Verify formatted facts are used in response
    assert result["key_facts"] == "Formatted facts"


def test_request_research_and_implementation_uses_key_fact_repository(reset_memory, mock_functions):
    """Test that request_research_and_implementation uses KeyFactRepository correctly."""
    # Mock running the research agent
    with patch('ra_aid.agents.research_agent.run_research_agent'):
        # Call the function
        result = request_research_and_implementation("test query")
        
        # Verify repository was called
        mock_functions['get_key_fact_repository'].assert_called_once()
        mock_functions['get_key_fact_repository'].return_value.get_facts_dict.assert_called_once()
        
        # Verify formatter was called with repository results
        mock_functions['format_key_facts_dict'].assert_called_once_with(
            mock_functions['get_key_fact_repository'].return_value.get_facts_dict.return_value
        )
        
        # Verify formatted facts are used in response
        assert result["key_facts"] == "Formatted facts"


def test_request_implementation_uses_key_fact_repository(reset_memory, mock_functions):
    """Test that request_implementation uses KeyFactRepository correctly."""
    # Mock running the planning agent
    with patch('ra_aid.agents.planning_agent.run_planning_agent'):
        # Call the function
        result = request_implementation("test task")
        
        # Verify repository was called
        mock_functions['get_key_fact_repository'].assert_called_once()
        mock_functions['get_key_fact_repository'].return_value.get_facts_dict.assert_called_once()
        
        # Verify formatter was called with repository results
        mock_functions['format_key_facts_dict'].assert_called_once_with(
            mock_functions['get_key_fact_repository'].return_value.get_facts_dict.return_value
        )
        
        # Check that the formatted key facts are included in the response
        assert "Formatted facts" in result


def test_request_task_implementation_uses_key_fact_repository(reset_memory, mock_functions):
    """Test that request_task_implementation uses KeyFactRepository correctly."""
    # Mock running the implementation agent
    with patch('ra_aid.agents.implementation_agent.run_task_implementation_agent'):
        # Call the function
        result = request_task_implementation("test task")
        
        # Verify repository was called
        mock_functions['get_key_fact_repository'].assert_called_once()
        mock_functions['get_key_fact_repository'].return_value.get_facts_dict.assert_called_once()
        
        # Verify formatter was called with repository results
        mock_functions['format_key_facts_dict'].assert_called_once_with(
            mock_functions['get_key_fact_repository'].return_value.get_facts_dict.return_value
        )
        
        # Check that the formatted key facts are included in the response
        assert "Formatted facts" in result