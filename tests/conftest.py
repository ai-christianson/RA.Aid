"""
Global pytest fixtures for RA-AID tests.

This module provides global fixtures that are automatically applied to all tests,
ensuring consistent test environments and proper isolation.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def mock_config_repository():
    """Mock the config repository."""
    from ra_aid.database.repositories.config_repository import get_config_repository

    repo = MagicMock()
    # Default config values
    config_values = {"recursion_limit": 2}
    repo.get_all.return_value = config_values
    repo.get.side_effect = lambda key, default=None: config_values.get(key, default)
    get_config_repository.return_value = repo
    yield repo


@pytest.fixture()
def mock_trajectory_repository():
    """Mock the TrajectoryRepository to avoid database operations during tests."""
    with patch('ra_aid.database.repositories.trajectory_repository.TrajectoryRepository') as mock:
        # Setup a mock repository
        mock_repo = MagicMock()
        mock_repo.create.return_value = MagicMock(id=1)
        mock.return_value = mock_repo
        yield mock_repo


@pytest.fixture()
def mock_human_input_repository():
    """Mock the HumanInputRepository to avoid database operations during tests."""
    with patch('ra_aid.database.repositories.human_input_repository.HumanInputRepository') as mock:
        # Setup a mock repository
        mock_repo = MagicMock()
        mock_repo.get_most_recent_id.return_value = 1
        mock_repo.create.return_value = MagicMock(id=1)
        mock.return_value = mock_repo
        yield mock_repo


@pytest.fixture()
def mock_repository_access(mock_trajectory_repository, mock_human_input_repository):
    """Mock all repository accessor functions."""
    with patch('ra_aid.database.repositories.trajectory_repository.get_trajectory_repository', 
              return_value=mock_trajectory_repository):
        with patch('ra_aid.database.repositories.human_input_repository.get_human_input_repository',
                  return_value=mock_human_input_repository):
            yield


@pytest.fixture(autouse=True)
def isolated_db_environment(tmp_path, monkeypatch):
    """
    Fixture to ensure all database operations during tests use a temporary directory.
    
    This fixture automatically applies to all tests. It mocks os.getcwd() to return
    a temporary directory path, ensuring that database operations never touch the
    actual .ra-aid directory in the current working directory.
    
    Args:
        tmp_path: Pytest fixture that provides a temporary directory for the test
        monkeypatch: Pytest fixture for modifying environment and functions
    """
    # Store the original current working directory
    original_cwd = os.getcwd()
    
    # Get the absolute path of the temporary directory
    tmp_path_str = str(tmp_path.absolute())
    
    # Create the .ra-aid directory in the temporary path
    ra_aid_dir = tmp_path / ".ra-aid"
    ra_aid_dir.mkdir(exist_ok=True)
    
    # Mock os.getcwd() to return the temporary directory path
    monkeypatch.setattr(os, "getcwd", lambda: tmp_path_str)
    
    # Run the test
    yield tmp_path
    
    # No need to restore os.getcwd() as monkeypatch does this automatically
    # No need to assert original_cwd is restored, as it's just the function that's mocked,
    # not the actual working directory