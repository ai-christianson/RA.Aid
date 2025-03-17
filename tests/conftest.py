"""
Global pytest fixtures for RA-AID tests.

This module provides global fixtures that are automatically applied to all tests,
ensuring consistent test environments and proper isolation.
"""

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_config_repository():
    """Mock the config repository."""
    from ra_aid.database.repositories.config_repository import (
        config_repo_var,
        ConfigRepository,
    )

    # Create a real ConfigRepository instance
    repo = ConfigRepository()
    # Set default config values
    repo._config["recursion_limit"] = 2
    repo._config["show_cost"] = False

    # Set the contextvar to use our repository
    token = config_repo_var.set(repo)

    yield repo

    # Reset the contextvar when done
    config_repo_var.reset(token)


@pytest.fixture()
def mock_trajectory_repository():
    """Mock the TrajectoryRepository to avoid database operations during tests."""
    with patch(
        "ra_aid.database.repositories.trajectory_repository.TrajectoryRepository"
    ) as mock:
        # Setup a mock repository
        mock_repo = MagicMock()
        mock_repo.create.return_value = MagicMock(id=1)
        mock.return_value = mock_repo
        yield mock_repo


@pytest.fixture()
def mock_human_input_repository():
    """Mock the HumanInputRepository to avoid database operations during tests."""
    with patch(
        "ra_aid.database.repositories.human_input_repository.HumanInputRepository"
    ) as mock:
        # Setup a mock repository
        mock_repo = MagicMock()
        mock_repo.get_most_recent_id.return_value = 1
        mock_repo.create.return_value = MagicMock(id=1)
        mock.return_value = mock_repo
        yield mock_repo


@pytest.fixture(autouse=True)
def mock_session_repository():
    """Mock the SessionRepository to avoid database operations during tests."""
    with patch(
        "ra_aid.database.repositories.session_repository.SessionRepository"
    ) as mock:
        # Setup a mock repository
        mock_repo = MagicMock()
        session_record = MagicMock()
        session_record.id = 1
        session_record.get_id.return_value = 1
        mock_repo.get_current_session_record.return_value = session_record
        mock.return_value = mock_repo

        # Set the contextvar
        from ra_aid.database.repositories.session_repository import session_repo_var

        token = session_repo_var.set(mock_repo)

        yield mock_repo

        # Reset the contextvar
        session_repo_var.reset(token)


@pytest.fixture()
def mock_repository_access(
    mock_trajectory_repository, mock_human_input_repository, mock_session_repository
):
    """Mock all repository accessor functions."""
    with patch(
        "ra_aid.database.repositories.trajectory_repository.get_trajectory_repository",
        return_value=mock_trajectory_repository,
    ):
        with patch(
            "ra_aid.database.repositories.human_input_repository.get_human_input_repository",
            return_value=mock_human_input_repository,
        ):
            with patch(
                "ra_aid.database.repositories.session_repository.get_session_repository",
                return_value=mock_session_repository,
            ):
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
