from unittest.mock import patch, MagicMock

import pytest

from ra_aid.database.repositories.config_repository import ConfigRepositoryManager
from ra_aid.tools.shell import run_shell_command


@pytest.fixture
def mock_console():
    with patch("ra_aid.tools.shell.console") as mock:
        yield mock


@pytest.fixture
def mock_prompt():
    with patch("ra_aid.tools.shell.Prompt") as mock:
        yield mock


@pytest.fixture
def mock_run_interactive():
    with patch("ra_aid.tools.shell.run_interactive_command") as mock:
        mock.return_value = (b"test output", 0)
        yield mock


@pytest.fixture(autouse=True)
def mock_config_repository():
    """Mock the ConfigRepository to avoid database operations during tests"""
    with patch('ra_aid.database.repositories.config_repository.config_repo_var') as mock_repo_var:
        # Setup a mock repository
        mock_repo = MagicMock()
        
        # Create a dictionary to simulate config
        config = {
            "cowboy_mode": False
        }
        
        # Setup get method to return config values (already set up in this file)
        
        # Note: get_all is deprecated, but kept for backward compatibility
        # Setup get_all method to return a reference to the config dict
        mock_repo.get_all.return_value = config
        
        # Setup get method to return config values
        def get_config(key, default=None):
            return config.get(key, default)
        mock_repo.get.side_effect = get_config
        
        # Setup set method to update config values
        def set_config(key, value):
            config[key] = value
        mock_repo.set.side_effect = set_config
        
        # Make the mock context var return our mock repo
        mock_repo_var.get.return_value = mock_repo
        
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


def test_shell_command_cowboy_mode(mock_console, mock_prompt, mock_run_interactive, mock_config_repository):
    """Test shell command execution in cowboy mode (no approval)"""
    # Set cowboy mode to True using the repository
    mock_config_repository.set("cowboy_mode", True)

    result = run_shell_command.invoke({"command": "echo test"})

    assert result["success"] is True
    assert result["return_code"] == 0
    assert "test output" in result["output"]
    mock_prompt.ask.assert_not_called()


def test_shell_command_cowboy_message(mock_console, mock_prompt, mock_run_interactive, mock_config_repository):
    """Test that cowboy mode displays a properly formatted cowboy message with correct spacing"""
    # Set cowboy mode to True using the repository
    mock_config_repository.set("cowboy_mode", True)

    with patch("ra_aid.tools.shell.get_cowboy_message") as mock_get_message:
        mock_get_message.return_value = "🤠 Test cowboy message!"
        result = run_shell_command.invoke({"command": "echo test"})

    assert result["success"] is True
    mock_console.print.assert_any_call("")
    mock_console.print.assert_any_call(" 🤠 Test cowboy message!")
    mock_console.print.assert_any_call("")
    mock_get_message.assert_called_once()


def test_shell_command_interactive_approved(
    mock_console, mock_prompt, mock_run_interactive, mock_config_repository
):
    """Test shell command execution with interactive approval"""
    # Set cowboy mode to False using the repository
    mock_config_repository.set("cowboy_mode", False)
    mock_prompt.ask.return_value = "y"

    result = run_shell_command.invoke({"command": "echo test"})

    assert result["success"] is True
    assert result["return_code"] == 0
    assert "test output" in result["output"]
    mock_prompt.ask.assert_called_once_with(
        "Execute this command? (y=yes, n=no, c=enable cowboy mode for session)",
        choices=["y", "n", "c"],
        default="y",
        show_choices=True,
        show_default=True,
    )


def test_shell_command_interactive_rejected(
    mock_console, mock_prompt, mock_run_interactive, mock_config_repository
):
    """Test shell command rejection in interactive mode"""
    # Set cowboy mode to False using the repository
    mock_config_repository.set("cowboy_mode", False)
    mock_prompt.ask.return_value = "n"

    result = run_shell_command.invoke({"command": "echo test"})

    assert result["success"] is False
    assert result["return_code"] == 1
    assert "cancelled by user" in result["output"]
    mock_prompt.ask.assert_called_once_with(
        "Execute this command? (y=yes, n=no, c=enable cowboy mode for session)",
        choices=["y", "n", "c"],
        default="y",
        show_choices=True,
        show_default=True,
    )
    mock_run_interactive.assert_not_called()


def test_shell_command_execution_error(mock_console, mock_prompt, mock_run_interactive, mock_config_repository):
    """Test handling of shell command execution errors"""
    # Set cowboy mode to True using the repository
    mock_config_repository.set("cowboy_mode", True)
    mock_run_interactive.side_effect = Exception("Command failed")

    result = run_shell_command.invoke({"command": "invalid command"})

    assert result["success"] is False
    assert result["return_code"] == 1
    assert "Command failed" in result["output"]