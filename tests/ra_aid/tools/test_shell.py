from unittest.mock import patch

import pytest

from ra_aid.tools.memory import _global_memory
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


def test_shell_command_cowboy_mode(mock_console, mock_prompt, mock_run_interactive):
    """Test shell command execution in cowboy mode (no approval)"""
    _global_memory["config"] = {"cowboy_mode": True}

    result = run_shell_command.invoke({"command": "echo test"})

    assert result["success"] is True
    assert result["return_code"] == 0
    assert "test output" in result["output"]
    mock_prompt.ask.assert_not_called()


def test_shell_command_cowboy_message(mock_console, mock_prompt, mock_run_interactive):
    """Test that cowboy mode displays a properly formatted cowboy message with correct spacing"""
    _global_memory["config"] = {"cowboy_mode": True}

    with patch("ra_aid.tools.shell.get_cowboy_message") as mock_get_message:
        mock_get_message.return_value = "🤠 Test cowboy message!"
        result = run_shell_command.invoke({"command": "echo test"})

    assert result["success"] is True
    mock_console.print.assert_any_call("")
    mock_console.print.assert_any_call(" 🤠 Test cowboy message!")
    mock_console.print.assert_any_call("")
    mock_get_message.assert_called_once()


def test_shell_command_interactive_approved(
    mock_console, mock_prompt, mock_run_interactive
):
    """Test shell command execution with interactive approval"""
    _global_memory["config"] = {"cowboy_mode": False}
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
    mock_console, mock_prompt, mock_run_interactive
):
    """Test shell command rejection in interactive mode"""
    _global_memory["config"] = {"cowboy_mode": False}
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


def test_shell_command_execution_error(mock_console, mock_prompt, mock_run_interactive):
    """Test handling of shell command execution errors"""
    _global_memory["config"] = {"cowboy_mode": True}
    mock_run_interactive.side_effect = Exception("Command failed")

    result = run_shell_command.invoke({"command": "invalid command"})

    assert result["success"] is False
    assert result["return_code"] == 1
    assert "Command failed" in result["output"]
