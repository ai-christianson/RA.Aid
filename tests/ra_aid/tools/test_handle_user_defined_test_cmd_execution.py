"""Tests for user-defined test command execution utilities."""

import pytest
from unittest.mock import patch, Mock
from rich.console import Console
from ra_aid.tools.handle_user_defined_test_cmd_execution import (
    TestState,
    TestCommandExecutor
)

@pytest.fixture
def test_state():
    """Create a test state fixture."""
    return TestState(
        prompt="test prompt",
        test_attempts=0,
        auto_test=False
    )

@pytest.fixture
def executor():
    """Create a TestCommandExecutor fixture."""
    return TestCommandExecutor(console=Mock(spec=Console))

def test_check_max_retries(executor):
    """Test max retries check."""
    assert not executor.check_max_retries(2)
    assert executor.check_max_retries(5)
    assert executor.check_max_retries(6)

def test_handle_test_failure(executor, test_state):
    """Test handling of test failures."""
    test_result = {"output": "error message"}
    state = executor.handle_test_failure(test_state, "original", test_result)
    assert not state.should_break
    assert "error message" in state.prompt
    executor.console.print.assert_called_once()

def test_run_test_command_success(executor, test_state):
    """Test successful test command execution."""
    with patch("ra_aid.tools.handle_user_defined_test_cmd_execution.run_shell_command") as mock_run:
        mock_run.return_value = {"success": True, "output": ""}
        state = executor.run_test_command("test", test_state, "original")
        assert state.should_break
        assert state.test_attempts == 1

def test_run_test_command_failure(executor, test_state):
    """Test failed test command execution."""
    with patch("ra_aid.tools.handle_user_defined_test_cmd_execution.run_shell_command") as mock_run:
        mock_run.return_value = {"success": False, "output": "error"}
        state = executor.run_test_command("test", test_state, "original")
        assert not state.should_break
        assert state.test_attempts == 1
        assert "error" in state.prompt

def test_run_test_command_error(executor, test_state):
    """Test test command execution error."""
    with patch("ra_aid.tools.handle_user_defined_test_cmd_execution.run_shell_command") as mock_run:
        mock_run.side_effect = Exception("Command failed")
        state = executor.run_test_command("test", test_state, "original")
        assert state.should_break
        assert state.test_attempts == 1

def test_handle_user_response_no(executor, test_state):
    """Test handling of 'no' response."""
    state = executor.handle_user_response("n", test_state, "test", "original")
    assert state.should_break
    assert not state.auto_test

def test_handle_user_response_auto(executor, test_state):
    """Test handling of 'auto' response."""
    with patch.object(executor, "run_test_command") as mock_run:
        mock_state = TestState("prompt", 1, True, True)
        mock_run.return_value = mock_state
        state = executor.handle_user_response("a", test_state, "test", "original")
        assert state.auto_test
        mock_run.assert_called_once_with("test", test_state, "original")

def test_handle_user_response_yes(executor, test_state):
    """Test handling of 'yes' response."""
    with patch.object(executor, "run_test_command") as mock_run:
        mock_state = TestState("prompt", 1, False, True)
        mock_run.return_value = mock_state
        state = executor.handle_user_response("y", test_state, "test", "original")
        assert not state.auto_test
        mock_run.assert_called_once_with("test", test_state, "original")

def test_execute_test_command_no_cmd(executor):
    """Test execution with no test command."""
    result = executor.execute_test_command({}, "prompt")
    assert result == (True, "prompt", False, 0)

def test_execute_test_command_manual(executor):
    """Test manual test execution."""
    config = {"test_cmd": "test"}
    with patch("ra_aid.tools.handle_user_defined_test_cmd_execution.ask_human") as mock_ask, \
         patch.object(executor, "handle_user_response") as mock_handle:
        mock_ask.return_value = "y"
        mock_state = TestState("new prompt", 1, False, True)
        mock_handle.return_value = mock_state
        result = executor.execute_test_command(config, "prompt")
        assert result == (True, "new prompt", False, 1)