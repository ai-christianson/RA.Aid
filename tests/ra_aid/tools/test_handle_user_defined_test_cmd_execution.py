"""Tests for user-defined test command execution utilities."""

import pytest
from unittest.mock import patch, Mock
from ra_aid.tools.handle_user_defined_test_cmd_execution import (
    TestState,
    execute_test_command,
    handle_test_failure,
    run_test_command,
    handle_user_response,
    check_max_retries
)

@pytest.fixture
def test_state():
    """Create a test state fixture."""
    return TestState(
        prompt="test prompt",
        test_attempts=0,
        auto_test=False
    )

def test_check_max_retries():
    """Test max retries check."""
    assert not check_max_retries(2, 3)
    assert check_max_retries(3, 3)
    assert check_max_retries(4, 3)

def test_handle_test_failure(test_state):
    """Test handling of test failures."""
    test_result = {"output": "error message"}
    with patch("ra_aid.tools.handle_user_defined_test_cmd_execution.display_test_failure"):
        state = handle_test_failure(test_state, "original", test_result)
        assert not state.should_break
        assert "error message" in state.prompt

def test_run_test_command_success(test_state):
    """Test successful test command execution."""
    with patch("ra_aid.tools.handle_user_defined_test_cmd_execution.run_shell_command") as mock_run:
        mock_run.return_value = {"success": True, "output": ""}
        state = run_test_command("test", test_state, "original")
        assert state.should_break
        assert state.test_attempts == 1

def test_run_test_command_failure(test_state):
    """Test failed test command execution."""
    with patch("ra_aid.tools.handle_user_defined_test_cmd_execution.run_shell_command") as mock_run:
        mock_run.return_value = {"success": False, "output": "error"}
        state = run_test_command("test", test_state, "original")
        assert not state.should_break
        assert state.test_attempts == 1
        assert "error" in state.prompt

def test_run_test_command_error(test_state):
    """Test test command execution error."""
    with patch("ra_aid.tools.handle_user_defined_test_cmd_execution.run_shell_command") as mock_run:
        mock_run.side_effect = Exception("Command failed")
        state = run_test_command("test", test_state, "original")
        assert state.should_break
        assert state.test_attempts == 1

def test_handle_user_response_no(test_state):
    """Test handling of 'no' response."""
    state = handle_user_response("n", test_state, "test", "original")
    assert state.should_break
    assert not state.auto_test

def test_handle_user_response_auto(test_state):
    """Test handling of 'auto' response."""
    with patch("ra_aid.tools.handle_user_defined_test_cmd_execution.run_test_command") as mock_run:
        mock_state = TestState("prompt", 1, True, True)
        mock_run.return_value = mock_state
        state = handle_user_response("a", test_state, "test", "original")
        assert state.auto_test
        mock_run.assert_called_once()

def test_handle_user_response_yes(test_state):
    """Test handling of 'yes' response."""
    with patch("ra_aid.tools.handle_user_defined_test_cmd_execution.run_test_command") as mock_run:
        mock_state = TestState("prompt", 1, False, True)
        mock_run.return_value = mock_state
        state = handle_user_response("y", test_state, "test", "original")
        assert not state.auto_test
        mock_run.assert_called_once()

def test_execute_test_command_no_cmd():
    """Test execution with no test command."""
    result = execute_test_command({}, "prompt")
    assert result == (True, "prompt", False, 0)

def test_execute_test_command_manual():
    """Test manual test execution."""
    config = {"test_cmd": "test"}
    with patch("ra_aid.tools.handle_user_defined_test_cmd_execution.ask_human") as mock_ask, \
         patch("ra_aid.tools.handle_user_defined_test_cmd_execution.handle_user_response") as mock_handle:
        mock_ask.invoke.return_value = "y"
        mock_state = TestState("new prompt", 1, False, True)
        mock_handle.return_value = mock_state
        result = execute_test_command(config, "prompt")
        assert result == (True, "new prompt", False, 1)

def test_execute_test_command_auto():
    """Test auto test execution."""
    config = {"test_cmd": "test", "max_test_cmd_retries": 3}
    with patch("ra_aid.tools.handle_user_defined_test_cmd_execution.run_test_command") as mock_run:
        mock_state = TestState("new prompt", 1, True, True)
        mock_run.return_value = mock_state
        result = execute_test_command(config, "prompt", auto_test=True)
        assert result == (True, "new prompt", True, 1)