"""Tests for user-defined test command execution utilities."""

import subprocess
from unittest.mock import Mock, patch

import pytest

from ra_aid.tools.handle_user_defined_test_cmd_execution import (
    TestCommandExecutor,
    TestState,
    execute_test_command,
)


@pytest.fixture
def test_state():
    """Create a test state fixture."""
    return TestState(
        prompt="test prompt", test_attempts=0, auto_test=False, should_break=False
    )


@pytest.fixture
def test_executor():
    """Create a test executor fixture."""
    config = {"test_cmd": "test", "max_test_cmd_retries": 3}
    return TestCommandExecutor(config, "test prompt")


def test_check_max_retries(test_executor):
    """Test max retries check."""
    test_executor.state.test_attempts = 2
    assert not test_executor.check_max_retries()

    test_executor.state.test_attempts = 3
    assert test_executor.check_max_retries()

    test_executor.state.test_attempts = 4
    assert test_executor.check_max_retries()


def test_handle_test_failure(test_executor):
    """Test handling of test failures."""
    test_result = {"output": "error message"}
    with patch("ra_aid.tools.handle_user_defined_test_cmd_execution.console.print"):
        test_executor.handle_test_failure("original", test_result)
        assert not test_executor.state.should_break
        assert "error message" in test_executor.state.prompt


def test_run_test_command_success(test_executor):
    """Test successful test command execution."""
    with patch(
        "ra_aid.tools.handle_user_defined_test_cmd_execution.run_shell_command"
    ) as mock_run:
        mock_run.return_value = {"success": True, "output": ""}
        test_executor.run_test_command("test", "original")
        assert test_executor.state.should_break
        assert test_executor.state.test_attempts == 1


def test_run_test_command_failure(test_executor):
    """Test failed test command execution."""
    with patch(
        "ra_aid.tools.handle_user_defined_test_cmd_execution.run_shell_command"
    ) as mock_run:
        mock_run.return_value = {"success": False, "output": "error"}
        test_executor.run_test_command("test", "original")
        assert not test_executor.state.should_break
        assert test_executor.state.test_attempts == 1
        assert "error" in test_executor.state.prompt


def test_run_test_command_error(test_executor):
    """Test test command execution error."""
    with patch(
        "ra_aid.tools.handle_user_defined_test_cmd_execution.run_shell_command"
    ) as mock_run:
        mock_run.side_effect = Exception("Generic error")
        test_executor.run_test_command("test", "original")
        assert test_executor.state.should_break
        assert test_executor.state.test_attempts == 1


def test_run_test_command_timeout(test_executor):
    """Test test command timeout handling."""
    with (
        patch(
            "ra_aid.tools.handle_user_defined_test_cmd_execution.run_shell_command"
        ) as mock_run,
        patch(
            "ra_aid.tools.handle_user_defined_test_cmd_execution.logger.warning"
        ) as mock_logger,
    ):
        # Create a TimeoutExpired exception
        timeout_exc = subprocess.TimeoutExpired(cmd="test", timeout=30)
        mock_run.side_effect = timeout_exc

        test_executor.run_test_command("test", "original")

        # Verify state updates
        assert not test_executor.state.should_break
        assert test_executor.state.test_attempts == 1
        assert "timed out after 30 seconds" in test_executor.state.prompt

        # Verify logging
        mock_logger.assert_called_once()


def test_run_test_command_called_process_error(test_executor):
    """Test handling of CalledProcessError exception."""
    with (
        patch(
            "ra_aid.tools.handle_user_defined_test_cmd_execution.run_shell_command"
        ) as mock_run,
        patch(
            "ra_aid.tools.handle_user_defined_test_cmd_execution.logger.error"
        ) as mock_logger,
    ):
        # Create a CalledProcessError exception
        process_error = subprocess.CalledProcessError(
            returncode=1, cmd="test", output="Command failed output"
        )
        mock_run.side_effect = process_error

        test_executor.run_test_command("test", "original")

        # Verify state updates
        assert not test_executor.state.should_break
        assert test_executor.state.test_attempts == 1
        assert "failed with exit code 1" in test_executor.state.prompt

        # Verify logging
        mock_logger.assert_called_once()


def test_handle_user_response_no(test_executor):
    """Test handling of 'no' response."""
    test_executor.handle_user_response("n", "test", "original")
    assert test_executor.state.should_break
    assert not test_executor.state.auto_test


def test_handle_user_response_auto(test_executor):
    """Test handling of 'auto' response."""
    with patch.object(test_executor, "run_test_command") as mock_run:
        test_executor.handle_user_response("a", "test", "original")
        assert test_executor.state.auto_test
        mock_run.assert_called_once_with("test", "original")


def test_handle_user_response_yes(test_executor):
    """Test handling of 'yes' response."""
    with patch.object(test_executor, "run_test_command") as mock_run:
        test_executor.handle_user_response("y", "test", "original")
        assert not test_executor.state.auto_test
        mock_run.assert_called_once_with("test", "original")


def test_execute_no_cmd():
    """Test execution with no test command."""
    executor = TestCommandExecutor({}, "prompt")
    result = executor.execute()
    assert result == (True, "prompt", False, 0)


def test_execute_manual():
    """Test manual test execution."""
    config = {"test_cmd": "test"}
    executor = TestCommandExecutor(config, "prompt")

    def mock_handle_response(response, cmd, prompt):
        # Simulate the behavior of handle_user_response and run_test_command
        executor.state.should_break = True
        executor.state.test_attempts = 1
        executor.state.prompt = "new prompt"

    with (
        patch(
            "ra_aid.tools.handle_user_defined_test_cmd_execution.ask_human"
        ) as mock_ask,
        patch.object(
            executor, "handle_user_response", side_effect=mock_handle_response
        ) as mock_handle,
    ):
        mock_ask.invoke.return_value = "y"

        result = executor.execute()
        mock_handle.assert_called_once_with("y", "test", "prompt")
        assert result == (True, "new prompt", False, 1)


def test_execute_auto():
    """Test auto test execution."""
    config = {"test_cmd": "test", "max_test_cmd_retries": 3}
    executor = TestCommandExecutor(config, "prompt", auto_test=True)

    # Set up state before creating mock
    executor.state.test_attempts = 1
    executor.state.should_break = True

    with patch.object(executor, "run_test_command") as mock_run:
        result = executor.execute()
        assert result == (True, "prompt", True, 1)
        mock_run.assert_called_once_with("test", "prompt")


def test_execute_test_command_function():
    """Test the execute_test_command function."""
    config = {"test_cmd": "test"}
    with patch(
        "ra_aid.tools.handle_user_defined_test_cmd_execution.TestCommandExecutor"
    ) as mock_executor_class:
        mock_executor = Mock()
        mock_executor.execute.return_value = (True, "new prompt", True, 1)
        mock_executor_class.return_value = mock_executor

        result = execute_test_command(config, "prompt", auto_test=True)
        assert result == (True, "new prompt", True, 1)
        mock_executor_class.assert_called_once_with(config, "prompt", 0, True)
        mock_executor.execute.assert_called_once()
