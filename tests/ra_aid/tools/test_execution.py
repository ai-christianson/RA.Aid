"""Tests for test execution utilities."""

from unittest.mock import patch

import pytest

from ra_aid.tools.handle_user_defined_test_cmd_execution import execute_test_command

# Test cases for execute_test_command
test_cases = [
    # Format: (name, config, original_prompt, test_attempts, auto_test,
    #          mock_responses, expected_result)
    # Case 1: No test command configured
    (
        "no_test_command",
        {"other_config": "value"},
        "original prompt",
        0,
        False,
        {},
        (True, "original prompt", False, 0),
    ),
    # Case 2: User declines to run test
    (
        "user_declines_test",
        {"test_cmd": "pytest"},
        "original prompt",
        0,
        False,
        {"ask_human_response": "n"},
        (True, "original prompt", False, 0),
    ),
    # Case 3: User enables auto-test
    (
        "user_enables_auto_test",
        {"test_cmd": "pytest"},
        "original prompt",
        0,
        False,
        {
            "ask_human_response": "a",
            "shell_cmd_result": {"success": True, "output": "All tests passed"},
        },
        (True, "original prompt", True, 1),
    ),
    # Case 4: Auto-test success
    (
        "auto_test_success",
        {"test_cmd": "pytest"},
        "original prompt",
        0,
        True,
        {"shell_cmd_result": {"success": True, "output": "All tests passed"}},
        (True, "original prompt", True, 1),
    ),
    # Case 5: Auto-test failure with retry
    (
        "auto_test_failure_retry",
        {"test_cmd": "pytest"},
        "original prompt",
        0,
        True,
        {"shell_cmd_result": {"success": False, "output": "Test failed"}},
        (
            False,
            "original prompt. Previous attempt failed with: <test_cmd_stdout>Test failed</test_cmd_stdout>",
            True,
            1,
        ),
    ),
    # Case 6: Max retries reached
    (
        "max_retries_reached",
        {"test_cmd": "pytest", "max_test_cmd_retries": 3},
        "original prompt",
        3,
        True,
        {},
        (True, "original prompt", True, 3),
    ),
    # Case 7: User runs test manually
    (
        "manual_test_success",
        {"test_cmd": "pytest"},
        "original prompt",
        0,
        False,
        {
            "ask_human_response": "y",
            "shell_cmd_result": {"success": True, "output": "All tests passed"},
        },
        (True, "original prompt", False, 1),
    ),
    # Case 8: Manual test failure
    (
        "manual_test_failure",
        {"test_cmd": "pytest"},
        "original prompt",
        0,
        False,
        {
            "ask_human_response": "y",
            "shell_cmd_result": {"success": False, "output": "Test failed"},
        },
        (
            False,
            "original prompt. Previous attempt failed with: <test_cmd_stdout>Test failed</test_cmd_stdout>",
            False,
            1,
        ),
    ),
    # Case 9: Manual test error
    (
        "manual_test_error",
        {"test_cmd": "pytest"},
        "original prompt",
        0,
        False,
        {
            "ask_human_response": "y",
            "shell_cmd_result_error": Exception("Command failed"),
        },
        (True, "original prompt", False, 1),
    ),
    # Case 10: Auto-test error
    (
        "auto_test_error",
        {"test_cmd": "pytest"},
        "original prompt",
        0,
        True,
        {"shell_cmd_result_error": Exception("Command failed")},
        (True, "original prompt", True, 1),
    ),
]


@pytest.mark.parametrize(
    "name,config,original_prompt,test_attempts,auto_test,mock_responses,expected",
    test_cases,
    ids=[case[0] for case in test_cases],
)
def test_execute_test_command(
    name: str,
    config: dict,
    original_prompt: str,
    test_attempts: int,
    auto_test: bool,
    mock_responses: dict,
    expected: tuple,
) -> None:
    """Test execute_test_command with different scenarios.

    Args:
        name: Test case name
        config: Test configuration
        original_prompt: Original prompt text
        test_attempts: Number of test attempts
        auto_test: Auto-test flag
        mock_responses: Mock response data
        expected: Expected result tuple
    """
    with (
        patch(
            "ra_aid.tools.handle_user_defined_test_cmd_execution.ask_human"
        ) as mock_ask_human,
        patch(
            "ra_aid.tools.handle_user_defined_test_cmd_execution.run_shell_command"
        ) as mock_run_cmd,
        patch(
            "ra_aid.tools.handle_user_defined_test_cmd_execution.console"
        ) as _mock_console,
        patch(
            "ra_aid.tools.handle_user_defined_test_cmd_execution.logger"
        ) as mock_logger,
    ):
        # Configure mocks based on mock_responses
        if "ask_human_response" in mock_responses:
            mock_ask_human.invoke.return_value = mock_responses["ask_human_response"]

        if "shell_cmd_result_error" in mock_responses:
            mock_run_cmd.side_effect = mock_responses["shell_cmd_result_error"]
        elif "shell_cmd_result" in mock_responses:
            mock_run_cmd.return_value = mock_responses["shell_cmd_result"]

        # Execute test command
        result = execute_test_command(config, original_prompt, test_attempts, auto_test)

        # Verify result matches expected
        assert result == expected, f"Test case '{name}' failed"

        # Verify mock interactions
        if config.get("test_cmd") and not auto_test:
            mock_ask_human.invoke.assert_called_once()

        if auto_test and test_attempts < config.get("max_test_cmd_retries", 5):
            if config.get("test_cmd"):
                # Verify run_shell_command called with command and default timeout
                mock_run_cmd.assert_called_once_with(
                    config["test_cmd"], timeout=config.get("timeout", 30)
                )

        # Verify logging for max retries
        if test_attempts >= config.get("max_test_cmd_retries", 5):
            mock_logger.warning.assert_called_once_with("Max test retries reached")


def test_execute_test_command_error_handling() -> None:
    """Test error handling in execute_test_command."""
    config = {"test_cmd": "pytest"}

    with (
        patch(
            "ra_aid.tools.handle_user_defined_test_cmd_execution.run_shell_command"
        ) as mock_run_cmd,
        patch(
            "ra_aid.tools.handle_user_defined_test_cmd_execution.logger"
        ) as mock_logger,
    ):
        # Simulate run_shell_command raising an exception
        mock_run_cmd.side_effect = Exception("Command failed")

        result = execute_test_command(config, "original prompt", 0, True)

        # Should handle error and continue
        assert result == (True, "original prompt", True, 1)
        mock_logger.warning.assert_called_once()
