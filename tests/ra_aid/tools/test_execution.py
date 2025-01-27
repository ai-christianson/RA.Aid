"""Tests for test execution utilities."""

import pytest
from unittest.mock import Mock, patch
from rich.console import Console
from ra_aid.tools.handle_user_defined_test_cmd_execution import TestCommandExecutor, DEFAULT_MAX_RETRIES

@pytest.fixture
def executor():
    """Create a TestCommandExecutor fixture."""
    return TestCommandExecutor(console=Mock(spec=Console))

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
        (True, "original prompt", False, 0)
    ),
    
    # Case 2: User declines to run test
    (
        "user_declines_test",
        {"test_cmd": "pytest"},
        "original prompt",
        0,
        False,
        {"ask_human_response": "n"},
        (True, "original prompt", False, 0)
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
            "shell_cmd_result": {"success": True, "output": "All tests passed"}
        },
        (True, "original prompt", True, 1)
    ),
    
    # Case 4: Auto-test success
    (
        "auto_test_success",
        {"test_cmd": "pytest"},
        "original prompt",
        0,
        True,
        {"shell_cmd_result": {"success": True, "output": "All tests passed"}},
        (True, "original prompt", True, 1)
    ),
    
    # Case 5: Auto-test failure with retry
    (
        "auto_test_failure_retry",
        {"test_cmd": "pytest"},
        "original prompt",
        0,
        True,
        {"shell_cmd_result": {"success": False, "output": "Test failed"}},
        (False, "original prompt. Previous attempt failed with: <test_cmd_stdout>Test failed</test_cmd_stdout>", True, 1)
    ),
    
    # Case 6: Max retries reached
    (
        "max_retries_reached",
        {"test_cmd": "pytest"},
        "original prompt",
        DEFAULT_MAX_RETRIES,
        True,
        {"shell_cmd_result": {"success": False, "output": "Test failed"}},
        (True, "original prompt", True, DEFAULT_MAX_RETRIES)
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
            "shell_cmd_result": {"success": True, "output": "All tests passed"}
        },
        (True, "original prompt", False, 1)
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
            "shell_cmd_result": {"success": False, "output": "Test failed"}
        },
        (False, "original prompt. Previous attempt failed with: <test_cmd_stdout>Test failed</test_cmd_stdout>", False, 1)
    ),
]

@pytest.mark.parametrize(
    "name,config,original_prompt,test_attempts,auto_test,mock_responses,expected",
    test_cases,
    ids=[case[0] for case in test_cases]
)
def test_execute_test_command(
    executor,
    name: str,
    config: dict,
    original_prompt: str,
    test_attempts: int,
    auto_test: bool,
    mock_responses: dict,
    expected: tuple,
):
    """Test execute_test_command with different scenarios.
    
    Args:
        executor: TestCommandExecutor instance
        name: Test case name
        config: Test configuration
        original_prompt: Original prompt text
        test_attempts: Number of test attempts
        auto_test: Auto-test flag
        mock_responses: Mock response data
        expected: Expected result tuple
    """
    with patch("ra_aid.tools.handle_user_defined_test_cmd_execution.ask_human") as mock_ask, \
         patch("ra_aid.tools.handle_user_defined_test_cmd_execution.run_shell_command") as mock_run:
        
        # Configure mocks based on responses
        if "ask_human_response" in mock_responses:
            mock_ask.return_value = mock_responses["ask_human_response"]
            
        if "shell_cmd_result" in mock_responses:
            mock_run.return_value = mock_responses["shell_cmd_result"]
            
        result = executor.execute_test_command(
            config,
            original_prompt,
            test_attempts,
            auto_test
        )
        
        assert result == expected

def test_execute_test_command_error_handling(executor):
    """Test error handling in execute_test_command."""
    with patch("ra_aid.tools.handle_user_defined_test_cmd_execution.run_shell_command") as mock_run:
        mock_run.side_effect = Exception("Command failed")
        
        result = executor.execute_test_command(
            {"test_cmd": "pytest"},
            "original prompt",
            auto_test=True
        )
        
        # Should return with should_break=True when an error occurs
        assert result[0] is True  # should_break
        assert result[2] is True  # auto_test
        assert result[3] == 1     # test_attempts