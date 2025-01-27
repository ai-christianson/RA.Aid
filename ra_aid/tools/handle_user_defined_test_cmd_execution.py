"""Utilities for executing and managing user-defined test commands."""

from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from ra_aid.tools.human import ask_human
from ra_aid.tools.shell import run_shell_command
from ra_aid.logging_config import get_logger

console = Console()
logger = get_logger(__name__)

@dataclass
class TestState:
    """State for test execution."""
    prompt: str
    test_attempts: int
    auto_test: bool
    should_break: bool = False

def display_test_failure(attempts: int, max_retries: int) -> None:
    """Display test failure message.
    
    Args:
        attempts: Current number of attempts
        max_retries: Maximum allowed retries
    """
    console.print(
        Panel(
            Markdown(f"Test failed. Attempt number {attempts} of {max_retries}. Retrying and informing of failure output"),
            title="🔎 User Defined Test",
            border_style="red bold"
        )
    )

def handle_test_failure(state: TestState, original_prompt: str, test_result: Dict[str, Any]) -> TestState:
    """Handle test command failure.
    
    Args:
        state: Current test state
        original_prompt: Original prompt text
        test_result: Test command result
        
    Returns:
        Updated test state
    """
    state.prompt = f"{original_prompt}. Previous attempt failed with: <test_cmd_stdout>{test_result['output']}</test_cmd_stdout>"
    display_test_failure(state.test_attempts, 5)  # Default max retries
    state.should_break = False
    return state

def run_test_command(cmd: str, state: TestState, original_prompt: str) -> TestState:
    """Run test command and handle result.
    
    Args:
        cmd: Test command to execute
        state: Current test state
        original_prompt: Original prompt text
        
    Returns:
        Updated test state
    """
    try:
        test_result = run_shell_command(cmd)
        state.test_attempts += 1
        
        if not test_result["success"]:
            return handle_test_failure(state, original_prompt, test_result)
            
        state.should_break = True
        return state
        
    except Exception as e:
        logger.warning(f"Test command execution failed: {str(e)}")
        state.test_attempts += 1
        state.should_break = True
        return state

def handle_user_response(response: str, state: TestState, cmd: str, original_prompt: str) -> TestState:
    """Handle user's response to test prompt.
    
    Args:
        response: User's response (y/n/a)
        state: Current test state
        cmd: Test command
        original_prompt: Original prompt text
        
    Returns:
        Updated test state
    """
    response = response.strip().lower()
    
    if response == "n":
        state.should_break = True
        return state
        
    if response == "a":
        state.auto_test = True
        return run_test_command(cmd, state, original_prompt)
        
    if response == "y":
        return run_test_command(cmd, state, original_prompt)
        
    return state

def check_max_retries(attempts: int, max_retries: int) -> bool:
    """Check if max retries reached.
    
    Args:
        attempts: Current number of attempts
        max_retries: Maximum allowed retries
        
    Returns:
        True if max retries reached
    """
    if attempts >= max_retries:
        logger.warning("Max test retries reached")
        return True
    return False

def execute_test_command(
    config: Dict[str, Any],
    original_prompt: str,
    test_attempts: int = 0,
    auto_test: bool = False,
) -> Tuple[bool, str, bool, int]:
    """Execute a test command and handle retries.

    Args:
        config: Configuration dictionary containing test settings
        original_prompt: The original prompt to append errors to
        test_attempts: Current number of test attempts
        auto_test: Whether auto-test mode is enabled

    Returns:
        Tuple containing:
        - bool: Whether to break the retry loop
        - str: Updated prompt
        - bool: Updated auto_test flag
        - int: Updated test_attempts count
    """
    state = TestState(
        prompt=original_prompt,
        test_attempts=test_attempts,
        auto_test=auto_test
    )

    if not config.get("test_cmd"):
        state.should_break = True
        return state.should_break, state.prompt, state.auto_test, state.test_attempts

    max_retries = config.get("max_test_cmd_retries", 5)
    cmd = config["test_cmd"]

    if not auto_test:
        print()
        response = ask_human.invoke({"question": "Would you like to run the test command? (y=yes, n=no, a=enable auto-test)"})
        state = handle_user_response(response, state, cmd, original_prompt)
    else:
        if check_max_retries(test_attempts, max_retries):
            state.should_break = True
        else:
            state = run_test_command(cmd, state, original_prompt)

    return state.should_break, state.prompt, state.auto_test, state.test_attempts