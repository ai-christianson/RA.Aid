"""Utilities for executing and managing user-defined test commands."""

import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ra_aid.logging_config import get_logger
from ra_aid.tools.human import ask_human
from ra_aid.tools.shell import run_shell_command

console = Console()
logger = get_logger(__name__)


@dataclass
class TestState:
    """State for test execution."""

    prompt: str
    test_attempts: int
    auto_test: bool
    should_break: bool = False


class TestCommandExecutor:
    """Class for executing and managing test commands."""

    def __init__(
        self,
        config: Dict[str, Any],
        original_prompt: str,
        test_attempts: int = 0,
        auto_test: bool = False,
    ):
        """Initialize the test command executor.

        Args:
            config: Configuration dictionary containing test settings
            original_prompt: The original prompt to append errors to
            test_attempts: Current number of test attempts
            auto_test: Whether auto-test mode is enabled
        """
        self.config = config
        self.state = TestState(
            prompt=original_prompt,
            test_attempts=test_attempts,
            auto_test=auto_test,
            should_break=False,
        )
        self.max_retries = config.get("max_test_cmd_retries", 5)

    def display_test_failure(self) -> None:
        """Display test failure message."""
        console.print(
            Panel(
                Markdown(
                    f"Test failed. Attempt number {self.state.test_attempts} of {self.max_retries}. Retrying and informing of failure output"
                ),
                title="🔎 User Defined Test",
                border_style="red bold",
            )
        )

    def handle_test_failure(
        self, original_prompt: str, test_result: Dict[str, Any]
    ) -> None:
        """Handle test command failure.

        Args:
            original_prompt: Original prompt text
            test_result: Test command result
        """
        self.state.prompt = f"{original_prompt}. Previous attempt failed with: <test_cmd_stdout>{test_result['output']}</test_cmd_stdout>"
        self.display_test_failure()
        self.state.should_break = False

    def run_test_command(self, cmd: str, original_prompt: str) -> None:
        """Run test command and handle result.

        Args:
            cmd: Test command to execute
            original_prompt: Original prompt text
        """
        timeout = self.config.get("timeout", 30)
        try:
            logger.info(f"Executing test command: {cmd} with timeout {timeout}s")
            test_result = run_shell_command(cmd, timeout=timeout)
            self.state.test_attempts += 1

            if not test_result["success"]:
                self.handle_test_failure(original_prompt, test_result)
                return

            self.state.should_break = True
            logger.info("Test command executed successfully")

        except subprocess.TimeoutExpired:
            logger.warning(f"Test command timed out after {timeout}s: {cmd}")
            self.state.test_attempts += 1
            self.state.prompt = (
                f"{original_prompt}. Previous attempt timed out after {timeout} seconds"
            )
            self.display_test_failure()

        except subprocess.CalledProcessError as e:
            logger.error(
                f"Test command failed with exit code {e.returncode}: {cmd}\nOutput: {e.output}"
            )
            self.state.test_attempts += 1
            self.state.prompt = f"{original_prompt}. Previous attempt failed with exit code {e.returncode}: {e.output}"
            self.display_test_failure()

        except Exception as e:
            logger.warning(f"Test command execution failed: {str(e)}")
            self.state.test_attempts += 1
            self.state.should_break = True

    def handle_user_response(
        self, response: str, cmd: str, original_prompt: str
    ) -> None:
        """Handle user's response to test prompt.
        Args:
            response: User's response (y/n/a)
            cmd: Test command
            original_prompt: Original prompt text
        """
        response = response.strip().lower()

        if response == "n":
            self.state.should_break = True
            return

        if response == "a":
            self.state.auto_test = True
            self.run_test_command(cmd, original_prompt)
            return

        if response == "y":
            self.run_test_command(cmd, original_prompt)

    def check_max_retries(self) -> bool:
        """Check if max retries reached.

        Returns:
            True if max retries reached
        """
        if self.state.test_attempts >= self.max_retries:
            logger.warning("Max test retries reached")
            return True
        return False

    def execute(self) -> Tuple[bool, str, bool, int]:
        """Execute test command and handle retries.

        Returns:
            Tuple containing:
            - bool: Whether to break the retry loop
            - str: Updated prompt
            - bool: Updated auto_test flag
            - int: Updated test_attempts count
        """
        if not self.config.get("test_cmd"):
            self.state.should_break = True
            return (
                self.state.should_break,
                self.state.prompt,
                self.state.auto_test,
                self.state.test_attempts,
            )

        cmd = self.config["test_cmd"]

        if not self.state.auto_test:
            print()
            response = ask_human.invoke(
                {
                    "question": "Would you like to run the test command? (y=yes, n=no, a=enable auto-test)"
                }
            )
            self.handle_user_response(response, cmd, self.state.prompt)
        else:
            if self.check_max_retries():
                logger.error(
                    f"Maximum number of test retries ({self.max_retries}) reached. Stopping test execution."
                )
                console.print(
                    Panel(
                        f"Maximum retries ({self.max_retries}) reached. Test execution stopped.",
                        title="⚠️ Test Execution",
                        border_style="yellow bold",
                    )
                )
                self.state.should_break = True
            else:
                self.run_test_command(cmd, self.state.prompt)

        return (
            self.state.should_break,
            self.state.prompt,
            self.state.auto_test,
            self.state.test_attempts,
        )


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
    executor = TestCommandExecutor(config, original_prompt, test_attempts, auto_test)
    return executor.execute()
