"""Utilities for executing and managing user-defined test commands."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from ra_aid.tools.human import ask_human
from ra_aid.tools.shell import run_shell_command
from ra_aid.logging_config import get_logger

DEFAULT_MAX_RETRIES = 5

@dataclass
class TestState:
    """State for test execution."""
    prompt: str
    test_attempts: int
    auto_test: bool
    should_break: bool = False

class TestCommandStrategy(ABC):
    """Base strategy for test command execution."""
    
    @abstractmethod
    def execute(self, cmd: str, state: TestState, original_prompt: str) -> TestState:
        """Execute test command."""
        pass

class AutoTestStrategy(TestCommandStrategy):
    """Strategy for automatic test execution."""
    
    def __init__(self, executor):
        self.executor = executor
    
    def execute(self, cmd: str, state: TestState, original_prompt: str) -> TestState:
        return self.executor.run_test_command(cmd, state, original_prompt)

class ManualTestStrategy(TestCommandStrategy):
    """Strategy for manual test execution."""
    
    def __init__(self, executor):
        self.executor = executor
    
    def execute(self, cmd: str, state: TestState, original_prompt: str) -> TestState:
        response = ask_human("Do you want to run the test command? (y/n/a)")
        return self.executor.handle_user_response(response, state, cmd, original_prompt)

class TestCommandExecutor:
    """Class for executing and managing user-defined test commands."""
    
    def __init__(self, console: Optional[Console] = None, max_retries: int = DEFAULT_MAX_RETRIES):
        self.console = console or Console()
        self.logger = get_logger(__name__)
        self.max_retries = max_retries
        self.auto_strategy = AutoTestStrategy(self)
        self.manual_strategy = ManualTestStrategy(self)

    def display_test_failure(self, attempts: int) -> None:
        """Display test failure message."""
        self.console.print(
            Panel(
                Markdown(f"Test failed. Attempt number {attempts} of {self.max_retries}. Retrying and informing of failure output"),
                title="🔎 User Defined Test",
                border_style="red bold"
            )
        )

    def check_max_retries(self, attempts: int) -> bool:
        """Check if max retries reached."""
        return attempts >= self.max_retries

    def handle_test_failure(self, state: TestState, original_prompt: str, test_result: Dict[str, Any]) -> TestState:
        """Handle test command failure."""
        state.prompt = f"{original_prompt}. Previous attempt failed with: <test_cmd_stdout>{test_result['output']}</test_cmd_stdout>"
        self.display_test_failure(state.test_attempts)
        state.should_break = False
        return state

    def run_test_command(self, cmd: str, state: TestState, original_prompt: str) -> TestState:
        """Run test command and handle result."""
        try:
            test_result = run_shell_command(cmd)
            return self._handle_test_result(test_result, state, original_prompt)
        except Exception as e:
            self.logger.warning(f"Test command execution failed: {str(e)}")
            state.test_attempts += 1
            state.should_break = True
            return state

    def _handle_test_result(self, test_result: Dict[str, Any], state: TestState, original_prompt: str) -> TestState:
        """Handle test command result."""
        if not test_result["success"]:
            state.test_attempts += 1
            return self.handle_test_failure(state, original_prompt, test_result)
        state.test_attempts += 1
        state.should_break = True
        return state

    def handle_user_response(self, response: str, state: TestState, cmd: str, original_prompt: str) -> TestState:
        """Handle user's response to test prompt."""
        response = response.strip().lower()
        if response == "n":
            state.should_break = True
            return state
        if response == "a":
            state.auto_test = True
        return self.run_test_command(cmd, state, original_prompt)

    def execute_test_command(
        self,
        config: Dict[str, Any],
        original_prompt: str,
        test_attempts: int = 0,
        auto_test: bool = False,
    ) -> Tuple[bool, str, bool, int]:
        """Execute a test command and handle retries."""
        if "test_cmd" not in config:
            return True, original_prompt, auto_test, test_attempts

        if auto_test and test_attempts >= self.max_retries:
            self.logger.warning("Max test retries reached")
            return True, original_prompt, auto_test, test_attempts

        state = TestState(original_prompt, test_attempts, auto_test)
        strategy = self.auto_strategy if auto_test else self.manual_strategy
        state = strategy.execute(config["test_cmd"], state, original_prompt)
        return state.should_break, state.prompt, state.auto_test, state.test_attempts