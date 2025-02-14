"""Custom exceptions for RA.Aid."""

from typing import Optional

from langchain_core.messages import BaseMessage


class AgentInterrupt(Exception):
    """Exception raised when an agent's execution is interrupted.

    This exception is used for internal agent interruption handling,
    separate from KeyboardInterrupt which is reserved for top-level handling.
    """

    pass


class ToolExecutionError(Exception):
    """Exception raised when a tool execution fails.

    This exception is used to distinguish tool execution failures
    from other types of errors in the agent system.
    """

    def __init__(
        self,
        message: str,
        base_message: Optional[BaseMessage] = None,
        tool_name: Optional[str] = None,
    ):
        super().__init__(message)
        self.base_message = base_message
        self.tool_name = tool_name


class FallbackToolExecutionError(Exception):
    """Exception raised when a fallback tool execution fails."""

    pass
