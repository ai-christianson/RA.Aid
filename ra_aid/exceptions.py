"""Custom exceptions for RA.Aid."""


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

    pass
