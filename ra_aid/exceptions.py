"""Custom exceptions for RA.Aid."""

class AgentInterrupt(Exception):
    """Exception raised when an agent's execution is interrupted.
    
    This exception is used for internal agent interruption handling,
    separate from KeyboardInterrupt which is reserved for top-level handling.
    """
    pass
