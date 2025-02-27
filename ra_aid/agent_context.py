"""Context manager for tracking agent state and completion status."""

import threading
from contextlib import contextmanager
from typing import Dict, Optional, Set

# Thread-local storage for context variables
_thread_local = threading.local()


class AgentContext:
    """Context manager for agent state tracking."""

    def __init__(self, parent_context=None):
        """Initialize a new agent context.

        Args:
            parent_context: Optional parent context to inherit state from
        """
        # Initialize completion flags
        self.task_completed = False
        self.plan_completed = False
        self.completion_message = ""
        
        # Inherit state from parent if provided
        if parent_context:
            self.task_completed = parent_context.task_completed
            self.plan_completed = parent_context.plan_completed
            self.completion_message = parent_context.completion_message

    def mark_task_completed(self, message: str) -> None:
        """Mark the current task as completed.

        Args:
            message: Completion message explaining how/why the task is complete
        """
        self.task_completed = True
        self.completion_message = message

    def mark_plan_completed(self, message: str) -> None:
        """Mark the current plan as completed.

        Args:
            message: Completion message explaining how the plan was completed
        """
        self.task_completed = True
        self.plan_completed = True
        self.completion_message = message

    def reset_completion_flags(self) -> None:
        """Reset all completion flags."""
        self.task_completed = False
        self.plan_completed = False
        self.completion_message = ""

    @property
    def is_completed(self) -> bool:
        """Check if the current context is marked as completed."""
        return self.task_completed or self.plan_completed


def get_current_context() -> Optional[AgentContext]:
    """Get the current agent context for this thread.

    Returns:
        The current AgentContext or None if no context is active
    """
    return getattr(_thread_local, "current_context", None)


@contextmanager
def agent_context(parent_context=None):
    """Context manager for agent execution.

    Creates a new agent context and makes it the current context for the duration
    of the with block. Restores the previous context when exiting the block.

    Args:
        parent_context: Optional parent context to inherit state from

    Yields:
        The newly created AgentContext
    """
    # Save the previous context
    previous_context = getattr(_thread_local, "current_context", None)
    
    # Create a new context, inheriting from parent if provided
    # If parent_context is None but previous_context exists, use previous_context as parent
    if parent_context is None and previous_context is not None:
        context = AgentContext(previous_context)
    else:
        context = AgentContext(parent_context)
    
    # Set as current context
    _thread_local.current_context = context
    
    try:
        yield context
    finally:
        # Restore previous context
        _thread_local.current_context = previous_context


def mark_task_completed(message: str) -> None:
    """Mark the current task as completed.

    Args:
        message: Completion message explaining how/why the task is complete
    """
    context = get_current_context()
    if context:
        context.mark_task_completed(message)


def mark_plan_completed(message: str) -> None:
    """Mark the current plan as completed.

    Args:
        message: Completion message explaining how the plan was completed
    """
    context = get_current_context()
    if context:
        context.mark_plan_completed(message)


def reset_completion_flags() -> None:
    """Reset completion flags in the current context."""
    context = get_current_context()
    if context:
        context.reset_completion_flags()


def is_completed() -> bool:
    """Check if the current context is marked as completed.

    Returns:
        True if the current context is marked as completed, False otherwise
    """
    context = get_current_context()
    return context.is_completed if context else False


def get_completion_message() -> str:
    """Get the completion message from the current context.

    Returns:
        The completion message or empty string if no context or no message
    """
    context = get_current_context()
    return context.completion_message if context else ""
