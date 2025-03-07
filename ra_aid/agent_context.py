"""Context manager for tracking agent state and completion status."""

import threading  # Keep for backward compatibility
import contextvars
from contextlib import contextmanager
from typing import Optional

# Create contextvar to hold the agent context
agent_context_var = contextvars.ContextVar("agent_context", default=None)


class AgentContext:
    """Context manager for agent state tracking."""

    def __init__(self, parent_context=None):
        """Initialize a new agent context.

        Args:
            parent_context: Optional parent context to inherit state from
        """
        # Store reference to parent context
        self.parent = parent_context

        # Initialize completion flags
        self.task_completed = False
        self.plan_completed = False
        self.completion_message = ""
        self.agent_should_exit = False
        self.agent_has_crashed = False
        self.agent_crashed_message = None

        # Note: Completion flags (task_completed, plan_completed, completion_message,
        # agent_should_exit) are no longer inherited from parent contexts

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

    def mark_should_exit(self, propagation_depth: Optional[int] = 0) -> None:
        """Mark that the agent should exit execution.

        Args:
            propagation_depth: How far up the context hierarchy to propagate the flag.
                None: Propagate to all parent contexts
                0 (default): Only mark the current context
                1: Mark the current context and its immediate parent
                2+: Propagate up the specified number of levels
        """
        self.agent_should_exit = True

        # Propagate to parent context based on propagation_depth
        if propagation_depth is None:
            # Maintain current behavior of unlimited propagation
            if self.parent:
                self.parent.mark_should_exit(propagation_depth)
        elif propagation_depth > 0:
            # Propagate to parent with decremented depth
            if self.parent:
                self.parent.mark_should_exit(propagation_depth - 1)
        # If propagation_depth is 0, don't propagate to parent

    def mark_agent_crashed(self, message: str) -> None:
        """Mark the agent as crashed with the given message.

        Unlike exit state, crash state does not propagate to parent contexts.

        Args:
            message: Error message explaining the crash
        """
        self.agent_has_crashed = True
        self.agent_crashed_message = message

    def is_crashed(self) -> bool:
        """Check if the agent has crashed.

        Returns:
            True if the agent has crashed, False otherwise
        """
        return self.agent_has_crashed

    @property
    def is_completed(self) -> bool:
        """Check if the current context is marked as completed."""
        return self.task_completed or self.plan_completed
        
    @property
    def depth(self) -> int:
        """Calculate the depth of this context based on parent chain.
        
        Returns:
            int: 0 for a context with no parent, parent.depth + 1 otherwise
        """
        if self.parent is None:
            return 0
        return self.parent.depth + 1


def get_current_context() -> Optional[AgentContext]:
    """Get the current agent context for this thread.

    Returns:
        The current AgentContext or None if no context is active
    """
    return agent_context_var.get()


def get_depth() -> int:
    """Get the depth of the current agent context.
    
    Returns:
        int: Depth of the current context, or 0 if no context exists
    """
    ctx = get_current_context()
    if ctx is None:
        return 0
    return ctx.depth


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
    previous_context = agent_context_var.get()

    # Create a new context, inheriting from parent if provided
    # If parent_context is None but previous_context exists, use previous_context as parent
    if parent_context is None and previous_context is not None:
        context = AgentContext(previous_context)
    else:
        context = AgentContext(parent_context)

    # Set as current context and get token for resetting later
    token = agent_context_var.set(context)

    try:
        yield context
    finally:
        # Restore previous context
        agent_context_var.reset(token)


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


def should_exit() -> bool:
    """Check if the agent should exit execution.

    Returns:
        True if the agent should exit, False otherwise
    """
    context = get_current_context()
    return context.agent_should_exit if context else False


def mark_should_exit(propagation_depth: Optional[int] = 0) -> None:
    """Mark that the agent should exit execution.
    
    Args:
        propagation_depth: How far up the context hierarchy to propagate the flag.
            None: Propagate to all parent contexts
            0 (default): Only mark the current context
            1: Mark the current context and its immediate parent
            2+: Propagate up the specified number of levels
    """
    context = get_current_context()
    if context:
        context.mark_should_exit(propagation_depth)


def is_crashed() -> bool:
    """Check if the current agent has crashed.

    Returns:
        True if the current agent has crashed, False otherwise
    """
    context = get_current_context()
    return context.is_crashed() if context else False


def mark_agent_crashed(message: str) -> None:
    """Mark the current agent as crashed with the given message.

    Args:
        message: Error message explaining the crash
    """
    context = get_current_context()
    if context:
        context.mark_agent_crashed(message)


def get_crash_message() -> Optional[str]:
    """Get the crash message from the current context.

    Returns:
        The crash message or None if the agent has not crashed
    """
    context = get_current_context()
    return context.agent_crashed_message if context and context.is_crashed() else None