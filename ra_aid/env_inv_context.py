"""
Context management for environment inventory.

This module provides thread-safe access to environment inventory information
using context variables.
"""

import contextvars
from typing import Dict, Any, Optional, Type

# Create contextvar to hold the environment inventory
env_inv_var = contextvars.ContextVar("env_inv", default=None)


class EnvInvManager:
    """
    Context manager for environment inventory.

    This class provides a context manager interface for environment inventory,
    using the contextvars approach for thread safety.

    Example:
        from ra_aid.env_inv import EnvDiscovery
        
        # Get environment inventory
        env_discovery = EnvDiscovery()
        env_discovery.discover()
        env_data = env_discovery.format_markdown()
        
        # Set as current environment inventory
        with EnvInvManager(env_data) as env_mgr:
            # Environment inventory is now available through get_env_inv()
            pass
    """

    def __init__(self, env_data: Dict[str, Any]):
        """
        Initialize the EnvInvManager.

        Args:
            env_data: Dictionary containing environment inventory data
        """
        self.env_data = env_data

    def __enter__(self) -> 'EnvInvManager':
        """
        Set the environment inventory and return self.

        Returns:
            EnvInvManager: The initialized manager
        """
        env_inv_var.set(self.env_data)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        """
        Reset the environment inventory when exiting the context.

        Args:
            exc_type: The exception type if an exception was raised
            exc_val: The exception value if an exception was raised
            exc_tb: The traceback if an exception was raised
        """
        # Reset the contextvar to None
        env_inv_var.set(None)

        # Don't suppress exceptions
        return False


def get_env_inv() -> Dict[str, Any]:
    """
    Get the current environment inventory.

    Returns:
        Dict[str, Any]: The current environment inventory

    Raises:
        RuntimeError: If no environment inventory has been initialized with EnvInvManager
    """
    env_data = env_inv_var.get()
    if env_data is None:
        raise RuntimeError(
            "No environment inventory available. "
            "Make sure to initialize one with EnvInvManager first."
        )
    return env_data
