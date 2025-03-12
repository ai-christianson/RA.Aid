"""Repository for managing configuration values."""

import contextvars
from typing import Any, Dict, Optional

# Create contextvar to hold the ConfigRepository instance
config_repo_var = contextvars.ContextVar("config_repo", default=None)


class ConfigRepository:
    """
    Repository for managing configuration values in memory.
    
    This class provides methods to get, set, update, and retrieve all configuration values.
    It does not require database models and operates entirely in memory.
    """
    
    def __init__(self, initial_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ConfigRepository.
        
        Args:
            initial_config: Optional dictionary of initial configuration values
        """
        self._config: Dict[str, Any] = {}
        
        # Initialize with default values from config.py
        from ra_aid.config import (
            DEFAULT_RECURSION_LIMIT,
            DEFAULT_MAX_TEST_CMD_RETRIES,
            DEFAULT_MAX_TOOL_FAILURES,
            FALLBACK_TOOL_MODEL_LIMIT,
            RETRY_FALLBACK_COUNT,
            DEFAULT_TEST_CMD_TIMEOUT,
            DEFAULT_SHOW_COST,
            VALID_PROVIDERS,
        )
        
        self._config = {
            "recursion_limit": DEFAULT_RECURSION_LIMIT,
            "max_test_cmd_retries": DEFAULT_MAX_TEST_CMD_RETRIES,
            "max_tool_failures": DEFAULT_MAX_TOOL_FAILURES,
            "fallback_tool_model_limit": FALLBACK_TOOL_MODEL_LIMIT,
            "retry_fallback_count": RETRY_FALLBACK_COUNT,
            "test_cmd_timeout": DEFAULT_TEST_CMD_TIMEOUT,
            "show_cost": DEFAULT_SHOW_COST,
            "valid_providers": VALID_PROVIDERS,
        }
        
        # Update with any provided initial configuration
        if initial_config:
            self._config.update(initial_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key to retrieve
            default: Default value to return if key is not found
            
        Returns:
            The configuration value or default if not found
        """
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value by key.
        
        Args:
            key: Configuration key to set
            value: Value to set for the key
        """
        self._config[key] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update multiple configuration values at once.
        
        Args:
            config_dict: Dictionary of configuration key-value pairs to update
        """
        self._config.update(config_dict)
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values.
        
        Returns:
            Dictionary containing all configuration values
        """
        return self._config.copy()


class ConfigRepositoryManager:
    """
    Context manager for ConfigRepository.
    
    This class provides a context manager interface for ConfigRepository,
    using the contextvars approach for thread safety.
    
    Example:
        with ConfigRepositoryManager() as repo:
            # Use the repository
            value = repo.get("key")
            repo.set("key", new_value)
    """
    
    def __init__(self, initial_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ConfigRepositoryManager.
        
        Args:
            initial_config: Optional dictionary of initial configuration values
        """
        self.initial_config = initial_config
        
    def __enter__(self) -> 'ConfigRepository':
        """
        Initialize the ConfigRepository and return it.
        
        Returns:
            ConfigRepository: The initialized repository
        """
        repo = ConfigRepository(self.initial_config)
        config_repo_var.set(repo)
        return repo
        
    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[object],
    ) -> None:
        """
        Reset the repository when exiting the context.
        
        Args:
            exc_type: The exception type if an exception was raised
            exc_val: The exception value if an exception was raised
            exc_tb: The traceback if an exception was raised
        """
        # Reset the contextvar to None
        config_repo_var.set(None)
        
        # Don't suppress exceptions
        return False


def get_config_repository() -> ConfigRepository:
    """
    Get the current ConfigRepository instance.
    
    Returns:
        ConfigRepository: The current repository instance
        
    Raises:
        RuntimeError: If no repository is set in the current context
    """
    repo = config_repo_var.get()
    if repo is None:
        raise RuntimeError(
            "ConfigRepository not initialized in current context. "
            "Make sure to use ConfigRepositoryManager."
        )
    return repo