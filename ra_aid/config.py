"""Configuration utilities."""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

DEFAULT_RECURSION_LIMIT = 100
DEFAULT_MAX_TEST_CMD_RETRIES = 3
DEFAULT_MAX_TOOL_FAILURES = 3
FALLBACK_TOOL_MODEL_LIMIT = 5
RETRY_FALLBACK_COUNT = 3
DEFAULT_TEST_CMD_TIMEOUT = 60 * 5  # 5 minutes in seconds

# Function to get the configuration file path
def get_config_file_path(project_state_dir: Optional[str] = None) -> Path:
    """Get the path to the configuration file.

    Args:
        project_state_dir: Optional custom directory to use instead of .ra-aid in current directory

    Returns:
        Path: The path to the configuration file
    """
    if project_state_dir:
        ra_aid_dir = Path(project_state_dir)
    else:
        cwd = os.getcwd()
        ra_aid_dir = Path(os.path.join(cwd, ".ra-aid"))

    # Create the directory if it doesn't exist
    ra_aid_dir.mkdir(parents=True, exist_ok=True)

    return ra_aid_dir / "config.json"

# Function to load default values from the configuration file
def load_default_values(project_state_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load default values from the configuration file.

    Args:
        project_state_dir: Optional custom directory to use instead of .ra-aid in current directory

    Returns:
        Dict[str, Any]: The loaded default values
    """
    config_file = get_config_file_path(project_state_dir)
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # If the file is corrupted or can't be read, return empty defaults
            return {}
    return {}

# Function to save default values to the configuration file
def save_default_values(values: Dict[str, Any], project_state_dir: Optional[str] = None) -> None:
    """Save default values to the configuration file.

    Args:
        values: The values to save
        project_state_dir: Optional custom directory to use instead of .ra-aid in current directory
    """
    config_file = get_config_file_path(project_state_dir)
    try:
        # Load existing values first to avoid overwriting other settings
        existing_values = {}
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    existing_values = json.load(f)
            except (json.JSONDecodeError, IOError):
                # If the file is corrupted or can't be read, start with empty defaults
                existing_values = {}

        # Update with new values
        existing_values.update(values)

        # Save back to file
        with open(config_file, "w") as f:
            json.dump(existing_values, f, indent=2)
    except IOError:
        # If the file can't be written, just log a warning
        import logging
        logging.getLogger(__name__).warning(f"Failed to save default values to {config_file}")

# Load default values from the configuration file
_default_values = load_default_values()

# Set default provider and model from the loaded values or use empty strings as fallback
DEFAULT_PROVIDER = _default_values.get("provider", "anthropic")
DEFAULT_MODEL = _default_values.get("model", "claude-3-7-sonnet-20250219")
DEFAULT_EXPERT_MODEL = _default_values.get("expert_model", "claude-3-7-sonnet-20250219")


DEFAULT_ANTHROPIC_MODEL = "claude-3-7-sonnet-20250219"
DEFAULT_OPENAI_MODEL = "o4-mini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-pro-preview-05-06"
DEFAULT_DEEPSEEK_MODEL = "deepseek-reasoner"


DEFAULT_EXPERT_ANTHROPIC_MODEL = "claude-3-7-sonnet-20250219"
DEFAULT_EXPERT_OPENAI_MODEL = "o4-mini"
DEFAULT_EXPERT_GEMINI_MODEL = "gemini-2.5-pro-preview-05-06"
DEFAULT_EXPERT_DEEPSEEK_MODEL = "deepseek-reasoner"
DEFAULT_SHOW_COST = True
DEFAULT_MAX_COST = None
DEFAULT_MAX_TOKENS = None
DEFAULT_EXIT_AT_LIMIT = False


VALID_PROVIDERS = [
    "anthropic",
    "openai",
    "openrouter",
    "openai-compatible",
    "deepseek",
    "gemini",
    "ollama",
    "fireworks",
    "groq",
]
