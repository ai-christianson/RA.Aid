"""Utilities for detecting and working with specific model types."""



from typing import Any, Dict


def is_claude_37(model: str) -> bool:
    """Check if the model is a Claude 3.7 model.
    
    Args:
        model: The model name to check
        
    Returns:
        bool: True if the model is a Claude 3.7 model, False otherwise
    """
    patterns = ["claude-3.7", "claude3.7", "claude-3-7"]
    return any(pattern in model for pattern in patterns)

def model_name_has_claude(model_name: str) -> bool:
    """Check if a model name contains 'claude'.

    Args:
        model_name: The model name to check

    Returns:
        bool: True if the model name contains 'claude'
    """
    return model_name and "claude" in model_name.lower()


def is_anthropic_claude(config: Dict[str, Any]) -> bool:
    """Check if the provider and model name indicate an Anthropic Claude model.

    Args:
        config: Configuration dictionary containing provider and model information

    Returns:
        bool: True if this is an Anthropic Claude model
    """
    # For backwards compatibility, allow passing of config directly
    provider = config.get("provider", "")
    model_name = config.get("model", "")
    result = (
        provider.lower() == "anthropic"
        and model_name
        and "claude" in model_name.lower()
    ) or (
        provider.lower() == "openrouter"
        and model_name.lower().startswith("anthropic/claude-")
    )
    return result
