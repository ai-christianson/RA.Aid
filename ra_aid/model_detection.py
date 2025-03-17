"""Utilities for detecting and working with specific model types."""

from typing import Optional, Dict, Any


def is_claude_37(model: str) -> bool:
    """Check if the model is a Claude 3.7 model.
    
    Args:
        model: The model name to check
        
    Returns:
        bool: True if the model is a Claude 3.7 model, False otherwise
    """
    patterns = ["claude-3.7", "claude3.7", "claude-3-7"]
    return any(pattern in model for pattern in patterns)
