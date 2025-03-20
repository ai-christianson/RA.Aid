"""Unit tests for model_detection.py."""

import pytest
from ra_aid.model_detection import is_claude_37, normalize_model_name


def test_normalize_model_name():
    """Test normalize_model_name function with various model names."""
    # Test provider prefix removal
    assert normalize_model_name("anthropic/claude-3.7") == "claude-3.7"
    assert normalize_model_name("google/gemini-2.0-pro") == "gemini-2.0-pro"
    assert normalize_model_name("openai/gpt-4") == "gpt-4"
    
    # Test version suffix removal
    assert normalize_model_name("gemini-2.0-pro-exp-02-05:free") == "gemini-2.0-pro-exp-02-05"
    assert normalize_model_name("claude-3.5-sonnet:v1") == "claude-3.5-sonnet"
    
    # Test both prefix and suffix
    assert normalize_model_name("google/gemini-2.0-pro:free") == "gemini-2.0-pro"
    
    # Test no changes needed
    assert normalize_model_name("claude-3.7") == "claude-3.7"
    assert normalize_model_name("gpt-4") == "gpt-4"


def test_is_claude_37():
    """Test is_claude_37 function with various model names."""
    # Test positive cases
    assert is_claude_37("claude-3.7")
    assert is_claude_37("claude3.7")
    assert is_claude_37("claude-3-7")
    assert is_claude_37("anthropic/claude-3.7")
    assert is_claude_37("anthropic/claude3.7")
    assert is_claude_37("anthropic/claude-3-7")
    assert is_claude_37("claude-3.7-sonnet")
    assert is_claude_37("claude3.7-haiku")
    
    # Test negative cases
    assert not is_claude_37("claude-3")
    assert not is_claude_37("claude-3.5")
    assert not is_claude_37("claude3.5")
    assert not is_claude_37("claude-3-5")
    assert not is_claude_37("gpt-4")
    assert not is_claude_37("")
