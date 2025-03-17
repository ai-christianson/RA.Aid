"""Unit tests for model_detection.py."""

import pytest
from ra_aid.model_detection import is_claude_37


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
