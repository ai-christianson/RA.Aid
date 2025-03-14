"""Unit tests for model_detection.py."""

import pytest
from ra_aid.model_detection import is_anthropic_claude, is_claude_37


def test_is_anthropic_claude():
    """Test is_anthropic_claude function with various configurations."""
    # Test Anthropic provider cases
    assert is_anthropic_claude({"provider": "anthropic", "model": "claude-2"})
    assert is_anthropic_claude({"provider": "ANTHROPIC", "model": "claude-instant"})
    assert not is_anthropic_claude({"provider": "anthropic", "model": "gpt-4"})

    # Test OpenRouter provider cases
    assert is_anthropic_claude(
        {"provider": "openrouter", "model": "anthropic/claude-2"}
    )
    assert is_anthropic_claude(
        {"provider": "openrouter", "model": "anthropic/claude-instant"}
    )
    assert not is_anthropic_claude({"provider": "openrouter", "model": "openai/gpt-4"})

    # Test edge cases
    assert not is_anthropic_claude({})  # Empty config
    assert not is_anthropic_claude({"provider": "anthropic"})  # Missing model
    assert not is_anthropic_claude({"model": "claude-2"})  # Missing provider
    assert not is_anthropic_claude(
        {"provider": "other", "model": "claude-2"}
    )  # Wrong provider


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
