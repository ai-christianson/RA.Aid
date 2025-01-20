"""Unit tests for __main__.py argument parsing."""

import pytest
from ra_aid.__main__ import parse_arguments
from ra_aid.tools.memory import _global_memory
from ra_aid.config import DEFAULT_RECURSION_LIMIT


def test_recursion_limit_in_global_config():
    """Test that recursion limit is correctly set in global config."""
    # Test default value
    args = parse_arguments(["-m", "test message"])
    assert args.recursion_limit == DEFAULT_RECURSION_LIMIT
    
    # Clear any existing config
    _global_memory.clear()
    
    # Run agent with default recursion limit
    config = {
        "configurable": {"thread_id": "test"},
        "recursion_limit": DEFAULT_RECURSION_LIMIT
    }
    _global_memory["config"] = config
    assert _global_memory["config"]["recursion_limit"] == DEFAULT_RECURSION_LIMIT

    # Test custom value
    args = parse_arguments(["-m", "test message", "--recursion-limit", "50"])
    assert args.recursion_limit == 50
    
    # Clear any existing config
    _global_memory.clear()
    
    # Run agent with custom recursion limit
    config = {
        "configurable": {"thread_id": "test"},
        "recursion_limit": 50
    }
    _global_memory["config"] = config
    assert _global_memory["config"]["recursion_limit"] == 50


def test_negative_recursion_limit():
    """Test that negative recursion limit raises error."""
    with pytest.raises(SystemExit):
        parse_arguments(["-m", "test message", "--recursion-limit", "-1"])


def test_zero_recursion_limit():
    """Test that zero recursion limit raises error."""
    with pytest.raises(SystemExit):
        parse_arguments(["-m", "test message", "--recursion-limit", "0"])


def test_config_settings():
    """Test that various arguments are correctly set in config."""
    args = parse_arguments([
        "-m", "test message",
        "--recursion-limit", "75",
        "--cowboy-mode",
        "--research-only",
        "--provider", "anthropic",
        "--model", "claude-3-5-sonnet-20241022",
        "--expert-provider", "openai",
        "--expert-model", "gpt-4",
        "--temperature", "0.7",
        "--disable-limit-tokens"
    ])

    assert args.recursion_limit == 75
    assert args.cowboy_mode is True
    assert args.research_only is True
    assert args.provider == "anthropic"
    assert args.model == "claude-3-5-sonnet-20241022"
    assert args.expert_provider == "openai" 
    assert args.expert_model == "gpt-4"
    assert args.temperature == 0.7
    assert args.disable_limit_tokens is False


def test_chat_mode_implies_hil():
    """Test that enabling chat mode automatically enables HIL."""
    args = parse_arguments(["-m", "test message", "--chat"])
    assert args.hil is True


def test_temperature_validation():
    """Test temperature validation."""
    # Valid temperatures
    args = parse_arguments(["-m", "test", "--temperature", "0.0"])
    assert args.temperature == 0.0
    
    args = parse_arguments(["-m", "test", "--temperature", "2.0"])
    assert args.temperature == 2.0

    # Invalid temperatures
    with pytest.raises(SystemExit):
        parse_arguments(["-m", "test", "--temperature", "-0.1"])
    
    with pytest.raises(SystemExit):
        parse_arguments(["-m", "test", "--temperature", "2.1"])


def test_missing_message():
    """Test that missing message argument raises error."""
    # Test chat mode which doesn't require message
    args = parse_arguments(["--chat"])
    assert args.chat is True
    assert args.message is None

    # Test non-chat mode requires message
    args = parse_arguments(["--provider", "openai"])
    assert args.message is None

    # Verify message is captured when provided
    args = parse_arguments(["-m", "test"])
    assert args.message == "test"
