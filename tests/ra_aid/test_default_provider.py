"""Tests for default provider and model configuration."""

import os
import pytest
from dataclasses import dataclass
from typing import Optional

from ra_aid.env import validate_environment
from ra_aid.__main__ import parse_arguments

@dataclass
class MockArgs:
    """Mock arguments for testing."""
    provider: str
    expert_provider: Optional[str] = None
    model: Optional[str] = None
    expert_model: Optional[str] = None
    message: Optional[str] = None
    research_only: bool = False
    chat: bool = False

@pytest.fixture
def clean_env(monkeypatch):
    """Remove all provider-related environment variables."""
    env_vars = [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_API_BASE",
        "EXPERT_ANTHROPIC_API_KEY",
        "EXPERT_OPENAI_API_KEY",
        "EXPERT_OPENAI_API_BASE",
        "TAVILY_API_KEY",
        "ANTHROPIC_MODEL",
        "RA_AID_DEFAULT_PROVIDER",
        "RA_AID_DEFAULT_MODEL"
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)
    yield

def test_default_anthropic_provider(clean_env, monkeypatch):
    """Test that Anthropic is the default provider when no environment variables are set."""
    args = parse_arguments(["-m", "test message"])
    assert args.provider == "anthropic"
    assert args.model == "claude-3-5-sonnet-20241022"

def test_default_provider_from_env(clean_env, monkeypatch):
    """Test that RA_AID_DEFAULT_PROVIDER is used when set."""
    monkeypatch.setenv("RA_AID_DEFAULT_PROVIDER", "openai")
    monkeypatch.setenv("RA_AID_DEFAULT_MODEL", "gpt-4")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    args = parse_arguments(["-m", "test message"])
    assert args.provider == "openai"
    assert args.model == "gpt-4"

def test_default_model_from_env(clean_env, monkeypatch):
    """Test that RA_AID_DEFAULT_MODEL is used when set."""
    monkeypatch.setenv("RA_AID_DEFAULT_PROVIDER", "openai")
    monkeypatch.setenv("RA_AID_DEFAULT_MODEL", "gpt-4")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    args = parse_arguments(["-m", "test message"])
    assert args.provider == "openai"
    assert args.model == "gpt-4"

def test_cli_overrides_default_provider(clean_env, monkeypatch):
    """Test that CLI arguments override environment defaults."""
    monkeypatch.setenv("RA_AID_DEFAULT_PROVIDER", "openai")
    monkeypatch.setenv("RA_AID_DEFAULT_MODEL", "gpt-4")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    
    args = parse_arguments(["-m", "test message", "--provider", "anthropic"])
    assert args.provider == "anthropic"
    assert args.model == "claude-3-5-sonnet-20241022"

def test_cli_overrides_default_model(clean_env, monkeypatch):
    """Test that CLI model argument overrides environment default."""
    monkeypatch.setenv("RA_AID_DEFAULT_PROVIDER", "openai")
    monkeypatch.setenv("RA_AID_DEFAULT_MODEL", "gpt-4")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    
    args = parse_arguments(["-m", "test message", "--model", "gpt-3.5-turbo"])
    assert args.provider == "openai"
    assert args.model == "gpt-3.5-turbo"

def test_invalid_default_provider(clean_env, monkeypatch):
    """Test that invalid default provider raises error."""
    monkeypatch.setenv("RA_AID_DEFAULT_PROVIDER", "invalid")
    with pytest.raises(SystemExit):
        parse_arguments(["-m", "test message"])

def test_research_only_ignores_provider(clean_env, monkeypatch):
    """Test that --research-only ignores provider validation."""
    monkeypatch.setenv("RA_AID_DEFAULT_PROVIDER", "openai")
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    args = parse_arguments(["-m", "test message", "--research-only"])
    assert args.research_only is True
    expert_enabled, expert_missing, web_research_enabled, web_research_missing = validate_environment(args)
    assert web_research_enabled
    assert not web_research_missing
