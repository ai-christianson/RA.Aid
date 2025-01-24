"""Integration tests for provider validation and environment handling."""

import os
import pytest
from dataclasses import dataclass
from typing import Optional

from ra_aid.env import validate_environment
from ra_aid.provider_strategy import (
    ProviderFactory,
    ValidationResult,
    AnthropicStrategy,
    OpenAIStrategy,
    OpenAICompatibleStrategy,
    OpenRouterStrategy,
    GeminiStrategy,
)


@dataclass
class MockArgs:
    """Mock arguments for testing."""

    provider: str
    expert_provider: Optional[str] = None
    model: Optional[str] = None
    expert_model: Optional[str] = None
    research_provider: Optional[str] = None
    research_model: Optional[str] = None
    planner_provider: Optional[str] = None
    planner_model: Optional[str] = None


@pytest.fixture
def clean_env():
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
        "GEMINI_API_KEY",
        "EXPERT_GEMINI_API_KEY",
        "GEMINI_MODEL",
    ]

    # Store original values
    original_values = {}
    for var in env_vars:
        original_values[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]

    yield

    # Restore original values
    for var, value in original_values.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]


def test_provider_validation_respects_cli_args(clean_env):
    """Test that provider validation respects CLI args over defaults."""
    # Set up environment with only OpenAI credentials
    os.environ["OPENAI_API_KEY"] = "test-key"

    # Should succeed with OpenAI provider
    args = MockArgs(provider="openai")
    expert_enabled, expert_missing, web_enabled, web_missing = validate_environment(
        args
    )
    assert not expert_missing

    # Should fail with Anthropic provider even though it's first alphabetically
    args = MockArgs(provider="anthropic", model="claude-3-haiku-20240307")
    with pytest.raises(SystemExit):
        validate_environment(args)


def test_expert_provider_fallback(clean_env):
    """Test expert provider falls back to main provider keys."""
    os.environ["OPENAI_API_KEY"] = "test-key"
    args = MockArgs(provider="openai", expert_provider="openai")

    expert_enabled, expert_missing, _, _ = validate_environment(args)
    assert expert_enabled
    assert not expert_missing


def test_openai_compatible_base_url(clean_env):
    """Test OpenAI-compatible provider requires base URL."""
    os.environ["OPENAI_API_KEY"] = "test-key"
    args = MockArgs(provider="openai-compatible")

    with pytest.raises(SystemExit):
        validate_environment(args)

    os.environ["OPENAI_API_BASE"] = "http://test"
    expert_enabled, expert_missing, _, _ = validate_environment(args)
    assert not expert_missing


def test_expert_provider_separate_keys(clean_env):
    """Test expert provider can use separate keys."""
    os.environ["OPENAI_API_KEY"] = "main-key"
    os.environ["EXPERT_OPENAI_API_KEY"] = "expert-key"

    args = MockArgs(provider="openai", expert_provider="openai")
    expert_enabled, expert_missing, _, _ = validate_environment(args)
    assert expert_enabled
    assert not expert_missing


def test_web_research_independent(clean_env):
    """Test web research validation is independent of provider."""
    os.environ["OPENAI_API_KEY"] = "test-key"
    args = MockArgs(provider="openai")

    # Without Tavily key
    expert_enabled, expert_missing, web_enabled, web_missing = validate_environment(
        args
    )
    assert not web_enabled
    assert web_missing

    # With Tavily key
    os.environ["TAVILY_API_KEY"] = "test-key"
    expert_enabled, expert_missing, web_enabled, web_missing = validate_environment(
        args
    )
    assert web_enabled
    assert not web_missing


def test_provider_factory_unknown_provider(clean_env):
    """Test provider factory handles unknown providers."""
    strategy = ProviderFactory.create("unknown")
    assert strategy is None

    args = MockArgs(provider="unknown")
    with pytest.raises(SystemExit):
        validate_environment(args)


def test_provider_strategy_validation(clean_env):
    """Test individual provider strategies."""
    # Test Anthropic strategy
    strategy = AnthropicStrategy()
    result = strategy.validate()
    assert not result.valid
    assert "ANTHROPIC_API_KEY environment variable is not set" in result.missing_vars

    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    os.environ["ANTHROPIC_MODEL"] = "claude-3-haiku-20240307"
    result = strategy.validate()
    assert result.valid
    assert not result.missing_vars

    # Test OpenAI strategy
    strategy = OpenAIStrategy()
    result = strategy.validate()
    assert not result.valid
    assert "OPENAI_API_KEY environment variable is not set" in result.missing_vars

    os.environ["OPENAI_API_KEY"] = "test-key"
    result = strategy.validate()
    assert result.valid
    assert not result.missing_vars


def test_missing_provider_arg():
    """Test handling of missing provider argument."""
    with pytest.raises(SystemExit):
        validate_environment(None)

    with pytest.raises(SystemExit):
        validate_environment(MockArgs(provider=None))


def test_empty_provider_arg():
    """Test handling of empty provider argument."""
    with pytest.raises(SystemExit):
        validate_environment(MockArgs(provider=""))


def test_incomplete_openai_compatible_config(clean_env):
    """Test OpenAI-compatible provider with incomplete configuration."""
    strategy = OpenAICompatibleStrategy()

    # No configuration
    result = strategy.validate()
    assert not result.valid
    assert "OPENAI_API_KEY environment variable is not set" in result.missing_vars
    assert "OPENAI_API_BASE environment variable is not set" in result.missing_vars

    # Only API key
    os.environ["OPENAI_API_KEY"] = "test-key"
    result = strategy.validate()
    assert not result.valid
    assert "OPENAI_API_BASE environment variable is not set" in result.missing_vars

    # Only base URL
    os.environ.pop("OPENAI_API_KEY")
    os.environ["OPENAI_API_BASE"] = "http://test"
    result = strategy.validate()
    assert not result.valid
    assert "OPENAI_API_KEY environment variable is not set" in result.missing_vars


def test_incomplete_gemini_config(clean_env):
    """Test Gemini provider with incomplete configuration."""
    strategy = GeminiStrategy()

    # No configuration
    result = strategy.validate()
    assert not result.valid
    assert "GEMINI_API_KEY environment variable is not set" in result.missing_vars

    # Valid API key
    os.environ["GEMINI_API_KEY"] = "test-key"
    result = strategy.validate()
    assert result.valid
    assert not result.missing_vars


def test_incomplete_expert_config(clean_env):
    """Test expert provider with incomplete configuration."""
    # Set main provider but not expert
    os.environ["OPENAI_API_KEY"] = "test-key"
    args = MockArgs(provider="openai", expert_provider="openai-compatible")

    expert_enabled, expert_missing, _, _ = validate_environment(args)
    assert not expert_enabled
    assert len(expert_missing) == 1
    assert "EXPERT_OPENAI_API_BASE" in expert_missing[0]

    # Set expert key but not base URL
    os.environ["EXPERT_OPENAI_API_KEY"] = "test-key"
    expert_enabled, expert_missing, _, _ = validate_environment(args)
    assert not expert_enabled
    assert len(expert_missing) == 1
    assert "EXPERT_OPENAI_API_BASE" in expert_missing[0]


def test_empty_environment_variables(clean_env):
    """Test handling of empty environment variables."""
    # Empty API key
    os.environ["OPENAI_API_KEY"] = ""
    args = MockArgs(provider="openai")
    with pytest.raises(SystemExit):
        validate_environment(args)

    # Empty base URL
    os.environ["OPENAI_API_KEY"] = "test-key"
    os.environ["OPENAI_API_BASE"] = ""
    args = MockArgs(provider="openai-compatible")
    with pytest.raises(SystemExit):
        validate_environment(args)


def test_openrouter_validation(clean_env):
    """Test OpenRouter provider validation."""
    strategy = OpenRouterStrategy()

    # No API key
    result = strategy.validate()
    assert not result.valid
    assert "OPENROUTER_API_KEY environment variable is not set" in result.missing_vars

    # Empty API key
    os.environ["OPENROUTER_API_KEY"] = ""
    result = strategy.validate()
    assert not result.valid
    assert "OPENROUTER_API_KEY environment variable is not set" in result.missing_vars

    # Valid API key
    os.environ["OPENROUTER_API_KEY"] = "test-key"
    result = strategy.validate()
    assert result.valid
    assert not result.missing_vars


def test_multiple_expert_providers(clean_env):
    """Test validation with multiple expert providers."""
    os.environ["OPENAI_API_KEY"] = "test-key"
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    os.environ["ANTHROPIC_MODEL"] = "claude-3-haiku-20240307"

    # First expert provider valid, second invalid
    args = MockArgs(
        provider="openai",
        expert_provider="anthropic",
        expert_model="claude-3-haiku-20240307",
    )
    expert_enabled, expert_missing, _, _ = validate_environment(args)
    assert expert_enabled
    assert not expert_missing

    # Switch to invalid provider
    args = MockArgs(provider="openai", expert_provider="openai-compatible")
    expert_enabled, expert_missing, _, _ = validate_environment(args)
    assert not expert_enabled
    assert expert_missing
