import os
from dataclasses import dataclass
from typing import Optional

import pytest

from ra_aid.env import validate_environment, check_env, check_web_research_env


@dataclass
class MockArgs:
    provider: str
    expert_provider: str
    model: Optional[str] = None
    expert_model: Optional[str] = None
    research_provider: Optional[str] = None
    research_model: Optional[str] = None
    planner_provider: Optional[str] = None
    planner_model: Optional[str] = None


@pytest.fixture
def clean_env(monkeypatch):
    """Remove relevant environment variables before each test"""
    env_vars = [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_API_BASE",
        "EXPERT_ANTHROPIC_API_KEY",
        "EXPERT_OPENAI_API_KEY",
        "EXPERT_OPENROUTER_API_KEY",
        "EXPERT_OPENAI_API_BASE",
        "JINA_API_KEY",
        "ANTHROPIC_MODEL",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)


def test_anthropic_validation(clean_env, monkeypatch):
    args = MockArgs(
        provider="anthropic", expert_provider="openai", model="claude-3-haiku-20240307"
    )

    # Should fail without API key
    with pytest.raises(SystemExit):
        validate_environment(args)

    # Should pass with API key and model
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    expert_enabled, expert_missing, web_research_enabled, web_research_missing = (
        validate_environment(args)
    )
    assert not expert_enabled
    assert not web_research_enabled
    assert "JINA_API_KEY environment variable is not set" in web_research_missing


def test_openai_validation(clean_env, monkeypatch):
    args = MockArgs(provider="openai", expert_provider="openai")

    # Should fail without API key
    with pytest.raises(SystemExit):
        validate_environment(args)

    # Should pass with API key and enable expert mode with fallback
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    expert_enabled, expert_missing, web_research_enabled, web_research_missing = (
        validate_environment(args)
    )
    assert expert_enabled
    assert not expert_missing
    assert not web_research_enabled
    assert "JINA_API_KEY environment variable is not set" in web_research_missing
    assert os.environ.get("EXPERT_OPENAI_API_KEY") == "test-key"


def test_openai_compatible_validation(clean_env, monkeypatch):
    args = MockArgs(provider="openai-compatible", expert_provider="openai-compatible")

    # Should fail without API key and base URL
    with pytest.raises(SystemExit):
        validate_environment(args)

    # Should fail with only API key
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(SystemExit):
        validate_environment(args)

    # Should pass with both API key and base URL
    monkeypatch.setenv("OPENAI_API_BASE", "http://test")
    expert_enabled, expert_missing, web_research_enabled, web_research_missing = (
        validate_environment(args)
    )
    assert expert_enabled
    assert not expert_missing
    assert not web_research_enabled
    assert "JINA_API_KEY environment variable is not set" in web_research_missing
    assert os.environ.get("EXPERT_OPENAI_API_KEY") == "test-key"
    assert os.environ.get("EXPERT_OPENAI_API_BASE") == "http://test"


def test_expert_fallback(clean_env, monkeypatch):
    args = MockArgs(provider="openai", expert_provider="openai")

    # Set only base API key
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # Should enable expert mode with fallback
    expert_enabled, expert_missing, web_research_enabled, web_research_missing = (
        validate_environment(args)
    )
    assert expert_enabled
    assert not expert_missing
    assert not web_research_enabled
    assert "JINA_API_KEY environment variable is not set" in web_research_missing
    assert os.environ.get("EXPERT_OPENAI_API_KEY") == "test-key"

    # Should use explicit expert key if available
    monkeypatch.setenv("EXPERT_OPENAI_API_KEY", "expert-key")
    expert_enabled, expert_missing, web_research_enabled, web_research_missing = (
        validate_environment(args)
    )
    assert expert_enabled
    assert not expert_missing
    assert not web_research_enabled
    assert "JINA_API_KEY environment variable is not set" in web_research_missing
    assert os.environ.get("EXPERT_OPENAI_API_KEY") == "expert-key"


def test_cross_provider_fallback(clean_env, monkeypatch):
    """Test that fallback works even when providers differ"""
    args = MockArgs(
        provider="openai",
        expert_provider="anthropic",
        expert_model="claude-3-haiku-20240307",
    )

    # Set base API key for main provider and expert provider
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")

    # Should enable expert mode with fallback to ANTHROPIC base key
    expert_enabled, expert_missing, web_research_enabled, web_research_missing = (
        validate_environment(args)
    )
    assert expert_enabled
    assert not expert_missing
    assert not web_research_enabled
    assert "JINA_API_KEY environment variable is not set" in web_research_missing

    # Try with openai-compatible expert provider
    args = MockArgs(provider="anthropic", expert_provider="openai-compatible")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_API_BASE", "http://test")

    expert_enabled, expert_missing, web_research_enabled, web_research_missing = (
        validate_environment(args)
    )
    assert expert_enabled
    assert not expert_missing
    assert not web_research_enabled
    assert "JINA_API_KEY environment variable is not set" in web_research_missing
    assert os.environ.get("EXPERT_OPENAI_API_KEY") == "openai-key"
    assert os.environ.get("EXPERT_OPENAI_API_BASE") == "http://test"


def test_no_warning_on_fallback(clean_env, monkeypatch):
    """Test that no warning is issued when fallback succeeds"""
    args = MockArgs(provider="openai", expert_provider="openai")

    # Set only base API key
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # Should enable expert mode with fallback and no warnings
    expert_enabled, expert_missing, web_research_enabled, web_research_missing = (
        validate_environment(args)
    )
    assert expert_enabled
    assert not expert_missing
    assert not web_research_enabled
    assert "JINA_API_KEY environment variable is not set" in web_research_missing
    assert os.environ.get("EXPERT_OPENAI_API_KEY") == "test-key"

    # Should use explicit expert key if available
    monkeypatch.setenv("EXPERT_OPENAI_API_KEY", "expert-key")
    expert_enabled, expert_missing, web_research_enabled, web_research_missing = (
        validate_environment(args)
    )
    assert expert_enabled
    assert not expert_missing
    assert not web_research_enabled
    assert "JINA_API_KEY environment variable is not set" in web_research_missing
    assert os.environ.get("EXPERT_OPENAI_API_KEY") == "expert-key"


def test_different_providers_no_expert_key(clean_env, monkeypatch):
    """Test behavior when providers differ and only base keys are available"""
    args = MockArgs(
        provider="anthropic", expert_provider="openai", model="claude-3-haiku-20240307"
    )

    # Set only base keys
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    # Should enable expert mode and use base OPENAI key
    expert_enabled, expert_missing, web_research_enabled, web_research_missing = (
        validate_environment(args)
    )
    assert expert_enabled
    assert not expert_missing
    assert not web_research_enabled
    assert "JINA_API_KEY environment variable is not set" in web_research_missing


def test_mixed_provider_openai_compatible(clean_env, monkeypatch):
    """Test behavior with openai-compatible expert and different main provider"""
    args = MockArgs(
        provider="anthropic",
        expert_provider="openai-compatible",
        model="claude-3-haiku-20240307",
    )

    # Set all required keys and URLs
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_API_BASE", "http://test")

    # Should enable expert mode and use base openai key and URL
    expert_enabled, expert_missing, web_research_enabled, web_research_missing = (
        validate_environment(args)
    )
    assert expert_enabled
    assert not expert_missing
    assert not web_research_enabled
    assert "JINA_API_KEY environment variable is not set" in web_research_missing
    assert os.environ.get("EXPERT_OPENAI_API_KEY") == "openai-key"
    assert os.environ.get("EXPERT_OPENAI_API_BASE") == "http://test"


def test_check_web_research_env():
    """Test checking web research environment variables."""
    # Test with no key
    if "JINA_API_KEY" in os.environ:
        del os.environ["JINA_API_KEY"]
    missing = check_web_research_env()
    assert len(missing) == 1
    assert "JINA_API_KEY environment variable is not set" in missing[0]
    
    # Test with key set
    os.environ["JINA_API_KEY"] = "test_key"
    missing = check_web_research_env()
    assert len(missing) == 0
    
    # Clean up
    del os.environ["JINA_API_KEY"]


def test_check_env():
    """Test checking all environment variables."""
    # Save original env vars
    orig_env = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
        "JINA_API_KEY": os.environ.get("JINA_API_KEY")
    }
    
    try:
        # Test with no env vars set
        for key in orig_env:
            if key in os.environ:
                del os.environ[key]
                
        all_required, required_missing, all_optional, optional_missing = check_env()
        
        assert not all_required
        assert "OPENAI_API_KEY environment variable is not set" in required_missing
        assert not all_optional
        assert "ANTHROPIC_API_KEY environment variable is not set" in optional_missing
        assert "JINA_API_KEY environment variable is not set" in optional_missing
        
        # Test with only required vars
        os.environ["OPENAI_API_KEY"] = "test_key"
        all_required, required_missing, all_optional, optional_missing = check_env()
        
        assert all_required
        assert len(required_missing) == 0
        assert not all_optional
        assert "ANTHROPIC_API_KEY environment variable is not set" in optional_missing
        assert "JINA_API_KEY environment variable is not set" in optional_missing
        
        # Test with all vars
        os.environ["ANTHROPIC_API_KEY"] = "test_key"
        os.environ["JINA_API_KEY"] = "test_key"
        all_required, required_missing, all_optional, optional_missing = check_env()
        
        assert all_required
        assert len(required_missing) == 0
        assert all_optional
        assert len(optional_missing) == 0
        
    finally:
        # Restore original env vars
        for key, value in orig_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]


def test_web_research_validation(clean_env, monkeypatch):
    """Test web research validation with Jina DeepSearch."""
    args = MockArgs(provider="openai", expert_provider=None)
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")
    
    # Test without Jina API key
    expert_enabled, expert_missing, web_research_enabled, web_research_missing = (
        validate_environment(args)
    )
    assert not web_research_enabled
    assert "JINA_API_KEY environment variable is not set" in web_research_missing
    
    # Test with Jina API key
    monkeypatch.setenv("JINA_API_KEY", "test_key")
    expert_enabled, expert_missing, web_research_enabled, web_research_missing = (
        validate_environment(args)
    )
    assert web_research_enabled
    assert not web_research_missing
