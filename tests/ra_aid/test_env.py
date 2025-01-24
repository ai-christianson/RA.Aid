import os
import pytest
from dataclasses import dataclass
from typing import Optional

from ra_aid.env import validate_environment

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
        'ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'OPENROUTER_API_KEY',
        'OPENAI_API_BASE', 'EXPERT_ANTHROPIC_API_KEY', 'EXPERT_OPENAI_API_KEY',
        'EXPERT_OPENROUTER_API_KEY', 'EXPERT_OPENAI_API_BASE', 'TAVILY_API_KEY', 'ANTHROPIC_MODEL'
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)

def test_anthropic_validation(clean_env, monkeypatch):
    args = MockArgs(provider="anthropic", expert_provider="openai", model="claude-3-haiku-20240307")

    # Should fail without API key
    with pytest.raises(SystemExit):
        validate_environment(args)

    # Should pass with API key and model
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key')
    expert_enabled, expert_missing, web_research_enabled, web_research_missing = validate_environment(args)
    assert not expert_enabled
    assert not web_research_enabled
    assert 'TAVILY_API_KEY environment variable is not set' in web_research_missing

def test_openai_validation(clean_env, monkeypatch):
    args = MockArgs(provider="openai", expert_provider="openai")

    # Should fail without API key
    with pytest.raises(SystemExit):
        validate_environment(args)

    # Should pass with API key and enable expert mode with fallback
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    expert_enabled, expert_missing, web_research_enabled, web_research_missing = validate_environment(args)
    assert expert_enabled
    assert not expert_missing
    assert not web_research_enabled
    assert 'TAVILY_API_KEY environment variable is not set' in web_research_missing
    assert os.environ.get('EXPERT_OPENAI_API_KEY') == 'test-key'

def test_openai_compatible_validation(clean_env, monkeypatch):
    args = MockArgs(provider="openai-compatible", expert_provider="openai-compatible")

    # Should fail without API key and base URL
    with pytest.raises(SystemExit):
        validate_environment(args)

    # Should fail with only API key
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    with pytest.raises(SystemExit):
        validate_environment(args)

    # Should pass with both API key and base URL
    monkeypatch.setenv('OPENAI_API_BASE', 'http://test')
    expert_enabled, expert_missing, web_research_enabled, web_research_missing = validate_environment(args)
    assert expert_enabled
    assert not expert_missing
    assert not web_research_enabled
    assert 'TAVILY_API_KEY environment variable is not set' in web_research_missing
    assert os.environ.get('EXPERT_OPENAI_API_KEY') == 'test-key'
    assert os.environ.get('EXPERT_OPENAI_API_BASE') == 'http://test'

def test_expert_fallback(clean_env, monkeypatch):
    args = MockArgs(provider="openai", expert_provider="openai")

    # Set only base API key
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')

    # Should enable expert mode with fallback
    expert_enabled, expert_missing, web_research_enabled, web_research_missing = validate_environment(args)
    assert expert_enabled
    assert not expert_missing
    assert not web_research_enabled
    assert 'TAVILY_API_KEY environment variable is not set' in web_research_missing
    assert os.environ.get('EXPERT_OPENAI_API_KEY') == 'test-key'

    # Should use explicit expert key if available
    monkeypatch.setenv('EXPERT_OPENAI_API_KEY', 'expert-key')
    expert_enabled, expert_missing, web_research_enabled, web_research_missing = validate_environment(args)
    assert expert_enabled
    assert not expert_missing
    assert not web_research_enabled
    assert 'TAVILY_API_KEY environment variable is not set' in web_research_missing
    assert os.environ.get('EXPERT_OPENAI_API_KEY') == 'expert-key'

def test_cross_provider_fallback(clean_env, monkeypatch):
    """Test that fallback works even when providers differ"""
    args = MockArgs(provider="openai", expert_provider="anthropic", expert_model="claude-3-haiku-20240307")

    # Set base API key for main provider and expert provider
    monkeypatch.setenv('OPENAI_API_KEY', 'openai-key')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'anthropic-key')
    monkeypatch.setenv('ANTHROPIC_MODEL', 'claude-3-haiku-20240307')

    # Should enable expert mode with fallback to ANTHROPIC base key
    expert_enabled, expert_missing, web_research_enabled, web_research_missing = validate_environment(args)
    assert expert_enabled
    assert not expert_missing
    assert not web_research_enabled
    assert 'TAVILY_API_KEY environment variable is not set' in web_research_missing

    # Try with openai-compatible expert provider
    args = MockArgs(provider="anthropic", expert_provider="openai-compatible")
    monkeypatch.setenv('OPENAI_API_KEY', 'openai-key')
    monkeypatch.setenv('OPENAI_API_BASE', 'http://test')

    expert_enabled, expert_missing, web_research_enabled, web_research_missing = validate_environment(args)
    assert expert_enabled
    assert not expert_missing
    assert not web_research_enabled
    assert 'TAVILY_API_KEY environment variable is not set' in web_research_missing
    assert os.environ.get('EXPERT_OPENAI_API_KEY') == 'openai-key'
    assert os.environ.get('EXPERT_OPENAI_API_BASE') == 'http://test'

def test_no_warning_on_fallback(clean_env, monkeypatch):
    """Test that no warning is issued when fallback succeeds"""
    args = MockArgs(provider="openai", expert_provider="openai")

    # Set only base API key
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')

    # Should enable expert mode with fallback and no warnings
    expert_enabled, expert_missing, web_research_enabled, web_research_missing = validate_environment(args)
    assert expert_enabled
    assert not expert_missing
    assert not web_research_enabled
    assert 'TAVILY_API_KEY environment variable is not set' in web_research_missing
    assert os.environ.get('EXPERT_OPENAI_API_KEY') == 'test-key'

    # Should use explicit expert key if available
    monkeypatch.setenv('EXPERT_OPENAI_API_KEY', 'expert-key')
    expert_enabled, expert_missing, web_research_enabled, web_research_missing = validate_environment(args)
    assert expert_enabled
    assert not expert_missing
    assert not web_research_enabled
    assert 'TAVILY_API_KEY environment variable is not set' in web_research_missing
    assert os.environ.get('EXPERT_OPENAI_API_KEY') == 'expert-key'

def test_different_providers_no_expert_key(clean_env, monkeypatch):
    """Test behavior when providers differ and only base keys are available"""
    args = MockArgs(provider="anthropic", expert_provider="openai", model="claude-3-haiku-20240307")

    # Set only base keys
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'anthropic-key')
    monkeypatch.setenv('OPENAI_API_KEY', 'openai-key')

    # Should enable expert mode and use base OPENAI key
    expert_enabled, expert_missing, web_research_enabled, web_research_missing = validate_environment(args)
    assert expert_enabled
    assert not expert_missing
    assert not web_research_enabled
    assert 'TAVILY_API_KEY environment variable is not set' in web_research_missing


def test_mixed_provider_openai_compatible(clean_env, monkeypatch):
    """Test behavior with openai-compatible expert and different main provider"""
    args = MockArgs(provider="anthropic", expert_provider="openai-compatible", model="claude-3-haiku-20240307")

    # Set all required keys and URLs
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'anthropic-key')
    monkeypatch.setenv('OPENAI_API_KEY', 'openai-key')
    monkeypatch.setenv('OPENAI_API_BASE', 'http://test')

    # Should enable expert mode and use base openai key and URL
    expert_enabled, expert_missing, web_research_enabled, web_research_missing = validate_environment(args)
    assert expert_enabled
    assert not expert_missing
    assert not web_research_enabled
    assert 'TAVILY_API_KEY environment variable is not set' in web_research_missing
    assert os.environ.get('EXPERT_OPENAI_API_KEY') == 'openai-key'
    assert os.environ.get('EXPERT_OPENAI_API_BASE') == 'http://test'
