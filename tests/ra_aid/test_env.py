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

@pytest.fixture
def clean_env(monkeypatch):
    """Remove relevant environment variables before each test"""
    env_vars = [
        'ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'OPENROUTER_API_KEY',
        'OPENAI_API_BASE', 'EXPERT_ANTHROPIC_API_KEY', 'EXPERT_OPENAI_API_KEY',
        'EXPERT_OPENROUTER_API_KEY', 'EXPERT_OPENAI_API_BASE'
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)

def test_anthropic_validation(clean_env, monkeypatch):
    args = MockArgs(provider="anthropic", expert_provider="openai")
    
    # Should fail without API key
    with pytest.raises(SystemExit):
        validate_environment(args)
    
    # Should pass with API key
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key')
    expert_enabled, missing = validate_environment(args)
    assert not expert_enabled
    assert 'EXPERT_OPENAI_API_KEY environment variable is not set' in missing

def test_openai_validation(clean_env, monkeypatch):
    args = MockArgs(provider="openai", expert_provider="openai")
    
    # Should fail without API key
    with pytest.raises(SystemExit):
        validate_environment(args)
    
    # Should pass with API key and enable expert mode with fallback
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    expert_enabled, missing = validate_environment(args)
    assert expert_enabled
    assert not missing
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
    expert_enabled, missing = validate_environment(args)
    assert expert_enabled
    assert not missing
    assert os.environ.get('EXPERT_OPENAI_API_KEY') == 'test-key'
    assert os.environ.get('EXPERT_OPENAI_API_BASE') == 'http://test'

def test_expert_fallback(clean_env, monkeypatch):
    args = MockArgs(provider="openai", expert_provider="openai")
    
    # Set only base API key
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    
    # Should enable expert mode with fallback
    expert_enabled, missing = validate_environment(args)
    assert expert_enabled
    assert not missing
    assert os.environ.get('EXPERT_OPENAI_API_KEY') == 'test-key'
    
    # Should use explicit expert key if available
    monkeypatch.setenv('EXPERT_OPENAI_API_KEY', 'expert-key')
    expert_enabled, missing = validate_environment(args)
    assert expert_enabled
    assert not missing
    assert os.environ.get('EXPERT_OPENAI_API_KEY') == 'expert-key'
