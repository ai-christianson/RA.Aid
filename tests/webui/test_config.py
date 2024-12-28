import pytest
from unittest.mock import patch
from webui.config import load_environment_status, WebUIConfig

@pytest.fixture
def clean_env():
    """Clean environment variables"""
    with patch.dict('os.environ', {}, clear=True):
        yield

def test_load_environment_status_no_keys(clean_env):
    """Test environment status with no API keys configured"""
    status = load_environment_status()
    assert status == {
        'anthropic': False,
        'openai': False,
        'openrouter': False,
        'tavily': False
    }

def test_load_environment_status_with_keys(clean_env, monkeypatch):
    """Test environment status with API keys configured"""
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key')
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    monkeypatch.setenv('TAVILY_API_KEY', 'test-key')

    status = load_environment_status()
    assert status == {
        'anthropic': True,
        'openai': True,
        'openrouter': False,
        'tavily': True
    }

def test_webui_config_defaults():
    """Test WebUIConfig default values"""
    config = WebUIConfig(
        provider='openai',
        model='gpt-4',
        research_only=False,
        cowboy_mode=False,
        hil=True,
        web_research_enabled=True
    )
    assert config.provider == 'openai'
    assert config.model == 'gpt-4'
    assert config.research_only is False
    assert config.cowboy_mode is False
    assert config.hil is True
    assert config.web_research_enabled is True

def test_webui_config_custom_values():
    """Test WebUIConfig with custom values"""
    config = WebUIConfig(
        provider='anthropic',
        model='claude-3',
        research_only=True,
        cowboy_mode=True,
        hil=False,
        web_research_enabled=False
    )
    assert config.provider == 'anthropic'
    assert config.model == 'claude-3'
    assert config.research_only is True
    assert config.cowboy_mode is True
    assert config.hil is False
    assert config.web_research_enabled is False

def test_webui_config_validation():
    """Test WebUIConfig validation"""
    config = WebUIConfig(
        provider='invalid_provider',
        model='test-model',
        research_only=False,
        cowboy_mode=False,
        hil=True,
        web_research_enabled=True
    )
    assert config.provider == 'invalid_provider'
    assert config.model == 'test-model'
    assert config.research_only is False
    assert config.cowboy_mode is False
    assert config.hil is True
    assert config.web_research_enabled is True

    # Test dictionary conversion
    config_dict = config.to_dict()
    assert config_dict == {
        'provider': 'invalid_provider',
        'model': 'test-model',
        'research_only': False,
        'cowboy_mode': False,
        'hil': True,
        'web_research_enabled': True
    } 