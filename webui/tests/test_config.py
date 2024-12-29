import pytest
import os
from webui.config import WebUIConfig, load_environment_status

@pytest.fixture(autouse=True)
def setup_env():
    """Setup test environment variables"""
    # Store original environment
    original_env = {}
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"]:
        if key in os.environ:
            original_env[key] = os.environ[key]
            
    # Set test environment
    os.environ["OPENAI_API_KEY"] = "test_openai"
    os.environ["ANTHROPIC_API_KEY"] = "test_anthropic"
    os.environ["OPENROUTER_API_KEY"] = "test_openrouter"
    
    yield
    
    # Restore original environment
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"]:
        if key in original_env:
            os.environ[key] = original_env[key]
        elif key in os.environ:
            del os.environ[key]

def test_webui_config_defaults():
    """Test WebUIConfig default values"""
    config = WebUIConfig()
    assert config.provider == "openai"
    assert config.model == "gpt-4"
    assert config.research_only is False
    assert config.cowboy_mode is False
    assert config.hil is True
    assert config.web_research_enabled is True

def test_webui_config_custom():
    """Test WebUIConfig with custom values"""
    config = WebUIConfig(
        provider="anthropic",
        model="claude-2.1",
        research_only=True,
        cowboy_mode=True,
        hil=False,
        web_research_enabled=False
    )
    assert config.provider == "anthropic"
    assert config.model == "claude-2.1"
    assert config.research_only is True
    assert config.cowboy_mode is True
    assert config.hil is False
    assert config.web_research_enabled is False

def test_webui_config_validation():
    """Test WebUIConfig validation"""
    # Test invalid provider
    with pytest.raises(ValueError, match="Invalid provider: invalid"):
        WebUIConfig(provider="invalid")
        
    # Test missing API key
    del os.environ["OPENAI_API_KEY"]
    with pytest.raises(ValueError, match="API key not set for provider: openai"):
        WebUIConfig()

def test_webui_config_str():
    """Test WebUIConfig string representation"""
    config = WebUIConfig()
    config_str = str(config)
    assert "provider=openai" in config_str
    assert "model=gpt-4" in config_str
    assert "research_only=False" in config_str
    assert "cowboy_mode=False" in config_str
    assert "hil=True" in config_str
    assert "web_research_enabled=True" in config_str

def test_load_environment_status():
    """Test environment status loading"""
    status = load_environment_status()
    assert status["openai"] is True
    assert status["anthropic"] is True
    assert status["openrouter"] is True
    
    # Test missing API key
    del os.environ["OPENAI_API_KEY"]
    status = load_environment_status()
    assert status["openai"] is False
