import os
import pytest
from unittest.mock import patch, Mock
from langchain_openai.chat_models import ChatOpenAI
from langchain_anthropic.chat_models import ChatAnthropic
from dataclasses import dataclass

from ra_aid.env import validate_environment
from ra_aid.llm import initialize_llm, initialize_expert_llm

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

@pytest.fixture
def mock_openai():
    """
    Mock ChatOpenAI class for testing OpenAI provider initialization.
    Prevents actual API calls during testing.
    """
    with patch('ra_aid.llm.ChatOpenAI') as mock:
        mock.return_value = Mock(spec=ChatOpenAI)
        yield mock

def test_initialize_expert_defaults(clean_env, mock_openai, monkeypatch):
    """Test expert LLM initialization with default parameters."""
    monkeypatch.setenv("EXPERT_OPENAI_API_KEY", "test-key")
    llm = initialize_expert_llm()
    
    mock_openai.assert_called_once_with(
        api_key="test-key",
        model="o1-preview"
    )

def test_initialize_expert_openai_custom(clean_env, mock_openai, monkeypatch):
    """Test expert OpenAI initialization with custom parameters."""
    monkeypatch.setenv("EXPERT_OPENAI_API_KEY", "test-key")
    llm = initialize_expert_llm("openai", "gpt-4-preview")
    
    mock_openai.assert_called_once_with(
        api_key="test-key",
        model="gpt-4-preview"
    )

def test_initialize_expert_anthropic(clean_env, mock_anthropic, monkeypatch):
    """Test expert Anthropic initialization."""
    monkeypatch.setenv("EXPERT_ANTHROPIC_API_KEY", "test-key")
    llm = initialize_expert_llm("anthropic", "claude-3")
    
    mock_anthropic.assert_called_once_with(
        api_key="test-key",
        model_name="claude-3"
    )

def test_initialize_expert_openrouter(clean_env, mock_openai, monkeypatch):
    """Test expert OpenRouter initialization."""
    monkeypatch.setenv("EXPERT_OPENROUTER_API_KEY", "test-key")
    llm = initialize_expert_llm("openrouter", "models/mistral-large")
    
    mock_openai.assert_called_once_with(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="models/mistral-large"
    )

def test_initialize_expert_openai_compatible(clean_env, mock_openai, monkeypatch):
    """Test expert OpenAI-compatible initialization."""
    monkeypatch.setenv("EXPERT_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EXPERT_OPENAI_API_BASE", "http://test-url")
    llm = initialize_expert_llm("openai-compatible", "local-model")
    
    mock_openai.assert_called_once_with(
        api_key="test-key",
        base_url="http://test-url",
        model="local-model"
    )

def test_initialize_expert_unsupported_provider(clean_env):
    """Test error handling for unsupported provider in expert mode."""
    with pytest.raises(ValueError, match=r"Unsupported provider: unknown"):
        initialize_expert_llm("unknown", "model")

def test_initialize_openai(clean_env, mock_openai):
    """Test OpenAI provider initialization"""
    os.environ["OPENAI_API_KEY"] = "test-key"
    model = initialize_llm("openai", "gpt-4")
    
    mock_openai.assert_called_once_with(
        api_key="test-key",
        model="gpt-4"
    )

def test_initialize_anthropic(clean_env, mock_anthropic):
    """Test Anthropic provider initialization"""
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    model = initialize_llm("anthropic", "claude-3")
    
    mock_anthropic.assert_called_once_with(
        api_key="test-key",
        model_name="claude-3"
    )

def test_initialize_openrouter(clean_env, mock_openai):
    """Test OpenRouter provider initialization"""
    os.environ["OPENROUTER_API_KEY"] = "test-key"
    model = initialize_llm("openrouter", "mistral-large")
    
    mock_openai.assert_called_once_with(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="mistral-large"
    )

def test_initialize_openai_compatible(clean_env, mock_openai):
    """Test OpenAI-compatible provider initialization"""
    os.environ["OPENAI_API_KEY"] = "test-key"
    os.environ["OPENAI_API_BASE"] = "https://custom-endpoint/v1"
    model = initialize_llm("openai-compatible", "local-model")
    
    mock_openai.assert_called_once_with(
        api_key="test-key",
        base_url="https://custom-endpoint/v1",
        model="local-model"
    )

def test_initialize_unsupported_provider(clean_env):
    """Test initialization with unsupported provider raises ValueError"""
    with pytest.raises(ValueError) as exc_info:
        initialize_llm("unsupported", "model")
    assert str(exc_info.value) == "Unsupported provider: unsupported"

def test_provider_name_validation():
    """Test provider name validation and normalization."""
    # Test all supported providers
    providers = ["openai", "anthropic", "openrouter", "openai-compatible"]
    for provider in providers:
        try:
            with patch(f'ra_aid.llm.ChatOpenAI'), patch('ra_aid.llm.ChatAnthropic'):
                initialize_llm(provider, "test-model")
        except ValueError:
            pytest.fail(f"Valid provider {provider} raised ValueError")
    
    # Test case sensitivity
    with patch('ra_aid.llm.ChatOpenAI'):
        with pytest.raises(ValueError):
            initialize_llm("OpenAI", "test-model")

def test_initialize_llm_cross_provider(clean_env, mock_openai, mock_anthropic, monkeypatch):
    """Test initializing different providers in sequence."""
    # Initialize OpenAI
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    llm1 = initialize_llm("openai", "gpt-4")
    
    # Initialize Anthropic
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key") 
    llm2 = initialize_llm("anthropic", "claude-3")
    
    # Verify both were initialized correctly
    mock_openai.assert_called_once_with(
        api_key="openai-key",
        model="gpt-4"
    )
    mock_anthropic.assert_called_once_with(
        api_key="anthropic-key",
        model_name="claude-3"
    )

def test_environment_variable_precedence(clean_env, mock_openai, monkeypatch):
    """Test environment variable precedence and fallback."""
    from ra_aid.env import validate_environment
    from dataclasses import dataclass
    
    @dataclass
    class Args:
        provider: str
        expert_provider: str
        model: str = None
        expert_model: str = None

    # Test expert mode with explicit key
    # Set up base environment first
    monkeypatch.setenv("OPENAI_API_KEY", "base-key") 
    monkeypatch.setenv("EXPERT_OPENAI_API_KEY", "expert-key")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")
    args = Args(provider="openai", expert_provider="openai")
    expert_enabled, expert_missing, web_enabled, web_missing = validate_environment(args)
    assert expert_enabled
    assert not expert_missing
    assert web_enabled
    assert not web_missing

    llm = initialize_expert_llm()
    mock_openai.assert_called_with(
        api_key="expert-key",
        model="o1-preview"
    )
    
    # Test empty key validation
    monkeypatch.setenv("EXPERT_OPENAI_API_KEY", "")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # Remove fallback
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)  # Remove web research
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")  # Add for provider validation
    args = Args(provider="anthropic", expert_provider="openai")  # Change base provider to avoid validation error
    expert_enabled, expert_missing, web_enabled, web_missing = validate_environment(args)
    assert not expert_enabled
    assert len(expert_missing) == 1
    assert expert_missing[0] == "EXPERT_OPENAI_API_KEY environment variable is not set"
    assert not web_enabled
    assert len(web_missing) == 1
    assert web_missing[0] == "TAVILY_API_KEY environment variable is not set"

@pytest.fixture
def mock_anthropic():
    """
    Mock ChatAnthropic class for testing Anthropic provider initialization.
    Prevents actual API calls during testing.
    """
    with patch('ra_aid.llm.ChatAnthropic') as mock:
        mock.return_value = Mock(spec=ChatAnthropic)
        yield mock
