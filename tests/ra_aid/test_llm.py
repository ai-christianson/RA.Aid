import os
from dataclasses import dataclass
from unittest.mock import Mock, patch

import pytest
from langchain_anthropic.chat_models import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_openai.chat_models import ChatOpenAI

from ra_aid.agents.ciayn_agent import CiaynAgent
from ra_aid.env import validate_environment
from ra_aid.llm import (
    create_llm_client,
    get_env_var,
    get_provider_config,
    initialize_expert_llm,
    initialize_llm,
)


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
        "GEMINI_API_KEY",
        "EXPERT_GEMINI_API_KEY",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def mock_openai():
    """
    Mock ChatOpenAI class for testing OpenAI provider initialization.
    Prevents actual API calls during testing.
    """
    with patch("ra_aid.llm.ChatOpenAI") as mock:
        mock.return_value = Mock(spec=ChatOpenAI)
        yield mock


def test_initialize_expert_defaults(clean_env, mock_openai, monkeypatch):
    """Test expert LLM initialization with default parameters."""
    monkeypatch.setenv("EXPERT_OPENAI_API_KEY", "test-key")
    _llm = initialize_expert_llm()

    mock_openai.assert_called_once_with(api_key="test-key", model="o1", temperature=0)


def test_initialize_expert_openai_custom(clean_env, mock_openai, monkeypatch):
    """Test expert OpenAI initialization with custom parameters."""
    monkeypatch.setenv("EXPERT_OPENAI_API_KEY", "test-key")
    _llm = initialize_expert_llm("openai", "gpt-4-preview")

    mock_openai.assert_called_once_with(
        api_key="test-key", model="gpt-4-preview", temperature=0
    )


def test_initialize_expert_gemini(clean_env, mock_gemini, monkeypatch):
    """Test expert Gemini initialization."""
    monkeypatch.setenv("EXPERT_GEMINI_API_KEY", "test-key")
    _llm = initialize_expert_llm("gemini", "gemini-2.0-flash-thinking-exp-1219")

    mock_gemini.assert_called_once_with(
        api_key="test-key", model="gemini-2.0-flash-thinking-exp-1219", temperature=0
    )


def test_initialize_expert_anthropic(clean_env, mock_anthropic, monkeypatch):
    """Test expert Anthropic initialization."""
    monkeypatch.setenv("EXPERT_ANTHROPIC_API_KEY", "test-key")
    _llm = initialize_expert_llm("anthropic", "claude-3")

    mock_anthropic.assert_called_once_with(
        api_key="test-key", model_name="claude-3", temperature=0
    )


def test_initialize_expert_openrouter(clean_env, mock_openai, monkeypatch):
    """Test expert OpenRouter initialization."""
    monkeypatch.setenv("EXPERT_OPENROUTER_API_KEY", "test-key")
    _llm = initialize_expert_llm("openrouter", "models/mistral-large")

    mock_openai.assert_called_once_with(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="models/mistral-large",
        temperature=0,
    )


def test_initialize_expert_openai_compatible(clean_env, mock_openai, monkeypatch):
    """Test expert OpenAI-compatible initialization."""
    monkeypatch.setenv("EXPERT_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EXPERT_OPENAI_API_BASE", "http://test-url")
    _llm = initialize_expert_llm("openai-compatible", "local-model")

    mock_openai.assert_called_once_with(
        api_key="test-key",
        base_url="http://test-url",
        model="local-model",
        temperature=0,
    )


def test_initialize_expert_unsupported_provider(clean_env):
    """Test error handling for unsupported provider in expert mode."""
    with pytest.raises(ValueError, match=r"Unsupported provider: unknown"):
        initialize_expert_llm("unknown", "model")


def test_estimate_tokens():
    """Test token estimation functionality."""
    # Test empty/None cases
    assert CiaynAgent._estimate_tokens(None) == 0
    assert CiaynAgent._estimate_tokens("") == 0

    # Test string content
    assert CiaynAgent._estimate_tokens("test") == 1  # 4 bytes
    assert CiaynAgent._estimate_tokens("hello world") == 2  # 11 bytes
    assert CiaynAgent._estimate_tokens("🚀") == 1  # 4 bytes

    # Test message content
    msg = HumanMessage(content="test message")
    assert CiaynAgent._estimate_tokens(msg) == 3  # 11 bytes


def test_initialize_openai(clean_env, mock_openai):
    """Test OpenAI provider initialization"""
    os.environ["OPENAI_API_KEY"] = "test-key"
    _model = initialize_llm("openai", "gpt-4")

    mock_openai.assert_called_once_with(api_key="test-key", model="gpt-4")


def test_initialize_gemini(clean_env, mock_gemini):
    """Test Gemini provider initialization"""
    os.environ["GEMINI_API_KEY"] = "test-key"
    _model = initialize_llm("gemini", "gemini-2.0-flash-thinking-exp-1219")

    mock_gemini.assert_called_once_with(
        api_key="test-key", model="gemini-2.0-flash-thinking-exp-1219"
    )


def test_initialize_anthropic(clean_env, mock_anthropic):
    """Test Anthropic provider initialization"""
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    _model = initialize_llm("anthropic", "claude-3")

    mock_anthropic.assert_called_once_with(api_key="test-key", model_name="claude-3")


def test_initialize_openrouter(clean_env, mock_openai):
    """Test OpenRouter provider initialization"""
    os.environ["OPENROUTER_API_KEY"] = "test-key"
    _model = initialize_llm("openrouter", "mistral-large")

    mock_openai.assert_called_once_with(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="mistral-large",
    )


def test_initialize_openai_compatible(clean_env, mock_openai):
    """Test OpenAI-compatible provider initialization"""
    os.environ["OPENAI_API_KEY"] = "test-key"
    os.environ["OPENAI_API_BASE"] = "https://custom-endpoint/v1"
    _model = initialize_llm("openai-compatible", "local-model")

    mock_openai.assert_called_once_with(
        api_key="test-key",
        base_url="https://custom-endpoint/v1",
        model="local-model",
        temperature=0.3,
    )


def test_initialize_unsupported_provider(clean_env):
    """Test initialization with unsupported provider raises ValueError"""
    with pytest.raises(ValueError) as exc_info:
        initialize_llm("unsupported", "model")
    assert str(exc_info.value) == "Unsupported provider: unsupported"


def test_temperature_defaults(clean_env, mock_openai, mock_anthropic, mock_gemini):
    """Test default temperature behavior for different providers."""
    os.environ["OPENAI_API_KEY"] = "test-key"
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    os.environ["OPENAI_API_BASE"] = "http://test-url"
    os.environ["GEMINI_API_KEY"] = "test-key"
    # Test openai-compatible default temperature
    initialize_llm("openai-compatible", "test-model")
    mock_openai.assert_called_with(
        api_key="test-key",
        base_url="http://test-url",
        model="test-model",
        temperature=0.3,
    )

    # Test other providers don't set temperature by default
    initialize_llm("openai", "test-model")
    mock_openai.assert_called_with(api_key="test-key", model="test-model")

    initialize_llm("anthropic", "test-model")
    mock_anthropic.assert_called_with(api_key="test-key", model_name="test-model")

    initialize_llm("gemini", "test-model")
    mock_gemini.assert_called_with(api_key="test-key", model="test-model")


def test_explicit_temperature(clean_env, mock_openai, mock_anthropic, mock_gemini):
    """Test explicit temperature setting for each provider."""
    os.environ["OPENAI_API_KEY"] = "test-key"
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    os.environ["OPENROUTER_API_KEY"] = "test-key"
    os.environ["GEMINI_API_KEY"] = "test-key"

    test_temp = 0.7

    # Test OpenAI
    initialize_llm("openai", "test-model", temperature=test_temp)
    mock_openai.assert_called_with(
        api_key="test-key", model="test-model", temperature=test_temp
    )

    # Test Gemini
    initialize_llm("gemini", "test-model", temperature=test_temp)
    mock_gemini.assert_called_with(
        api_key="test-key", model="test-model", temperature=test_temp
    )

    # Test Anthropic
    initialize_llm("anthropic", "test-model", temperature=test_temp)
    mock_anthropic.assert_called_with(
        api_key="test-key", model_name="test-model", temperature=test_temp
    )

    # Test OpenRouter
    initialize_llm("openrouter", "test-model", temperature=test_temp)
    mock_openai.assert_called_with(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test-model",
        temperature=test_temp,
    )


def test_temperature_validation(clean_env, mock_openai):
    """Test temperature validation in command line arguments."""
    from ra_aid.__main__ import parse_arguments

    # Test temperature below minimum
    with pytest.raises(SystemExit):
        parse_arguments(["--message", "test", "--temperature", "-0.1"])

    # Test temperature above maximum
    with pytest.raises(SystemExit):
        parse_arguments(["--message", "test", "--temperature", "2.1"])

    # Test valid temperature
    args = parse_arguments(["--message", "test", "--temperature", "0.7"])
    assert args.temperature == 0.7


def test_provider_name_validation():
    """Test provider name validation and normalization."""
    # Test all supported providers
    providers = ["openai", "anthropic", "openrouter", "openai-compatible", "gemini"]
    for provider in providers:
        try:
            with patch("ra_aid.llm.ChatOpenAI"), patch("ra_aid.llm.ChatAnthropic"):
                initialize_llm(provider, "test-model")
        except ValueError:
            pytest.fail(f"Valid provider {provider} raised ValueError")

    # Test case sensitivity
    with patch("ra_aid.llm.ChatOpenAI"):
        with pytest.raises(ValueError):
            initialize_llm("OpenAI", "test-model")


def test_initialize_llm_cross_provider(
    clean_env, mock_openai, mock_anthropic, mock_gemini, monkeypatch
):
    """Test initializing different providers in sequence."""
    # Initialize OpenAI
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    _llm1 = initialize_llm("openai", "gpt-4")

    # Initialize Anthropic
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    _llm2 = initialize_llm("anthropic", "claude-3")

    # Initialize Gemini
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    _llm3 = initialize_llm("gemini", "gemini-2.0-flash-thinking-exp-1219")

    # Verify both were initialized correctly
    mock_openai.assert_called_once_with(api_key="openai-key", model="gpt-4")
    mock_anthropic.assert_called_once_with(
        api_key="anthropic-key", model_name="claude-3"
    )
    mock_gemini.assert_called_once_with(
        api_key="gemini-key", model="gemini-2.0-flash-thinking-exp-1219"
    )


@dataclass
class Args:
    """Test arguments class."""

    provider: str
    expert_provider: str
    model: str = None
    expert_model: str = None


def test_environment_variable_precedence(clean_env, mock_openai, monkeypatch):
    """Test environment variable precedence and fallback."""
    # Test get_env_var helper with fallback
    monkeypatch.setenv("TEST_KEY", "base-value")
    monkeypatch.setenv("EXPERT_TEST_KEY", "expert-value")

    assert get_env_var("TEST_KEY") == "base-value"
    assert get_env_var("TEST_KEY", expert=True) == "expert-value"

    # Test fallback when expert value not set
    monkeypatch.delenv("EXPERT_TEST_KEY", raising=False)
    assert get_env_var("TEST_KEY", expert=True) == "base-value"

    # Test provider config
    monkeypatch.setenv("EXPERT_OPENAI_API_KEY", "expert-key")
    config = get_provider_config("openai", is_expert=True)
    assert config["api_key"] == "expert-key"

    # Test LLM client creation with expert mode
    _llm = create_llm_client("openai", "o1", is_expert=True)
    mock_openai.assert_called_with(api_key="expert-key", model="o1", temperature=0)

    # Test environment validation
    monkeypatch.setenv("EXPERT_OPENAI_API_KEY", "")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")

    args = Args(provider="anthropic", expert_provider="openai")
    expert_enabled, expert_missing, web_enabled, web_missing = validate_environment(
        args
    )
    assert not expert_enabled
    assert expert_missing
    assert not web_enabled
    assert web_missing


@pytest.fixture
def mock_anthropic():
    """
    Mock ChatAnthropic class for testing Anthropic provider initialization.
    Prevents actual API calls during testing.
    """
    with patch("ra_aid.llm.ChatAnthropic") as mock:
        mock.return_value = Mock(spec=ChatAnthropic)
        yield mock


@pytest.fixture
def mock_gemini():
    """Mock ChatGoogleGenerativeAI class for testing Gemini provider initialization."""
    with patch("ra_aid.llm.ChatGoogleGenerativeAI") as mock:
        mock.return_value = Mock(spec=ChatGoogleGenerativeAI)
        yield mock


@pytest.fixture
def mock_deepseek_reasoner():
    """Mock ChatDeepseekReasoner for testing DeepSeek provider initialization."""
    with patch("ra_aid.llm.ChatDeepseekReasoner") as mock:
        mock.return_value = Mock()
        yield mock


def test_initialize_deepseek(
    clean_env, mock_openai, mock_deepseek_reasoner, monkeypatch
):
    """Test DeepSeek provider initialization with different models."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    # Test with reasoner model
    _model = initialize_llm("deepseek", "deepseek-reasoner")
    mock_deepseek_reasoner.assert_called_with(
        api_key="test-key",
        base_url="https://api.deepseek.com",
        temperature=1,
        model="deepseek-reasoner",
    )

    # Test with non-reasoner model
    _model = initialize_llm("deepseek", "deepseek-chat")
    mock_openai.assert_called_with(
        api_key="test-key",
        base_url="https://api.deepseek.com",
        temperature=1,
        model="deepseek-chat",
    )


def test_initialize_expert_deepseek(
    clean_env, mock_openai, mock_deepseek_reasoner, monkeypatch
):
    """Test expert DeepSeek provider initialization."""
    monkeypatch.setenv("EXPERT_DEEPSEEK_API_KEY", "test-key")

    # Test with reasoner model
    _model = initialize_expert_llm("deepseek", "deepseek-reasoner")
    mock_deepseek_reasoner.assert_called_with(
        api_key="test-key",
        base_url="https://api.deepseek.com",
        temperature=0,
        model="deepseek-reasoner",
    )

    # Test with non-reasoner model
    _model = initialize_expert_llm("deepseek", "deepseek-chat")
    mock_openai.assert_called_with(
        api_key="test-key",
        base_url="https://api.deepseek.com",
        temperature=0,
        model="deepseek-chat",
    )


def test_initialize_openrouter_deepseek(
    clean_env, mock_openai, mock_deepseek_reasoner, monkeypatch
):
    """Test OpenRouter DeepSeek model initialization."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    # Test with DeepSeek R1 model
    _model = initialize_llm("openrouter", "deepseek/deepseek-r1")
    mock_deepseek_reasoner.assert_called_with(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        temperature=1,
        model="deepseek/deepseek-r1",
    )

    # Test with non-DeepSeek model
    _model = initialize_llm("openrouter", "mistral/mistral-large")
    mock_openai.assert_called_with(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="mistral/mistral-large",
    )


def test_initialize_expert_openrouter_deepseek(
    clean_env, mock_openai, mock_deepseek_reasoner, monkeypatch
):
    """Test expert OpenRouter DeepSeek model initialization."""
    monkeypatch.setenv("EXPERT_OPENROUTER_API_KEY", "test-key")

    # Test with DeepSeek R1 model via create_llm_client
    _model = create_llm_client("openrouter", "deepseek/deepseek-r1", is_expert=True)
    mock_deepseek_reasoner.assert_called_with(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        model="deepseek/deepseek-r1",
    )

    # Test with non-DeepSeek model
    _model = create_llm_client("openrouter", "mistral/mistral-large", is_expert=True)
    mock_openai.assert_called_with(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="mistral/mistral-large",
        temperature=0,
    )


def test_deepseek_environment_fallback(clean_env, mock_deepseek_reasoner, monkeypatch):
    """Test DeepSeek environment variable fallback behavior."""
    # Test environment variable helper with fallback
    monkeypatch.setenv("DEEPSEEK_API_KEY", "base-key")
    assert get_env_var("DEEPSEEK_API_KEY", expert=True) == "base-key"

    # Test provider config with fallback
    config = get_provider_config("deepseek", is_expert=True)
    assert config["api_key"] == "base-key"
    assert config["base_url"] == "https://api.deepseek.com"

    # Test with expert key
    monkeypatch.setenv("EXPERT_DEEPSEEK_API_KEY", "expert-key")
    config = get_provider_config("deepseek", is_expert=True)
    assert config["api_key"] == "expert-key"

    # Test client creation with expert key
    _model = create_llm_client("deepseek", "deepseek-reasoner", is_expert=True)
    mock_deepseek_reasoner.assert_called_with(
        api_key="expert-key",
        base_url="https://api.deepseek.com",
        temperature=0,
        model="deepseek-reasoner",
    )
