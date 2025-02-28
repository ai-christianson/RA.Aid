import os
from dataclasses import dataclass
from unittest import mock
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
    get_available_openai_models,
    get_env_var,
    get_provider_config,
    initialize_expert_llm,
    initialize_llm,
    select_expert_model,
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
    """Test expert LLM initialization with explicit parameters."""
    monkeypatch.setenv("EXPERT_OPENAI_API_KEY", "test-key")
    _llm = initialize_expert_llm("openai", "o1")

    mock_openai.assert_called_once_with(
        api_key="test-key",
        model="o1",
        reasoning_effort="high",
        timeout=180,
        max_retries=5,
    )


def test_initialize_expert_openai_custom(clean_env, mock_openai, monkeypatch):
    """Test expert OpenAI initialization with custom parameters."""
    monkeypatch.setenv("EXPERT_OPENAI_API_KEY", "test-key")
    _llm = initialize_expert_llm("openai", "gpt-4-preview")

    mock_openai.assert_called_once_with(
        api_key="test-key",
        model="gpt-4-preview",
        temperature=0,
        timeout=180,
        max_retries=5,
    )


def test_initialize_expert_gemini(clean_env, mock_gemini, monkeypatch):
    """Test expert Gemini initialization."""
    monkeypatch.setenv("EXPERT_GEMINI_API_KEY", "test-key")
    _llm = initialize_expert_llm("gemini", "gemini-2.0-flash-thinking-exp-1219")

    mock_gemini.assert_called_once_with(
        api_key="test-key",
        model="gemini-2.0-flash-thinking-exp-1219",
        temperature=0,
        timeout=180,
        max_retries=5,
    )


def test_initialize_expert_anthropic(clean_env, mock_anthropic, monkeypatch):
    """Test expert Anthropic initialization."""
    monkeypatch.setenv("EXPERT_ANTHROPIC_API_KEY", "test-key")
    _llm = initialize_expert_llm("anthropic", "claude-3")

    # Check that mock_anthropic was called
    assert mock_anthropic.called

    # Verify essential parameters
    kwargs = mock_anthropic.call_args.kwargs
    assert kwargs["api_key"] == "test-key"
    assert kwargs["model_name"] == "claude-3"
    assert kwargs["temperature"] == 0
    assert kwargs["timeout"] == 180
    assert kwargs["max_retries"] == 5


def test_initialize_expert_openrouter(clean_env, mock_openai, monkeypatch):
    """Test expert OpenRouter initialization."""
    monkeypatch.setenv("EXPERT_OPENROUTER_API_KEY", "test-key")
    _llm = initialize_expert_llm("openrouter", "models/mistral-large")

    mock_openai.assert_called_once_with(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="models/mistral-large",
        temperature=0,
        timeout=180,
        max_retries=5,
        default_headers={"HTTP-Referer": "https://ra-aid.ai", "X-Title": "RA.Aid"},
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
        timeout=180,
        max_retries=5,
    )


def test_initialize_expert_unsupported_provider(clean_env):
    """Test error handling for unsupported provider in expert mode."""
    with pytest.raises(
        ValueError, match=r"Missing required environment variable for provider: unknown"
    ):
        initialize_expert_llm("unknown", "model")


def test_estimate_tokens():
    """Test token estimation functionality."""
    # Test empty/None cases
    assert CiaynAgent._estimate_tokens(None) == 0
    assert CiaynAgent._estimate_tokens("") == 0

    # Test string content
    assert CiaynAgent._estimate_tokens("test") == 2  # 4 bytes
    assert CiaynAgent._estimate_tokens("hello world") == 5  # 11 bytes
    assert CiaynAgent._estimate_tokens("🚀") == 2  # 4 bytes

    # Test message content
    msg = HumanMessage(content="test message")
    assert CiaynAgent._estimate_tokens(msg) == 6  # 11 bytes


def test_initialize_openai(clean_env, mock_openai):
    """Test OpenAI provider initialization"""
    os.environ["OPENAI_API_KEY"] = "test-key"
    _model = initialize_llm("openai", "gpt-4", temperature=0.7)

    mock_openai.assert_called_once_with(
        api_key="test-key", model="gpt-4", temperature=0.7, timeout=180, max_retries=5
    )


def test_initialize_gemini(clean_env, mock_gemini):
    """Test Gemini provider initialization"""
    os.environ["GEMINI_API_KEY"] = "test-key"
    _model = initialize_llm(
        "gemini", "gemini-2.0-flash-thinking-exp-1219", temperature=0.7
    )

    mock_gemini.assert_called_with(
        api_key="test-key",
        model="gemini-2.0-flash-thinking-exp-1219",
        temperature=0.7,
        timeout=180,
        max_retries=5,
    )


def test_initialize_anthropic(clean_env, mock_anthropic):
    """Test Anthropic provider initialization"""
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    _model = initialize_llm("anthropic", "claude-3", temperature=0.7)

    # Check that mock_anthropic was called
    assert mock_anthropic.called

    # Verify essential parameters
    kwargs = mock_anthropic.call_args.kwargs
    assert kwargs["api_key"] == "test-key"
    assert kwargs["model_name"] == "claude-3"
    assert kwargs["temperature"] == 0.7
    assert kwargs["timeout"] == 180
    assert kwargs["max_retries"] == 5


def test_initialize_openrouter(clean_env, mock_openai):
    """Test OpenRouter provider initialization"""
    os.environ["OPENROUTER_API_KEY"] = "test-key"
    _model = initialize_llm("openrouter", "mistral-large", temperature=0.7)

    mock_openai.assert_called_with(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="mistral-large",
        temperature=0.7,
        timeout=180,
        max_retries=5,
        default_headers={"HTTP-Referer": "https://ra-aid.ai", "X-Title": "RA.Aid"},
    )


def test_initialize_openai_compatible(clean_env, mock_openai):
    """Test OpenAI-compatible provider initialization"""
    os.environ["OPENAI_API_KEY"] = "test-key"
    os.environ["OPENAI_API_BASE"] = "https://custom-endpoint/v1"
    _model = initialize_llm("openai-compatible", "local-model", temperature=0.3)

    mock_openai.assert_called_with(
        api_key="test-key",
        base_url="https://custom-endpoint/v1",
        model="local-model",
        temperature=0.3,
        timeout=180,
        max_retries=5,
    )


def test_initialize_unsupported_provider(clean_env):
    """Test initialization with unsupported provider raises ValueError"""
    with pytest.raises(
        ValueError, match=r"Missing required environment variable for provider: unknown"
    ):
        initialize_llm("unknown", "model")


def test_temperature_defaults(clean_env, mock_openai, mock_anthropic, mock_gemini):
    """Test default temperature behavior for different providers."""
    os.environ["OPENAI_API_KEY"] = "test-key"
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    os.environ["OPENAI_API_BASE"] = "http://test-url"
    os.environ["GEMINI_API_KEY"] = "test-key"

    # Test openai-compatible default temperature
    initialize_llm("openai-compatible", "test-model", temperature=0.3)
    mock_openai.assert_called_with(
        api_key="test-key",
        base_url="http://test-url",
        model="test-model",
        temperature=0.3,
        timeout=180,
        max_retries=5,
    )

    # Test default temperature when none is provided for models that support it
    initialize_llm("openai", "test-model")
    mock_openai.assert_called_with(
        api_key="test-key",
        model="test-model",
        temperature=0.7,
        timeout=180,
        max_retries=5,
    )

    initialize_llm("anthropic", "test-model")

    # Verify essential parameters for Anthropic
    kwargs = mock_anthropic.call_args.kwargs
    assert kwargs["api_key"] == "test-key"
    assert kwargs["model_name"] == "test-model"
    assert kwargs["temperature"] == 0.7
    assert kwargs["timeout"] == 180
    assert kwargs["max_retries"] == 5

    initialize_llm("gemini", "test-model")
    mock_gemini.assert_called_with(
        api_key="test-key",
        model="test-model",
        temperature=0.7,
        timeout=180,
        max_retries=5,
    )

    # Test expert models don't require temperature
    initialize_expert_llm("openai", "o1")
    mock_openai.assert_called_with(
        api_key="test-key",
        model="o1",
        reasoning_effort="high",
        timeout=180,
        max_retries=5,
    )

    initialize_expert_llm("openai", "o1-mini")
    mock_openai.assert_called_with(
        api_key="test-key",
        model="o1-mini",
        reasoning_effort="high",
        timeout=180,
        max_retries=5,
    )


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
        api_key="test-key",
        model="test-model",
        temperature=test_temp,
        timeout=180,
        max_retries=5,
    )

    # Test Gemini
    initialize_llm("gemini", "test-model", temperature=test_temp)
    mock_gemini.assert_called_with(
        api_key="test-key",
        model="test-model",
        temperature=test_temp,
        timeout=180,
        max_retries=5,
    )

    # Test Anthropic
    initialize_llm("anthropic", "test-model", temperature=test_temp)

    # Verify essential parameters for Anthropic
    kwargs = mock_anthropic.call_args.kwargs
    assert kwargs["api_key"] == "test-key"
    assert kwargs["model_name"] == "test-model"
    assert kwargs["temperature"] == test_temp
    assert kwargs["timeout"] == 180
    assert kwargs["max_retries"] == 5

    # Test OpenRouter
    initialize_llm("openrouter", "test-model", temperature=test_temp)
    mock_openai.assert_called_with(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test-model",
        temperature=test_temp,
        timeout=180,
        max_retries=5,
        default_headers={"HTTP-Referer": "https://ra-aid.ai", "X-Title": "RA.Aid"},
    )


def test_get_available_openai_models_success():
    """Test successful retrieval of OpenAI models."""
    mock_model = Mock()
    mock_model.id = "gpt-4"
    mock_models = Mock()
    mock_models.data = [mock_model]

    with mock.patch("ra_aid.llm.OpenAI") as mock_client:
        mock_client.return_value.models.list.return_value = mock_models
        models = get_available_openai_models()
        assert models == ["gpt-4"]
        mock_client.return_value.models.list.assert_called_once()


def test_get_available_openai_models_failure():
    """Test graceful handling of model retrieval failure."""
    with mock.patch("ra_aid.llm.OpenAI") as mock_client:
        mock_client.return_value.models.list.side_effect = Exception("API Error")
        models = get_available_openai_models()
        assert models == []
        mock_client.return_value.models.list.assert_called_once()


def test_select_expert_model_explicit():
    """Test model selection with explicitly specified model."""
    model = select_expert_model("openai", "gpt-4")
    assert model == "gpt-4"


def test_select_expert_model_non_openai():
    """Test model selection for non-OpenAI provider."""
    model = select_expert_model("anthropic", None)
    assert model is None


def test_select_expert_model_priority():
    """Test model selection follows priority order."""
    available_models = ["gpt-4", "o1", "o3-mini"]

    with mock.patch(
        "ra_aid.llm.get_available_openai_models", return_value=available_models
    ):
        model = select_expert_model("openai")
        assert model == "o3-mini"


def test_select_expert_model_no_match():
    """Test model selection when no priority models available."""
    available_models = ["gpt-4", "gpt-3.5"]

    with mock.patch(
        "ra_aid.llm.get_available_openai_models", return_value=available_models
    ):
        model = select_expert_model("openai")
        assert model is None


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
                initialize_llm(provider, "test-model", temperature=0.7)
        except ValueError as e:
            if "Temperature must be provided" not in str(e):
                pytest.fail(
                    f"Valid provider {provider} raised unexpected ValueError: {e}"
                )


def test_initialize_llm_cross_provider(
    clean_env, mock_openai, mock_anthropic, mock_gemini, monkeypatch
):
    """Test initializing different providers in sequence."""
    # Initialize OpenAI
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    _llm1 = initialize_llm("openai", "gpt-4", temperature=0.7)
    mock_openai.assert_called_with(
        api_key="openai-key", model="gpt-4", temperature=0.7, timeout=180, max_retries=5
    )

    # Initialize Anthropic
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    _llm2 = initialize_llm("anthropic", "claude-3", temperature=0.7)

    # Verify essential parameters for Anthropic
    kwargs = mock_anthropic.call_args.kwargs
    assert kwargs["api_key"] == "anthropic-key"
    assert kwargs["model_name"] == "claude-3"
    assert kwargs["temperature"] == 0.7
    assert kwargs["timeout"] == 180
    assert kwargs["max_retries"] == 5

    # Initialize Gemini
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    _llm3 = initialize_llm("gemini", "gemini-pro", temperature=0.7)
    mock_gemini.assert_called_with(
        api_key="gemini-key",
        model="gemini-pro",
        temperature=0.7,
        timeout=180,
        max_retries=5,
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
    mock_openai.assert_called_with(
        api_key="expert-key",
        model="o1",
        reasoning_effort="high",
        timeout=180,
        max_retries=5,
    )

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


def test_reasoning_effort_only_passed_to_supported_models(
    clean_env, mock_openai, monkeypatch
):
    """Test that reasoning_effort is only passed to supported models."""
    monkeypatch.setenv("EXPERT_OPENAI_API_KEY", "test-key")

    # Initialize expert LLM with GPT-4 (which doesn't support reasoning_effort)
    _llm = initialize_expert_llm("openai", "gpt-4")

    # Verify reasoning_effort was not included in kwargs
    mock_openai.assert_called_with(
        api_key="test-key",
        model="gpt-4",
        temperature=0,
        timeout=180,
        max_retries=5,
    )


def test_reasoning_effort_passed_to_supported_models(
    clean_env, mock_openai, monkeypatch
):
    """Test that reasoning_effort is passed to models that support it."""
    monkeypatch.setenv("EXPERT_OPENAI_API_KEY", "test-key")

    # Initialize expert LLM with o1 (which supports reasoning_effort)
    _llm = initialize_expert_llm("openai", "o1")

    # Verify reasoning_effort was included in kwargs
    mock_openai.assert_called_with(
        api_key="test-key",
        model="o1",
        reasoning_effort="high",
        timeout=180,
        max_retries=5,
    )


def test_initialize_deepseek(
    clean_env, mock_openai, mock_deepseek_reasoner, monkeypatch
):
    """Test DeepSeek provider initialization with different models."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    # Test with reasoner model
    _model = initialize_llm("deepseek", "deepseek-reasoner", temperature=0.7)
    mock_deepseek_reasoner.assert_called_with(
        api_key="test-key",
        base_url="https://api.deepseek.com",
        model="deepseek-reasoner",
        temperature=0.7,
        timeout=180,
        max_retries=5,
    )

    # Test with OpenAI-compatible model
    _model = initialize_llm("deepseek", "deepseek-chat", temperature=0.7)
    mock_openai.assert_called_with(
        api_key="test-key",
        base_url="https://api.deepseek.com",  # Updated to match implementation
        model="deepseek-chat",
        temperature=0.7,
        timeout=180,
        max_retries=5,
    )


def test_initialize_openrouter_deepseek(
    clean_env, mock_openai, mock_deepseek_reasoner, monkeypatch
):
    """Test OpenRouter DeepSeek model initialization."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    # Test with DeepSeek R1 model
    _model = initialize_llm("openrouter", "deepseek/deepseek-r1", temperature=0.7)
    mock_deepseek_reasoner.assert_called_with(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="deepseek/deepseek-r1",
        temperature=0.7,
        timeout=180,
        max_retries=5,
        default_headers={"HTTP-Referer": "https://ra-aid.ai", "X-Title": "RA.Aid"},
    )
