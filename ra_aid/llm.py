import os
from typing import Any, Dict, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from openai import OpenAI

from ra_aid.chat_models.deepseek_chat import ChatDeepseekReasoner
from ra_aid.console.output import cpm
from ra_aid.logging_config import get_logger

from .models_params import models_params


def get_available_openai_models() -> List[str]:
    """Fetch available OpenAI models using OpenAI client.

    Returns:
        List of available model names
    """
    try:
        # Use OpenAI client to fetch models
        client = OpenAI()
        models = client.models.list()
        return [str(model.id) for model in models.data]
    except Exception:
        # Return empty list if unable to fetch models
        return []


def select_expert_model(provider: str, model: Optional[str] = None) -> Optional[str]:
    """Select appropriate expert model based on provider and availability.

    Args:
        provider: The LLM provider
        model: Optional explicitly specified model name

    Returns:
        Selected model name or None if no suitable model found
    """
    if provider != "openai" or model is not None:
        return model

    # Try to get available models
    available_models = get_available_openai_models()

    # Priority order for expert models
    priority_models = ["o3-mini", "o1", "o1-preview"]

    # Return first available model from priority list
    for model_name in priority_models:
        if model_name in available_models:
            return model_name

    return None


known_temp_providers = {
    "openai",
    "anthropic",
    "openrouter",
    "openai-compatible",
    "gemini",
    "deepseek",
}

# Constants for API request configuration
LLM_REQUEST_TIMEOUT = 180
LLM_MAX_RETRIES = 5

logger = get_logger(__name__)


def get_env_var(name: str, expert: bool = False) -> Optional[str]:
    """Get environment variable with optional expert prefix and fallback."""
    prefix = "EXPERT_" if expert else ""
    value = os.getenv(f"{prefix}{name}")

    # If expert mode and no expert value, fall back to base value
    if expert and not value:
        value = os.getenv(name)

    return value


def create_deepseek_client(
    model_name: str,
    api_key: str,
    base_url: str,
    temperature: Optional[float] = None,
    is_expert: bool = False,
) -> BaseChatModel:
    """Create DeepSeek client with appropriate configuration."""
    if model_name.lower() == "deepseek-reasoner":
        return ChatDeepseekReasoner(
            api_key=api_key,
            base_url=base_url,
            temperature=(
                0 if is_expert else (temperature if temperature is not None else 1)
            ),
            model=model_name,
            timeout=LLM_REQUEST_TIMEOUT,
            max_retries=LLM_MAX_RETRIES,
        )

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        temperature=0 if is_expert else (temperature if temperature is not None else 1),
        model=model_name,
        timeout=LLM_REQUEST_TIMEOUT,
        max_retries=LLM_MAX_RETRIES,
    )


def create_openrouter_client(
    model_name: str,
    api_key: str,
    temperature: Optional[float] = None,
    is_expert: bool = False,
) -> BaseChatModel:
    """Create OpenRouter client with appropriate configuration."""
    default_headers = {"HTTP-Referer": "https://ra-aid.ai", "X-Title": "RA.Aid"}

    if model_name.startswith("deepseek/") and "deepseek-r1" in model_name.lower():
        return ChatDeepseekReasoner(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=(
                0 if is_expert else (temperature if temperature is not None else 1)
            ),
            model=model_name,
            timeout=LLM_REQUEST_TIMEOUT,
            max_retries=LLM_MAX_RETRIES,
            default_headers=default_headers,
        )

    return ChatOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model=model_name,
        timeout=LLM_REQUEST_TIMEOUT,
        max_retries=LLM_MAX_RETRIES,
        default_headers=default_headers,
        **({"temperature": temperature} if temperature is not None else {}),
    )


def get_provider_config(provider: str, is_expert: bool = False) -> Dict[str, Any]:
    """Get provider-specific configuration."""
    configs = {
        "openai": {
            "api_key": get_env_var("OPENAI_API_KEY", is_expert),
            "base_url": None,
        },
        "anthropic": {
            "api_key": get_env_var("ANTHROPIC_API_KEY", is_expert),
            "base_url": None,
        },
        "openrouter": {
            "api_key": get_env_var("OPENROUTER_API_KEY", is_expert),
            "base_url": "https://openrouter.ai/api/v1",
        },
        "openai-compatible": {
            "api_key": get_env_var("OPENAI_API_KEY", is_expert),
            "base_url": get_env_var("OPENAI_API_BASE", is_expert),
        },
        "gemini": {
            "api_key": get_env_var("GEMINI_API_KEY", is_expert),
            "base_url": None,
        },
        "deepseek": {
            "api_key": get_env_var("DEEPSEEK_API_KEY", is_expert),
            "base_url": "https://api.deepseek.com",
        },
    }
    config = configs.get(provider, {})
    if not config or not config.get("api_key"):
        raise ValueError(
            f"Missing required environment variable for provider: {provider}"
        )
    return config


def create_llm_client(
    provider: str,
    model_name: str,
    temperature: Optional[float] = None,
    is_expert: bool = False,
) -> BaseChatModel:
    """Create a language model client with appropriate configuration.

    Args:
        provider: The LLM provider to use
        model_name: Name of the model to use
        temperature: Optional temperature setting (0.0-2.0)
        is_expert: Whether this is an expert model (uses deterministic output)

    Returns:
        Configured language model client
    """
    config = get_provider_config(provider, is_expert)
    if not config:
        raise ValueError(f"Unsupported provider: {provider}")

    if is_expert and provider == "openai":
        model_name = select_expert_model(provider, model_name)
        if not model_name:
            raise ValueError("No suitable expert model available")

    logger.debug(
        "Creating LLM client with provider=%s, model=%s, temperature=%s, expert=%s",
        provider,
        model_name,
        temperature,
        is_expert,
    )

    # Get model configuration
    model_config = models_params.get(provider, {}).get(model_name, {})

    # Default to True for known providers that support temperature if not specified
    if "supports_temperature" not in model_config:
        model_config["supports_temperature"] = provider in known_temp_providers

    supports_temperature = model_config["supports_temperature"]
    supports_thinking = model_config.get("supports_thinking", False)

    # Handle temperature settings
    if is_expert:
        temp_kwargs = {"temperature": 0} if supports_temperature else {}
    elif supports_temperature:
        if temperature is None:
            temperature = 0.7
            cpm(
                "This model supports temperature argument but none was given. Setting default temperature to 0.7."
            )
        temp_kwargs = {"temperature": temperature}
    else:
        temp_kwargs = {}

    if supports_thinking:
        temp_kwargs = {"thinking": {"type": "enabled", "budget_tokens": 12000}}

    if provider == "deepseek":
        return create_deepseek_client(
            model_name=model_name,
            api_key=config["api_key"],
            base_url=config["base_url"],
            **temp_kwargs,
            is_expert=is_expert,
        )
    elif provider == "openrouter":
        return create_openrouter_client(
            model_name=model_name,
            api_key=config["api_key"],
            **temp_kwargs,
            is_expert=is_expert,
        )
    elif provider == "openai":
        openai_kwargs = {
            "api_key": config["api_key"],
            "model": model_name,
            **temp_kwargs,
        }
        if is_expert and model_config.get("supports_reasoning_effort", False):
            openai_kwargs["reasoning_effort"] = "high"

        return ChatOpenAI(
            **{
                **openai_kwargs,
                "timeout": LLM_REQUEST_TIMEOUT,
                "max_retries": LLM_MAX_RETRIES,
            }
        )
    elif provider == "anthropic":
        return ChatAnthropic(
            api_key=config["api_key"],
            model_name=model_name,
            timeout=LLM_REQUEST_TIMEOUT,
            max_retries=LLM_MAX_RETRIES,
            max_tokens=model_config.get("max_tokens", 64000),
            **temp_kwargs,
        )
    elif provider == "openai-compatible":
        return ChatOpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
            model=model_name,
            timeout=LLM_REQUEST_TIMEOUT,
            max_retries=LLM_MAX_RETRIES,
            **temp_kwargs,
        )
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            api_key=config["api_key"],
            model=model_name,
            timeout=LLM_REQUEST_TIMEOUT,
            max_retries=LLM_MAX_RETRIES,
            **temp_kwargs,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def initialize_llm(
    provider: str, model_name: str, temperature: float | None = None
) -> BaseChatModel:
    """Initialize a language model client based on the specified provider and model."""
    return create_llm_client(provider, model_name, temperature, is_expert=False)


def initialize_expert_llm(provider: str, model_name: str) -> BaseChatModel:
    """Initialize an expert language model client based on the specified provider and model."""
    return create_llm_client(provider, model_name, temperature=None, is_expert=True)


def validate_provider_env(provider: str) -> bool:
    """Check if the required environment variables for a provider are set."""
    required_vars = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "openai-compatible": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }
    key = required_vars.get(provider.lower())
    if key:
        return bool(os.getenv(key))
    return False
