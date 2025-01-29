import os
from typing import Any, Dict, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from ra_aid.chat_models.deepseek_chat import ChatDeepseekReasoner
from ra_aid.logging_config import get_logger

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
            temperature=0
            if is_expert
            else (temperature if temperature is not None else 1),
            model=model_name,
        )

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        temperature=0 if is_expert else (temperature if temperature is not None else 1),
        model=model_name,
    )


def create_openrouter_client(
    model_name: str,
    api_key: str,
    temperature: Optional[float] = None,
    is_expert: bool = False,
) -> BaseChatModel:
    """Create OpenRouter client with appropriate configuration."""
    if model_name.startswith("deepseek/") and "deepseek-r1" in model_name.lower():
        return ChatDeepseekReasoner(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0
            if is_expert
            else (temperature if temperature is not None else 1),
            model=model_name,
        )

    return ChatOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model=model_name,
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
    return configs.get(provider, {})


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

    logger.debug(
        "Creating LLM client with provider=%s, model=%s, temperature=%s, expert=%s",
        provider,
        model_name,
        temperature,
        is_expert,
    )

    # Handle temperature settings
    if is_expert:
        temp_kwargs = {"temperature": 0}
    elif temperature is not None:
        temp_kwargs = {"temperature": temperature}
    elif provider == "openai-compatible":
        temp_kwargs = {"temperature": 0.3}
    else:
        temp_kwargs = {}

    if provider == "deepseek":
        return create_deepseek_client(
            model_name=model_name,
            api_key=config["api_key"],
            base_url=config["base_url"],
            temperature=temperature if not is_expert else 0,
            is_expert=is_expert,
        )
    elif provider == "openrouter":
        return create_openrouter_client(
            model_name=model_name,
            api_key=config["api_key"],
            temperature=temperature if not is_expert else 0,
            is_expert=is_expert,
        )
    elif provider == "openai":
        return ChatOpenAI(
            api_key=config["api_key"],
            model=model_name,
            **temp_kwargs,
        )
    elif provider == "anthropic":
        return ChatAnthropic(
            api_key=config["api_key"],
            model_name=model_name,
            **temp_kwargs,
        )
    elif provider == "openai-compatible":
        return ChatOpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
            model=model_name,
            **temp_kwargs,
        )
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            api_key=config["api_key"],
            model=model_name,
            **temp_kwargs,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def initialize_llm(
    provider: str, model_name: str, temperature: float | None = None
) -> BaseChatModel:
    """Initialize a language model client based on the specified provider and model."""
    return create_llm_client(provider, model_name, temperature, is_expert=False)


def initialize_expert_llm(
    provider: str = "openai", model_name: str = "o1"
) -> BaseChatModel:
    """Initialize an expert language model client based on the specified provider and model."""
    return create_llm_client(provider, model_name, temperature=None, is_expert=True)
