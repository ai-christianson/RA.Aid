"""Utilities for detecting and working with specific model types."""

import litellm
from typing import Any, Dict, Optional

from langchain_core.language_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from ra_aid.config import DEFAULT_MODEL
from ra_aid.database.repositories.config_repository import get_config_repository
from ra_aid.logging_config import get_logger
from ra_aid.models_params import models_params, AgentBackendType, DEFAULT_AGENT_BACKEND


logger = get_logger(__name__)


def is_deepseek_v3(model_name: str) -> bool:
    """Check if the model is DeepSeek v3.

    Args:
        model_name: Name of the model

    Returns:
        True if the model is a DeepSeek v3 model, False otherwise
    """
    # DeepSeek v3 is identified by "deepseek-chat" in the model name
    # or when using "deepseek/deepseek-chat" through OpenRouter
    return (
        model_name == "deepseek-chat"
        or model_name.lower().endswith("/deepseek-chat")
        or "deepseek/deepseek-chat" in model_name.lower()
    )


def get_model_name_from_chat_model(model: Optional[BaseChatModel]) -> str:
    """Extract the model name from a BaseChatModel instance.

    Args:
        model: The BaseChatModel instance

    Returns:
        str: The model name extracted from the instance, or DEFAULT_MODEL if not found
    """
    if model is None:
        return DEFAULT_MODEL

    if hasattr(model, "metadata") and isinstance(model.metadata, dict):
        if "model_name" in model.metadata:
            return model.metadata["model_name"]

    # Fall back to other attributes
    if hasattr(model, "model"):
        return model.model
    elif hasattr(model, "model_name"):
        return model.model_name

    logger.warning(f"Could not extract model name from {model}, using DEFAULT_MODEL")
    return DEFAULT_MODEL


def get_provider_from_chat_model(model: Optional[BaseChatModel]) -> Optional[str]:
    """Extract the provider from a BaseChatModel instance's metadata if available.

    Args:
        model: The BaseChatModel instance

    Returns:
        Optional[str]: The provider name if found in metadata, None otherwise
    """
    if model is None:
        return None

    if hasattr(model, "metadata") and isinstance(model.metadata, dict):
        return model.metadata.get("provider")

    logger.warning(f"Could not extract provider name from chatmodel {model}, returning None")
    return None


def normalize_model_name(model_name: str) -> str:
    """
    Normalize a model name by removing provider prefixes and version suffixes.

    Args:
        model_name: The model name to normalize

    Returns:
        str: The normalized model name
    """
    # Specifically check for "models/" prefix
    if model_name.startswith("models/"):
        model_name = model_name[len("models/") :]
    # Remove provider prefix (e.g., "anthropic/", "google/")
    elif "/" in model_name:
        model_name = model_name.split("/", 1)[1]

    # Remove version suffix (e.g., ":free", ":v1")
    if ":" in model_name:
        model_name = model_name.split(":", 1)[0]

    return model_name


def is_claude_37(model: str) -> bool:
    """Check if the model is a Claude 3.7 model.

    Args:
        model: The model name to check

    Returns:
        bool: True if the model is a Claude 3.7 model, False otherwise
    """
    patterns = ["claude-3.7", "claude3.7", "claude-3-7"]
    return any(pattern in model for pattern in patterns)


def should_use_react_agent(model: BaseChatModel) -> bool:
    """
    Determine if we should use create_react_agent vs CiaynAgent based on model capabilities.

    Args:
        model: The language model to check

    Returns:
        bool: True if we should use create_react_agent, False if we should use CiaynAgent
    """
    use_react_agent = False
    model_name = get_model_name_from_chat_model(model)
    provider = get_provider_from_chat_model(model)

    try:
        supports_function_calling = litellm.supports_function_calling(
            model=model_name, custom_llm_provider=provider
        )
        use_react_agent = supports_function_calling
        logger.debug(
            f"Model {model_name} (normalized: {model_name}) supports_function_calling: {supports_function_calling}"
        )
    except Exception as e:
        logger.warning(f"Error checking function calling support: {e}.")

    try:
        if not provider:
            provider = get_config_repository().get("provider", "anthropic")
        provider_config = models_params.get(provider, {})
        model_config = provider_config.get(
            model_name, provider_config.get(model_name, {})
        )

        # If there's a specific backend configured, override the detection result
        if "default_backend" in model_config:
            configured_backend = model_config.get(
                "default_backend", DEFAULT_AGENT_BACKEND
            )

            use_react_agent = configured_backend == AgentBackendType.CREATE_REACT_AGENT
            logger.debug(
                f"Overriding agent backend selection based on config: {use_react_agent}"
            )
    except Exception as e:
        logger.warning(
            f"Error checking model config: {e}. Using function calling detection."
        )

    return use_react_agent


def model_name_has_claude(model_name: str) -> bool:
    """Check if a model name contains 'claude'.

    Args:
        model_name: The model name to check

    Returns:
        bool: True if the model name contains 'claude'
    """
    return model_name and "claude" in model_name.lower()


def is_anthropic_claude(config: Dict[str, Any]) -> bool:
    """Check if the provider and model name indicate an Anthropic Claude model.

    Args:
        config: Configuration dictionary containing provider and model information

    Returns:
        bool: True if this is an Anthropic Claude model
    """
    # For backwards compatibility, allow passing of config directly
    provider = config.get("provider", "")
    model_name = config.get("model", "")
    result = (
        provider.lower() == "anthropic"
        and model_name
        and "claude" in model_name.lower()
    ) or (
        provider.lower() == "openrouter"
        and model_name.lower().startswith("anthropic/claude-")
    )
    return result
