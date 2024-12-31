"""LLM utilities for RA.Aid."""

from typing import Optional, Dict, Any
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from ra_aid.logging_config import get_logger

logger = get_logger(__name__)

def create_model(
    provider: str,
    model_name: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs: Any
) -> BaseChatModel:
    """
    Create a language model instance based on provider and model name.
    
    Args:
        provider (str): The provider name (e.g., 'openai', 'anthropic')
        model_name (str): The model name (e.g., 'gpt-4', 'claude-3')
        temperature (float, optional): Model temperature. Defaults to 0.7.
        max_tokens (int, optional): Maximum tokens to generate. Defaults to None.
        **kwargs: Additional provider-specific arguments
        
    Returns:
        BaseChatModel: A language model instance
        
    Raises:
        ValueError: If provider is not supported
    """
    logger.info(f"Creating model: {provider}/{model_name}")
    
    model_kwargs = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        **kwargs
    }
    
    if provider == "anthropic":
        return ChatAnthropic(
            model_name=model_name,
            anthropic_api_key=None,  # Will use environment variable
            **model_kwargs
        )
    elif provider == "openai":
        return ChatOpenAI(
            model_name=model_name,
            openai_api_key=None,  # Will use environment variable
            **model_kwargs
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def initialize_llm(provider: str, model_name: str, **kwargs: Any) -> BaseChatModel:
    """Initialize a language model with the given configuration."""
    logger.info(f"Initializing LLM: {provider}/{model_name}")
    return create_model(provider, model_name, **kwargs)

def initialize_expert_llm(provider: str = "openai", model_name: str = "o1-preview") -> BaseChatModel:
    """Initialize an expert language model client based on the specified provider and model.

    Note: Environment variables must be validated before calling this function.
    Use validate_environment() to ensure all required variables are set.

    Args:
        provider: The LLM provider to use ('openai', 'anthropic', 'openrouter', 'openai-compatible').
                 Defaults to 'openai'.
        model_name: Name of the model to use. Defaults to 'o1-preview'.

    Returns:
        BaseChatModel: Configured expert language model client

    Raises:
        ValueError: If the provider is not supported
    """
    if provider == "openai":
        return ChatOpenAI(
            api_key=os.getenv("EXPERT_OPENAI_API_KEY"),
            model=model_name,
        )
    elif provider == "anthropic":
        return ChatAnthropic(
            api_key=os.getenv("EXPERT_ANTHROPIC_API_KEY"),
            model_name=model_name,
        )
    elif provider == "openrouter":
        return ChatOpenAI(
            api_key=os.getenv("EXPERT_OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=model_name,
        )
    elif provider == "openai-compatible":
        return ChatOpenAI(
            api_key=os.getenv("EXPERT_OPENAI_API_KEY"),
            base_url=os.getenv("EXPERT_OPENAI_API_BASE"),
            model=model_name,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
