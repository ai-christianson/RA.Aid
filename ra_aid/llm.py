import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

def initialize_llm(provider: str, model_name: str) -> BaseChatModel:
    """Initialize a language model client based on the specified provider and model.

    Note: Environment variables must be validated before calling this function.
    Use validate_environment() to ensure all required variables are set.

    Args:
        provider: The LLM provider to use ('openai', 'anthropic', 'openrouter', 'openai-compatible')
        model_name: Name of the model to use

    Returns:
        BaseChatModel: Configured language model client

    Raises:
        ValueError: If the provider is not supported
    """
    if provider == "openai":
        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model_name,
        )
    elif provider == "anthropic":
        return ChatAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model_name=model_name,
        )
    elif provider == "openrouter":
        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=model_name,
        )
    elif provider == "openai-compatible":
        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
            model=model_name,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

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
