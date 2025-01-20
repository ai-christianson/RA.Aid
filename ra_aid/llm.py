import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI



def initialize_llm(provider: str, model_name: str, temperature: float | None = None) -> BaseChatModel:
    """Initialize a language model client based on the specified provider and model.

    Note: Environment variables must be validated before calling this function.
    Use validate_environment() to ensure all required variables are set.

    Args:
        provider: The LLM provider to use ('openai', 'anthropic', 'openrouter', 'openai-compatible', 'gemini')
        model_name: Name of the model to use
        temperature: Optional temperature setting for controlling randomness (0.0-2.0).
                    If not specified, provider-specific defaults are used.

    Returns:
        BaseChatModel: Configured language model client

    Raises:
        ValueError: If the provider is not supported
    """
    if provider == "openai":
        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model_name,
            **({"temperature": temperature} if temperature is not None else {})
        )
    elif provider == "anthropic":
        return ChatAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model_name=model_name,
            **({"temperature": temperature} if temperature is not None else {})
        )
    elif provider == "openrouter":
        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=model_name,
            **({"temperature": temperature} if temperature is not None else {})
        )
    elif provider == "openai-compatible":
        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
            temperature=temperature if temperature is not None else 0.3,
            model=model_name,
        )
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            model=model_name,
            **({"temperature": temperature} if temperature is not None else {})
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def initialize_expert_llm(provider: str = "openai", model_name: str = "o1") -> BaseChatModel:
    """Initialize an expert language model client based on the specified provider and model.

    Note: Environment variables must be validated before calling this function.
    Use validate_environment() to ensure all required variables are set.

    Args:
        provider: The LLM provider to use ('openai', 'anthropic', 'openrouter', 'openai-compatible', 'gemini').
                 Defaults to 'openai'.
        model_name: Name of the model to use. Defaults to 'o1'.

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
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            api_key=os.getenv("EXPERT_GEMINI_API_KEY"),
            model=model_name,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
