import os
from typing import Union
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage


class TokenLimitExceeded(Exception):
    """Raised when a message or chat history exceeds the model's token limit."""
    pass

MODEL_CONTEXT_LENGTHS = {
    # OpenAI models
    "gpt-4-turbo-preview": 128000,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4o": 128000,
    
    # Anthropic models  
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-5-sonnet-20241022": 200000,
    "claude-2.1": 200000,
    "claude-2.0": 100000,
    "claude-instant-1.2": 100000,
    
    # Default fallback
    "default": 4096
}

def check_message_tokens(content: Union[str, BaseMessage], model_name: str) -> int:
    """Check if content exceeds the token limit for the specified model.
    
    Uses a simple byte length heuristic estimation (1 token per 4 bytes).
    
    Args:
        content: String content or Message object to check tokens for
        model_name: Name of the model to check against
        
    Returns:
        int: Estimated number of tokens
        
    Raises:
        TokenLimitExceeded: If content exceeds the model's token limit
    """
    # Get model's context limit
    context_limit = get_model_context_limit(model_name)
    
    # Extract text content from message if needed
    if isinstance(content, BaseMessage):
        text = content.content
    else:
        text = content
        
    # Skip empty content
    if not text:
        return 0
        
    # Estimate tokens using byte length heuristic
    estimated_tokens = len(text.encode('utf-8')) // 4
    
    # Check against limit
    if estimated_tokens > context_limit:
        raise TokenLimitExceeded(
            f"Content exceeds {model_name} token limit of {context_limit:,} "
            f"(estimated {estimated_tokens:,} tokens)"
        )
        
    return estimated_tokens


def get_model_context_limit(model_name: str) -> int:
    """Get the context length limit for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Context length limit for the model. If model is unknown, returns default limit.
    """
    return MODEL_CONTEXT_LENGTHS.get(model_name, MODEL_CONTEXT_LENGTHS["default"])

def initialize_llm(provider: str, model_name: str, temperature: float | None = None) -> BaseChatModel:
    """Initialize a language model client based on the specified provider and model.

    Note: Environment variables must be validated before calling this function.
    Use validate_environment() to ensure all required variables are set.

    Args:
        provider: The LLM provider to use ('openai', 'anthropic', 'openrouter', 'openai-compatible')
        model_name: Name of the model to use
        temperature: Optional temperature setting for controlling randomness (0.0-2.0).
                    If not specified, provider-specific defaults are used.

    Returns:
        BaseChatModel: Configured language model client

    Raises:
        ValueError: If the provider is not supported or model name is invalid
        TokenLimitExceeded: If model's context length cannot be determined
    """
    # Validate we can determine context length for this model
    if get_model_context_limit(model_name) == MODEL_CONTEXT_LENGTHS["default"]:
        raise ValueError(f"Unknown context length for model: {model_name}")
    
    # Validate token handling works for this model
    check_message_tokens("", model_name)
    """Initialize a language model client based on the specified provider and model.

    Note: Environment variables must be validated before calling this function.
    Use validate_environment() to ensure all required variables are set.

    Args:
        provider: The LLM provider to use ('openai', 'anthropic', 'openrouter', 'openai-compatible')
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
