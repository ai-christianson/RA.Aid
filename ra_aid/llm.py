import os
from typing import Any, Dict, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from openai import OpenAI

from ra_aid.chat_models.deepseek_chat import ChatDeepseekReasoner
from ra_aid.console.formatting import cpm
from ra_aid.logging_config import get_logger
from ra_aid.model_detection import is_claude_37

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
    "ollama",
}

# Constants for API request configuration
LLM_REQUEST_TIMEOUT = 180
LLM_MAX_RETRIES = 5

logger = get_logger(__name__)


def get_env_var(name: str, expert: bool = False, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable with optional expert prefix and fallback."""
    prefix = "EXPERT_" if expert else ""
    value = os.getenv(f"{prefix}{name}")

    # If expert mode and no expert value, fall back to base value
    if expert and not value:
        value = os.getenv(name)

    return value if value is not None else default


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
            timeout=int(get_env_var(name="LLM_REQUEST_TIMEOUT", default=LLM_REQUEST_TIMEOUT)),
            max_retries=int(get_env_var(name="LLM_MAX_RETRIES", default=LLM_MAX_RETRIES)),
        )

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        temperature=0 if is_expert else (temperature if temperature is not None else 1),
        model=model_name,
        timeout=int(get_env_var(name="LLM_REQUEST_TIMEOUT", default=LLM_REQUEST_TIMEOUT)),
        max_retries=int(get_env_var(name="LLM_MAX_RETRIES", default=LLM_MAX_RETRIES)),
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
            timeout=int(get_env_var(name="LLM_REQUEST_TIMEOUT", default=LLM_REQUEST_TIMEOUT)),
            max_retries=int(get_env_var(name="LLM_MAX_RETRIES", default=LLM_MAX_RETRIES)),
            default_headers=default_headers,
        )

    return ChatOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model=model_name,
        timeout=int(get_env_var(name="LLM_REQUEST_TIMEOUT", default=LLM_REQUEST_TIMEOUT)),
        max_retries=int(get_env_var(name="LLM_MAX_RETRIES", default=LLM_MAX_RETRIES)),
        default_headers=default_headers,
        **({
            "temperature": temperature
        } if temperature is not None else {}),
    )


def create_ollama_client(
    model_name: str,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    with_thinking: bool = False,
    is_expert: bool = False,
    num_ctx: int = 262144,
) -> BaseChatModel:
    """Create a ChatOllama client.

    Args:
        model_name: Name of the model to use
        base_url: Base URL for the Ollama API
        temperature: Temperature for generation
        with_thinking: Whether to enable thinking patterns
        is_expert: Whether this is for an expert model
        num_ctx: Context window size for the model (default: 262144)

    Returns:
        ChatOllama instance
    """
    from langchain_ollama import ChatOllama

    # Default base URL if not provided
    if not base_url:
        base_url = "http://localhost:11434"

    # Create temperature kwargs if specified
    temp_kwargs = {}
    if temperature is not None:
        temp_kwargs["temperature"] = temperature

    # Return the ChatOllama instance with appropriate configuration
    return ChatOllama(
        model=model_name,
        base_url=base_url,
        num_ctx=num_ctx,
        timeout=int(get_env_var(name="LLM_REQUEST_TIMEOUT", default=LLM_REQUEST_TIMEOUT)),
        max_retries=int(get_env_var(name="LLM_MAX_RETRIES", default=LLM_MAX_RETRIES)),
        **temp_kwargs,
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
        "ollama": {
            "api_key": None,  # No API key needed for Ollama
            "base_url": get_env_var("OLLAMA_BASE_URL", is_expert, "http://localhost:11434"),
        },
    }
    config = configs.get(provider, {})
    if not config:
        raise ValueError(f"Unsupported provider: {provider}")
    
    # Ollama doesn't require an API key
    if provider != "ollama" and not config.get("api_key"):
        raise ValueError(
            f"Missing required environment variable for provider: {provider}"
        )
    return config


def get_model_default_temperature(provider: str, model_name: str) -> float:
    """Get the default temperature for a given model.
    
    Args:
        provider: The LLM provider
        model_name: Name of the model
        
    Returns:
        The default temperature value from models_params, or DEFAULT_TEMPERATURE if not specified
    """
    from ra_aid.models_params import models_params, DEFAULT_TEMPERATURE
    
    # Extract the model_config directly from models_params
    model_config = models_params.get(provider, {}).get(model_name, {})
    default_temp = model_config.get("default_temperature")
    
    # Return the model's default_temperature if it exists, otherwise DEFAULT_TEMPERATURE
    return default_temp if default_temp is not None else DEFAULT_TEMPERATURE


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

    model_config = models_params.get(provider, {}).get(model_name, {})

    # Default to True for known providers that support temperature if not specified
    if "supports_temperature" not in model_config:
        # Just set the value in the dictionary without modifying the source
        model_config = dict(model_config)  # Create a copy to avoid modifying the original
        # Set default value for supports_temperature based on known providers
        model_config["supports_temperature"] = provider in known_temp_providers

    supports_temperature = model_config.get("supports_temperature")
    supports_thinking = model_config.get("supports_thinking", False)

    other_kwargs = {}
    if is_claude_37(model_name):
        other_kwargs = {"max_tokens": 64000}

    # Handle temperature settings
    if is_expert:
        temp_kwargs = {"temperature": 0} if supports_temperature else {}
    elif supports_temperature:
        if temperature is None:
            # Use the model's default temperature from models_params
            temperature = get_model_default_temperature(provider, model_name)
            msg = f"This model supports temperature argument but none was given. Using model default temperature: {temperature}."
            
            try:
                # Try to log to the database, but continue even if it fails
                # Import repository classes directly to avoid circular imports
                from ra_aid.database.repositories.trajectory_repository import (
                    TrajectoryRepository,
                )
                from ra_aid.database.repositories.human_input_repository import (
                    HumanInputRepository,
                )
                from ra_aid.database.connection import get_db

                # Create repositories directly
                db = get_db()
                if db is not None:  # Check if db is initialized
                    trajectory_repo = TrajectoryRepository(db)
                    human_input_repo = HumanInputRepository(db)
                    human_input_id = human_input_repo.get_most_recent_id()
                    
                    if human_input_id is not None:  # Check if we have a valid human input
                        trajectory_repo.create(
                            step_data={
                                "message": msg,
                                "display_title": "Information",
                            },
                            record_type="info",
                            human_input_id=human_input_id,
                        )
            except Exception:
                # Silently continue if database operations fail
                # This handles testing scenarios where database might not be initialized
                pass
                
            # Always log to the console
            cpm(msg)
                
        temp_kwargs = {"temperature": temperature}
    else:
        temp_kwargs = {}

    thinking_kwargs = {}
    if supports_thinking:
        thinking_kwargs = {"thinking": {"type": "enabled", "budget_tokens": 12000}}

    if provider == "deepseek":
        return create_deepseek_client(
            model_name=model_name,
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            **temp_kwargs,
            **thinking_kwargs,
            is_expert=is_expert,
        )
    elif provider == "openrouter":
        return create_openrouter_client(
            model_name=model_name,
            api_key=config.get("api_key"),
            **temp_kwargs,
            **thinking_kwargs,
            is_expert=is_expert,
        )
    elif provider == "openai":
        openai_kwargs = {
            "api_key": config.get("api_key"),
            "model": model_name,
            **temp_kwargs,
        }
        if is_expert and model_config.get("supports_reasoning_effort", False):
            openai_kwargs["reasoning_effort"] = "high"

        return ChatOpenAI(
            **{
                **openai_kwargs,
                "timeout": int(get_env_var(name="LLM_REQUEST_TIMEOUT", default=LLM_REQUEST_TIMEOUT)),
                "max_retries": int(get_env_var(name="LLM_MAX_RETRIES", default=LLM_MAX_RETRIES)),
            }
        )
    elif provider == "anthropic":
        return ChatAnthropic(
            api_key=config.get("api_key"),
            model_name=model_name,
            timeout=int(get_env_var(name="LLM_REQUEST_TIMEOUT", default=LLM_REQUEST_TIMEOUT)),
            max_retries=int(get_env_var(name="LLM_MAX_RETRIES", default=LLM_MAX_RETRIES)),
            **temp_kwargs,
            **thinking_kwargs,
            **other_kwargs,
        )
    elif provider == "openai-compatible":
        return ChatOpenAI(
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            model=model_name,
            timeout=int(get_env_var(name="LLM_REQUEST_TIMEOUT", default=LLM_REQUEST_TIMEOUT)),
            max_retries=int(get_env_var(name="LLM_MAX_RETRIES", default=LLM_MAX_RETRIES)),
            **temp_kwargs,
            **thinking_kwargs,
        )
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            api_key=config.get("api_key"),
            model=model_name,
            timeout=int(get_env_var(name="LLM_REQUEST_TIMEOUT", default=LLM_REQUEST_TIMEOUT)),
            max_retries=int(get_env_var(name="LLM_MAX_RETRIES", default=LLM_MAX_RETRIES)),
            **temp_kwargs,
            **thinking_kwargs,
        )
    elif provider == "ollama":
        # Get num_ctx from config repository based on whether this is for expert model
        from ra_aid.database.repositories.config_repository import get_config_repository
        
        # Get the config repository through the context manager pattern
        config_repo = get_config_repository()
        
        # Get the appropriate num_ctx value from config repository
        num_ctx_key = "expert_num_ctx" if is_expert else "num_ctx"
        num_ctx_value = config_repo.get(num_ctx_key, 262144)
        
        return create_ollama_client(
            model_name=model_name,
            base_url=config.get("base_url"),
            temperature=temperature,
            with_thinking=bool(thinking_kwargs),
            is_expert=is_expert,
            num_ctx=num_ctx_value,
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
        "ollama": None,  # Ollama doesn't require any environment variables to be set
    }
    
    key = required_vars.get(provider.lower())
    if key is None:
        # For providers like Ollama that don't require any environment variables
        if provider.lower() == "ollama":
            # Always return True for Ollama, since we have a default base URL (http://localhost:11434)
            return True
        return False
    return bool(os.getenv(key))