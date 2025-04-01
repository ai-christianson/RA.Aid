"""Environment validation utilities."""

import os
import sys
from typing import Any, List, Tuple
from dataclasses import dataclass

from ra_aid.provider_strategy import ProviderFactory, ValidationResult


@dataclass
class WebResearchValidationResult:
    """Result of web research validation."""
    jina_valid: bool  # Is Jina API Key present?
    tavily_valid: bool # Is Tavily API Key present?
    any_valid: bool   # Is at least one web search key present?
    missing_vars: List[str]


def validate_web_research() -> WebResearchValidationResult:
    """Validate web research environment variables for Jina and Tavily.

    Returns:
        WebResearchValidationResult indicating which keys are valid and any missing variables.
    """
    missing = []
    jina_key = "JINA_API_KEY"
    tavily_key = "TAVILY_API_KEY"

    jina_valid = bool(os.environ.get(jina_key))
    tavily_valid = bool(os.environ.get(tavily_key))
    any_valid = jina_valid or tavily_valid

    if not jina_valid:
        missing.append(f"{jina_key} environment variable is not set (required for deep search)")
    if not tavily_valid:
        missing.append(f"{tavily_key} environment variable is not set (required for quick search)")

    return WebResearchValidationResult(
        jina_valid=jina_valid,
        tavily_valid=tavily_valid,
        any_valid=any_valid,
        missing_vars=missing if not any_valid else [] # Only report missing if *none* are valid for overall status
    )


def validate_provider(provider: str) -> ValidationResult:
    """Validate provider configuration."""
    if not provider:
        return ValidationResult(valid=False, missing_vars=["No provider specified"])
    strategy = ProviderFactory.create(provider)
    if not strategy:
        return ValidationResult(
            valid=False, missing_vars=[f"Unknown provider: {provider}"]
        )
    return strategy.validate()


def copy_base_to_expert_vars(base_provider: str, expert_provider: str) -> None:
    """Copy base provider environment variables to expert provider if not set.

    Args:
        base_provider: Base provider name
        expert_provider: Expert provider name
    """
    # Map of base to expert environment variables for each provider
    provider_vars = {
        "openai": {
            "OPENAI_API_KEY": "EXPERT_OPENAI_API_KEY",
            "OPENAI_API_BASE": "EXPERT_OPENAI_API_BASE",
        },
        "openai-compatible": {
            "OPENAI_API_KEY": "EXPERT_OPENAI_API_KEY",
            "OPENAI_API_BASE": "EXPERT_OPENAI_API_BASE",
        },
        "anthropic": {
            "ANTHROPIC_API_KEY": "EXPERT_ANTHROPIC_API_KEY",
            "ANTHROPIC_MODEL": "EXPERT_ANTHROPIC_MODEL",
        },
        "openrouter": {"OPENROUTER_API_KEY": "EXPERT_OPENROUTER_API_KEY"},
        "gemini": {
            "GEMINI_API_KEY": "EXPERT_GEMINI_API_KEY",
            "GEMINI_MODEL": "EXPERT_GEMINI_MODEL",
        },
        "deepseek": {"DEEPSEEK_API_KEY": "EXPERT_DEEPSEEK_API_KEY"},
        "fireworks": {"FIREWORKS_API_KEY": "EXPERT_FIREWORKS_API_KEY"},
        "groq": {"GROQ_API_KEY": "EXPERT_GROQ_API_KEY"},
        "ollama": {"OLLAMA_BASE_URL": "EXPERT_OLLAMA_BASE_URL"},
    }

    # Get the variables to copy based on the expert provider
    vars_to_copy = provider_vars.get(expert_provider, {})
    for base_var, expert_var in vars_to_copy.items():
        # Only copy if expert var is not set and base var exists
        if not os.environ.get(expert_var) and os.environ.get(base_var):
            os.environ[expert_var] = os.environ[base_var]


def validate_expert_provider(provider: str) -> ValidationResult:
    """Validate expert provider configuration with fallback."""
    if not provider:
        return ValidationResult(valid=True, missing_vars=[])

    strategy = ProviderFactory.create(provider)
    if not strategy:
        return ValidationResult(
            valid=False, missing_vars=[f"Unknown expert provider: {provider}"]
        )

    # Copy base vars to expert vars for fallback
    copy_base_to_expert_vars(provider, provider)

    # Validate expert configuration
    result = strategy.validate()
    missing = []

    for var in result.missing_vars:
        key = var.split()[0]  # Get the key name without the error message
        expert_key = f"EXPERT_{key}"
        if not os.environ.get(expert_key):
            missing.append(f"{expert_key} environment variable is not set")

    return ValidationResult(valid=len(missing) == 0, missing_vars=missing)


def check_web_research_env() -> List[str]:
    """Check if web research environment variables are set (Jina or Tavily)."""
    web_research_missing = []
    jina_key = "JINA_API_KEY"
    tavily_key = "TAVILY_API_KEY"

    if not os.environ.get(jina_key) and not os.environ.get(tavily_key):
        web_research_missing.append(f"Neither {jina_key} nor {tavily_key} environment variable is set. Web search disabled.")
    elif not os.environ.get(jina_key):
         web_research_missing.append(f"{jina_key} not set (deep search unavailable).") # Informative message
    elif not os.environ.get(tavily_key):
         web_research_missing.append(f"{tavily_key} not set (quick search unavailable).") # Informative message

    return web_research_missing


def check_env() -> Tuple[bool, List[str], bool, List[str]]:
    """
    Check if required environment variables are set.
    
    Returns:
        Tuple containing:
        - bool: Whether all required env vars are set
        - List[str]: List of missing required env vars
        - bool: Whether all optional env vars are set
        - List[str]: List of missing optional env vars
    """
    required_missing = []
    optional_missing = []

    # Check required env vars
    # Making OPENAI optional for now if other providers are primary
    # required_vars = ["OPENAI_API_KEY"]
    # for var in required_vars:
    #     if not os.environ.get(var):
    #         required_missing.append(f"{var} environment variable is not set")

    # Check optional env vars (including various provider keys and web search keys)
    optional_vars = [
        "OPENAI_API_KEY", # Now optional
        "ANTHROPIC_API_KEY",
        "JINA_API_KEY",
        "TAVILY_API_KEY",
        # Add other provider keys as needed if they should be considered optional overall
        "OPENROUTER_API_KEY",
        "GEMINI_API_KEY",
        "DEEPSEEK_API_KEY",
        "FIREWORKS_API_KEY",
        "GROQ_API_KEY",
    ]
    at_least_one_web_key = False
    for var in optional_vars:
        if not os.environ.get(var):
            optional_missing.append(f"{var} environment variable is not set")
        elif var == "JINA_API_KEY" or var == "TAVILY_API_KEY":
            at_least_one_web_key = True

    # Refine web search missing message based on combined check
    web_research_specific_missing = check_web_research_env()
    if "Neither" in "".join(web_research_specific_missing): # If neither key is set
         # Remove individual key messages from optional_missing if the combined message exists
         optional_missing = [m for m in optional_missing if "JINA_API_KEY" not in m and "TAVILY_API_KEY" not in m]
         optional_missing.extend(web_research_specific_missing) # Add the combined "Neither..." message

    # Consider required empty for now, adjust logic if specific providers *must* be present
    all_required = len(required_missing) == 0
    all_optional = len(optional_missing) == 0

    return all_required, required_missing, all_optional, optional_missing


def print_missing_dependencies(missing_vars: List[str]) -> None:
    """Print missing dependencies and exit."""
    for var in missing_vars:
        print(f"Error: {var}", file=sys.stderr)
    sys.exit(1)


def validate_research_only_provider(args: Any) -> None:
    """Validate provider and model for research-only mode.

    Args:
        args: Arguments containing provider and expert provider settings

    Raises:
        SystemExit: If provider or model validation fails
    """
    # Get provider from args
    provider = args.provider if args and hasattr(args, "provider") else None
    if not provider:
        sys.exit("No provider specified")

    # For non-Anthropic providers in research-only mode, model must be specified
    if provider != "anthropic":
        model = args.model if hasattr(args, "model") and args.model else None
        if not model:
            sys.exit("Model is required for non-Anthropic providers")


def validate_research_only(args: Any) -> tuple[bool, list[str], bool, list[str]]:
    """Validate environment variables for research-only mode.

    Args:
        args: Arguments containing provider and expert provider settings

    Returns:
        Tuple containing:
        - expert_enabled: Whether expert mode is enabled
        - expert_missing: List of missing expert dependencies
        - web_research_enabled: Whether web research is enabled (at least one key)
        - web_research_missing: List of missing web research dependencies
    """
    # Initialize results
    expert_enabled = False # Research-only typically doesn't use expert loop? Check this assumption.
    expert_missing = []
    web_research_enabled = False
    web_research_missing = []

    # Validate web research dependencies
    jina_key = os.environ.get("JINA_API_KEY")
    tavily_key = os.environ.get("TAVILY_API_KEY")

    if jina_key or tavily_key:
        web_research_enabled = True
        if not jina_key:
             web_research_missing.append("JINA_API_KEY environment variable is not set (deep search unavailable)")
        if not tavily_key:
             web_research_missing.append("TAVILY_API_KEY environment variable is not set (quick search unavailable)")
    else:
         web_research_missing.append("Neither JINA_API_KEY nor TAVILY_API_KEY environment variable is set.")


    return expert_enabled, expert_missing, web_research_enabled, web_research_missing


def validate_environment(args: Any) -> tuple[bool, list[str], bool, list[str]]:
    """Validate environment variables for providers and web research tools.

    Args:
        args: Arguments containing provider and expert provider settings

    Returns:
        Tuple containing:
        - expert_enabled: Whether expert mode is enabled
        - expert_missing: List of missing expert dependencies
        - web_research_enabled: Whether web research is enabled
        - web_research_missing: List of missing web research dependencies
    """
    # For research-only mode, use separate validation
    if hasattr(args, "research_only") and args.research_only:
        # Only validate provider and model when testing provider validation
        if hasattr(args, "model") and args.model is None:
            validate_research_only_provider(args)
        return validate_research_only(args)

    # Initialize results
    expert_enabled = False
    expert_missing = []
    web_research_enabled = False
    web_research_missing = []

    # Get provider from args
    provider = args.provider if args and hasattr(args, "provider") else None
    if not provider:
        sys.exit("No provider specified")

    # Validate main provider
    strategy = ProviderFactory.create(provider, args)
    if not strategy:
        sys.exit(f"Unknown provider: {provider}")

    result = strategy.validate(args)
    if not result.valid:
        print_missing_dependencies(result.missing_vars)

    # Handle expert provider if enabled
    if args.expert_provider:
        # Copy base variables to expert if not set
        copy_base_to_expert_vars(provider, args.expert_provider)

        # Validate expert provider
        expert_strategy = ProviderFactory.create(args.expert_provider, args)
        if not expert_strategy:
            sys.exit(f"Unknown expert provider: {args.expert_provider}")

        expert_result = expert_strategy.validate(args)
        expert_missing = expert_result.missing_vars
        expert_enabled = len(expert_missing) == 0

        # If expert validation failed, try to copy base variables again and revalidate
        if not expert_enabled:
            copy_base_to_expert_vars(provider, args.expert_provider)
            expert_result = expert_strategy.validate(args)
            expert_missing = expert_result.missing_vars
            expert_enabled = len(expert_missing) == 0

    # Validate web research dependencies
    web_result = validate_web_research()
    web_research_enabled = web_result.any_valid
    web_research_missing = web_result.missing_vars

    return expert_enabled, expert_missing, web_research_enabled, web_research_missing
