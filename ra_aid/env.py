"""Environment validation utilities."""

import os
import sys
from dataclasses import dataclass
from typing import Tuple, List, Any

from ra_aid import print_error
from ra_aid.provider_strategy import ProviderFactory, ValidationResult

@dataclass
class ValidationResult:
    """Result of validation."""
    valid: bool
    missing_vars: List[str]

def validate_provider(provider: str) -> ValidationResult:
    """Validate provider configuration."""
    if not provider:
        return ValidationResult(valid=False, missing_vars=["No provider specified"])
    strategy = ProviderFactory.create(provider)
    if not strategy:
        return ValidationResult(valid=False, missing_vars=[f"Unknown provider: {provider}"])
    return strategy.validate()

def copy_base_to_expert_vars(base_provider: str, expert_provider: str) -> None:
    """Copy base provider environment variables to expert provider if not set.

    Args:
        base_provider: Base provider name
        expert_provider: Expert provider name
    """
    # Map of base to expert environment variables for each provider
    provider_vars = {
        'openai': {
            'OPENAI_API_KEY': 'EXPERT_OPENAI_API_KEY',
            'OPENAI_API_BASE': 'EXPERT_OPENAI_API_BASE'
        },
        'openai-compatible': {
            'OPENAI_API_KEY': 'EXPERT_OPENAI_API_KEY',
            'OPENAI_API_BASE': 'EXPERT_OPENAI_API_BASE'
        },
        'anthropic': {
            'ANTHROPIC_API_KEY': 'EXPERT_ANTHROPIC_API_KEY',
            'ANTHROPIC_MODEL': 'EXPERT_ANTHROPIC_MODEL'
        },
        'openrouter': {
            'OPENROUTER_API_KEY': 'EXPERT_OPENROUTER_API_KEY'
        },
        'gemini': {
            'GEMINI_API_KEY': 'EXPERT_GEMINI_API_KEY',
            'GEMINI_MODEL': 'EXPERT_GEMINI_MODEL'
        },
        'deepseek': {
            'DEEPSEEK_API_KEY': 'EXPERT_DEEPSEEK_API_KEY'
        }
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
        return ValidationResult(valid=False, missing_vars=[f"Unknown expert provider: {provider}"])

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

def validate_web_research() -> ValidationResult:
    """Validate web research configuration."""
    key = "TAVILY_API_KEY"
    return ValidationResult(
        valid=bool(os.environ.get(key)),
        missing_vars=[] if os.environ.get(key) else [f"{key} environment variable is not set"]
    )

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
    provider = args.provider if args and hasattr(args, 'provider') else None
    if not provider:
        sys.exit("No provider specified")

    # For non-Anthropic providers in research-only mode, model must be specified
    if provider != 'anthropic':
        model = args.model if hasattr(args, 'model') and args.model else None
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
        - web_research_enabled: Whether web research is enabled
        - web_research_missing: List of missing web research dependencies
    """
    # Initialize results
    expert_enabled = False
    expert_missing = []
    web_research_enabled = False
    web_research_missing = []

    # Validate web research dependencies
    tavily_key = os.environ.get('TAVILY_API_KEY')
    if not tavily_key:
        web_research_missing.append('TAVILY_API_KEY environment variable is not set')
    else:
        web_research_enabled = True

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
    if hasattr(args, 'research_only') and args.research_only:
        # Only validate provider and model when testing provider validation
        if hasattr(args, 'model') and args.model is None:
            validate_research_only_provider(args)
        return validate_research_only(args)

    # Initialize results
    expert_enabled = False
    expert_missing = []
    web_research_enabled = False
    web_research_missing = []

    # Get provider from args
    provider = args.provider if args and hasattr(args, 'provider') else None
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
    web_research_enabled = web_result.valid
    web_research_missing = web_result.missing_vars

    return expert_enabled, expert_missing, web_research_enabled, web_research_missing
