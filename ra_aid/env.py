"""Environment validation utilities."""

import os
import sys
from dataclasses import dataclass
from typing import Tuple, List

from ra_aid import print_error

@dataclass
class ProviderConfig:
    """Configuration for a provider."""
    key_name: str
    base_required: bool = False

PROVIDER_CONFIGS = {
    "anthropic": ProviderConfig("ANTHROPIC_API_KEY", base_required=True),
    "openai": ProviderConfig("OPENAI_API_KEY", base_required=True),
    "openrouter": ProviderConfig("OPENROUTER_API_KEY", base_required=True),
    "openai-compatible": ProviderConfig("OPENAI_API_KEY", base_required=True),
}

def validate_environment(args) -> Tuple[bool, List[str], bool, List[str]]:
    """Validate required environment variables and dependencies.
    
    Args:
        args: The parsed command line arguments containing:
            - provider: The main LLM provider
            - expert_provider: The expert LLM provider
        
    Returns:
        Tuple containing:
            - bool: Whether expert mode is enabled
            - List[str]: List of missing expert configuration items
            - bool: Whether web research is enabled
            - List[str]: List of missing web research configuration items
            
    Raises:
        SystemExit: If required base environment variables are missing
    """
    missing = []
    provider = args.provider
    expert_provider = args.expert_provider

    # Check API keys based on provider configs
    if provider in PROVIDER_CONFIGS:
        config = PROVIDER_CONFIGS[provider]
        if config.base_required and not os.environ.get(config.key_name):
            missing.append(f'{config.key_name} environment variable is not set')
            
    # Special case for openai-compatible needing base URL
    if provider == "openai-compatible" and not os.environ.get('OPENAI_API_BASE'):
        missing.append('OPENAI_API_BASE environment variable is not set')

    expert_missing = []
    if expert_provider in PROVIDER_CONFIGS:
        config = PROVIDER_CONFIGS[expert_provider]
        expert_key = f'EXPERT_{config.key_name}'
        expert_key_missing = not os.environ.get(expert_key)
        
        # Try fallback to base key for expert provider
        fallback_available = os.environ.get(config.key_name)
        if expert_key_missing and fallback_available:
            os.environ[expert_key] = os.environ[config.key_name]
            expert_key_missing = False
        
        # Only add to missing list if still missing after fallback attempt    
        if expert_key_missing:
            expert_missing.append(f'{expert_key} environment variable is not set')
            
        # Special case for openai-compatible expert needing base URL 
        if expert_provider == "openai-compatible":
            expert_base = 'EXPERT_OPENAI_API_BASE'
            base_missing = not os.environ.get(expert_base)
            base_fallback = os.environ.get('OPENAI_API_BASE')
            
            if base_missing and base_fallback:
                os.environ[expert_base] = os.environ['OPENAI_API_BASE']
                base_missing = False
                
            if base_missing:
                expert_missing.append(f'{expert_base} environment variable is not set')

    # If main keys missing, we must exit immediately
    if missing:
        print_error("Missing required dependencies:")
        for item in missing:
            print_error(f"- {item}")
        sys.exit(1)

    # If expert keys missing, we disable expert tools instead of exiting
    expert_enabled = True
    if expert_missing:
        expert_enabled = False

    # Check web research dependencies
    web_research_missing = []
    web_research_enabled = False
    
    if not os.environ.get('TAVILY_API_KEY'):
        web_research_missing.append('TAVILY_API_KEY environment variable is not set')
    else:
        web_research_enabled = True

    return expert_enabled, expert_missing, web_research_enabled, web_research_missing
