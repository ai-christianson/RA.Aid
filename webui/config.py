"""Configuration management for the WebUI."""

import os
from typing import Dict, Any
from webui import logger, log_function

# Get logger for this module
config_logger = logger.getChild("config")

@log_function(config_logger)
def load_environment_status() -> Dict[str, bool]:
    """Load the environment status based on available API keys."""
    # Debug: Print raw environment variable values
    config_logger.debug(f"Raw ANTHROPIC_API_KEY: {os.getenv('ANTHROPIC_API_KEY')}")
    config_logger.debug(f"Raw OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')}")
    config_logger.debug(f"Raw OPENROUTER_API_KEY: {os.getenv('OPENROUTER_API_KEY')}")
    
    status = {
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "openrouter": bool(os.getenv("OPENROUTER_API_KEY"))
    }
    
    # Debug: Print status after bool conversion
    config_logger.debug(f"Status after bool conversion: {status}")
    return status

class WebUIConfig:
    """Configuration for the WebUI."""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4", 
                 research_only: bool = False, cowboy_mode: bool = False,
                 hil: bool = True, web_research_enabled: bool = True):
        """Initialize WebUI configuration."""
        self.provider = provider
        self.model = model
        self.research_only = research_only
        self.cowboy_mode = cowboy_mode
        self.hil = hil
        self.web_research_enabled = web_research_enabled
        
        config_logger.info(f"Initializing WebUIConfig: provider={provider}, model={model}")
        config_logger.debug(f"Additional settings: research_only={research_only}, "
                          f"cowboy_mode={cowboy_mode}, hil={hil}, "
                          f"web_research_enabled={web_research_enabled}")
        
        # Validate configuration
        self.validate()
    
    @log_function(config_logger)
    def validate(self) -> None:
        """Validate the configuration."""
        if not self.provider:
            config_logger.error("Provider not specified")
            raise ValueError("Provider must be specified")
            
        if not self.model:
            config_logger.error("Model not specified")
            raise ValueError("Model must be specified")
            
        if self.provider not in ["openai", "anthropic", "openrouter"]:
            config_logger.error(f"Invalid provider: {self.provider}")
            raise ValueError(f"Invalid provider: {self.provider}")
            
        # Validate API keys are set
        api_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "openrouter": os.getenv("OPENROUTER_API_KEY")
        }
        
        if not api_keys[self.provider]:
            config_logger.error(f"API key not set for provider: {self.provider}")
            raise ValueError(f"API key not set for provider: {self.provider}")
            
        config_logger.info("Configuration validation successful")
            
    def __str__(self) -> str:
        """Return string representation of config."""
        return f"WebUIConfig(provider={self.provider}, model={self.model}, research_only={self.research_only}, cowboy_mode={self.cowboy_mode}, hil={self.hil}, web_research_enabled={self.web_research_enabled})" 