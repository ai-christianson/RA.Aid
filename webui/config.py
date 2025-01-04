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
                 hil: bool = True, web_research_enabled: bool = True,
                 temperature: float = 0.7, max_tokens: int = 2000):
        """Initialize WebUI configuration."""
        # Clean up model name if it has provider prefix
        if '/' in model:
            model = model.split('/')[-1]
            config_logger.debug(f"Cleaned up model name to: {model}")
            
        self._config = {
            "provider": provider,
            "model": model,
            "research_only": research_only,
            "cowboy_mode": cowboy_mode,
            "hil": hil,
            "web_research_enabled": web_research_enabled,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        config_logger.info(f"Initializing WebUIConfig: provider={provider}, model={model}")
        config_logger.debug(f"Additional settings: research_only={research_only}, "
                          f"cowboy_mode={cowboy_mode}, hil={hil}, "
                          f"web_research_enabled={web_research_enabled}, "
                          f"temperature={temperature}, max_tokens={max_tokens}")
        
        # Validate configuration
        self.validate()
    
    def __getitem__(self, key: str) -> Any:
        """Make the config dict-like."""
        return self._config[key]
    
    def __iter__(self):
        """Support iteration over config keys."""
        return iter(self._config)
    
    def __contains__(self, key: str) -> bool:
        """Support 'in' operator."""
        return key in self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get method."""
        return self._config.get(key, default)
    
    def copy(self) -> dict:
        """Return a copy of the config dict."""
        return self._config.copy()
    
    @property
    def provider(self) -> str:
        return self._config["provider"]
        
    @property
    def model(self) -> str:
        return self._config["model"]
        
    @property
    def research_only(self) -> bool:
        return self._config["research_only"]
        
    @property
    def cowboy_mode(self) -> bool:
        return self._config["cowboy_mode"]
        
    @property
    def hil(self) -> bool:
        return self._config["hil"]
        
    @property
    def web_research_enabled(self) -> bool:
        return self._config["web_research_enabled"]
    
    @property
    def temperature(self) -> float:
        return self._config["temperature"]
        
    @property
    def max_tokens(self) -> int:
        return self._config["max_tokens"]
    
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
        return f"WebUIConfig({', '.join(f'{k}={v}' for k, v in self._config.items())})" 