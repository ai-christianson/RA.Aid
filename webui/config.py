import os
from typing import Dict, Any

def load_environment_status() -> Dict[str, bool]:
    """Load the environment status based on available API keys."""
    status = {
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "openrouter": bool(os.getenv("OPENROUTER_API_KEY")),
        "tavily": bool(os.getenv("TAVILY_API_KEY"))
    }
    return status

class WebUIConfig:
    """Configuration settings for the WebUI."""
    
    def __init__(
        self, 
        provider: str, 
        model: str, 
        research_only: bool, 
        cowboy_mode: bool, 
        hil: bool, 
        web_research_enabled: bool
    ):
        self.provider = provider
        self.model = model
        self.research_only = research_only
        self.cowboy_mode = cowboy_mode
        self.hil = hil
        self.web_research_enabled = web_research_enabled
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "provider": self.provider,
            "model": self.model,
            "research_only": self.research_only,
            "cowboy_mode": self.cowboy_mode,
            "hil": self.hil,
            "web_research_enabled": self.web_research_enabled
        } 