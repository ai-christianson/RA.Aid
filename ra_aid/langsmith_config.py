"""LangSmith configuration for tracing and monitoring."""

from dataclasses import dataclass
from typing import Optional

@dataclass
class LangSmithConfig:
    """Configuration for LangSmith integration.
    
    Args:
        api_key: LangSmith API key
        project_name: Name of the project in LangSmith
        trace_enabled: Whether to enable tracing
    """
    api_key: Optional[str] = None
    project_name: str = "ra-aid"
    trace_enabled: bool = True

    def is_configured(self) -> bool:
        """Check if LangSmith is properly configured."""
        return bool(self.api_key and self.trace_enabled)
