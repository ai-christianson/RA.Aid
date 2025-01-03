from collections import deque
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Dict, Deque, Optional, Tuple

# Default rate limits in tokens per minute
DEFAULT_RATE_LIMITS = {
    "openai": 300_000,
    "anthropic": 100_000,
    "openrouter": 50_000,
    "default": 10_000
}

@dataclass
class TokenUsageEntry:
    """Represents a single token usage entry with timestamp."""
    timestamp: datetime
    tokens: int

class TokenUsage:
    """Tracks token usage within a sliding time window."""
    
    def __init__(self, window_seconds: int = 60):
        """Initialize token usage tracker.
        
        Args:
            window_seconds: The time window in seconds to track usage
        """
        self.window_seconds = window_seconds
        self.usage_history: Deque[TokenUsageEntry] = deque()
        self.lock = Lock()
        
    def add_usage(self, tokens: int) -> None:
        """Add token usage entry with current timestamp.
        
        Args:
            tokens: Number of tokens used
        """
        with self.lock:
            self.usage_history.append(TokenUsageEntry(
                timestamp=datetime.now(),
                tokens=tokens
            ))
            self._cleanup_expired()
    
    def get_current_usage(self) -> int:
        """Get total token usage within current window.
        
        Returns:
            Total tokens used in current window
        """
        with self.lock:
            self._cleanup_expired()
            return sum(entry.tokens for entry in self.usage_history)
    
    def _cleanup_expired(self) -> None:
        """Remove usage entries outside current window."""
        now = datetime.now()
        while self.usage_history:
            if (now - self.usage_history[0].timestamp).total_seconds() > self.window_seconds:
                self.usage_history.popleft()
            else:
                break

class RateLimiter:
    """Rate limiter for API token usage."""
    
    def __init__(
        self,
        window_seconds: int = 60,
        custom_limits: Optional[Dict[str, int]] = None
    ):
        """Initialize rate limiter.
        
        Args:
            window_seconds: Time window in seconds for rate limiting
            custom_limits: Optional custom rate limits per provider
        """
        self.provider_usage: Dict[str, TokenUsage] = {}
        self.rate_limits = DEFAULT_RATE_LIMITS.copy()
        if custom_limits:
            self.rate_limits.update(custom_limits)
        self.window_seconds = window_seconds
        self.lock = Lock()
    
    def check_rate_limit(self, provider: str, tokens: int) -> Tuple[bool, int]:
        """Check if token usage would exceed rate limit.
        
        Args:
            provider: Provider to check rate limit for
            tokens: Number of tokens to check
            
        Returns:
            Tuple of (would_exceed, current_usage)
        """
        with self.lock:
            usage = self._get_provider_usage(provider)
            current = usage.get_current_usage()
            limit = self.rate_limits.get(provider, self.rate_limits["default"])
            return (current + tokens > limit, current)
    
    def update_usage(self, provider: str, tokens: int) -> None:
        """Update token usage for provider.
        
        Args:
            provider: Provider to update usage for
            tokens: Number of tokens used
        """
        with self.lock:
            usage = self._get_provider_usage(provider)
            usage.add_usage(tokens)
    
    def _get_provider_usage(self, provider: str) -> TokenUsage:
        """Get or create TokenUsage tracker for provider.
        
        Args:
            provider: Provider to get usage tracker for
            
        Returns:
            TokenUsage tracker for provider
        """
        if provider not in self.provider_usage:
            self.provider_usage[provider] = TokenUsage(self.window_seconds)
        return self.provider_usage[provider]
