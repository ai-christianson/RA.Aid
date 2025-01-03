from collections import deque
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Dict, Deque, Optional, Tuple, Union
import time

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
            
    def wait_for_capacity(self, provider: str, tokens: int) -> float:
        """Wait until rate limit capacity is available.
        
        Args:
            provider: Provider to wait for capacity
            tokens: Number of tokens needed
            
        Returns:
            Time waited in seconds
        """
        with self.lock:
            usage = self._get_provider_usage(provider)
            limit = self.rate_limits.get(provider, self.rate_limits["default"])
            
            # Get current usage and clean up expired entries
            current = usage.get_current_usage()
            
            # If we have capacity, return immediately
            if current + tokens <= limit:
                return 0.0
                
            # Calculate required wait based on oldest entry that needs to expire
            required_tokens = (current + tokens) - limit
            total_wait = 0.0
            
            while required_tokens > 0:
                # Get oldest entry still in window
                if not usage.usage_history:
                    break
                    
                oldest = usage.usage_history[0]
                age = (datetime.now() - oldest.timestamp).total_seconds()
                wait_time = self.window_seconds - age
                
                if wait_time <= 0:
                    # Entry expired, clean up and recheck
                    usage._cleanup_expired()
                    current = usage.get_current_usage()
                    required_tokens = (current + tokens) - limit
                    continue
                
                # Wait for oldest entry to expire
                time.sleep(wait_time)
                total_wait += wait_time
                
                # Cleanup and recalculate required tokens
                usage._cleanup_expired() 
                current = usage.get_current_usage()
                required_tokens = (current + tokens) - limit
            
            return total_wait
    
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
