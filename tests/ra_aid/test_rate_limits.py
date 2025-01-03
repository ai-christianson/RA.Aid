import threading
import time
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from ra_aid.rate_limits import TokenUsage, RateLimiter, DEFAULT_RATE_LIMITS


def test_token_usage_tracking():
    """Test basic token usage tracking functionality."""
    usage = TokenUsage(window_seconds=60)
    
    # Add some usage
    usage.add_usage(100)
    usage.add_usage(200)
    
    assert usage.get_current_usage() == 300


def test_rate_limiter_basic():
    """Test basic rate limiter functionality."""
    limiter = RateLimiter()
    
    # Check initial state
    would_exceed, current = limiter.check_rate_limit("openai", 1000)
    assert not would_exceed
    assert current == 0
    
    # Add some usage
    limiter.update_usage("openai", 1000)
    would_exceed, current = limiter.check_rate_limit("openai", DEFAULT_RATE_LIMITS["openai"])
    assert would_exceed
    assert current == 1000


def test_window_sliding():
    """Test that usage window slides properly."""
    window_seconds = 1
    usage = TokenUsage(window_seconds=window_seconds)
    
    # Add usage and wait for window to pass
    usage.add_usage(100)
    time.sleep(window_seconds + 0.1)
    
    assert usage.get_current_usage() == 0


def test_thread_safety():
    """Test thread-safe operation."""
    limiter = RateLimiter()
    num_threads = 10
    iterations = 100
    
    def worker():
        for _ in range(iterations):
            limiter.update_usage("openai", 1)
    
    threads = [
        threading.Thread(target=worker)
        for _ in range(num_threads)
    ]
    
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    _, current = limiter.check_rate_limit("openai", 0)
    assert current == num_threads * iterations


def test_custom_rate_limits():
    """Test custom rate limit configuration."""
    custom_limits = {
        "openai": 1000,
        "custom_provider": 500
    }
    
    limiter = RateLimiter(custom_limits=custom_limits)
    
    # Test custom OpenAI limit
    would_exceed, _ = limiter.check_rate_limit("openai", 1001)
    assert would_exceed
    
    # Test custom provider limit
    would_exceed, _ = limiter.check_rate_limit("custom_provider", 501)
    assert would_exceed
    
    # Test fallback to default
    would_exceed, _ = limiter.check_rate_limit("unknown", DEFAULT_RATE_LIMITS["default"] + 1)
    assert would_exceed


@pytest.mark.parametrize("provider,limit", [
    ("openai", DEFAULT_RATE_LIMITS["openai"]),
    ("anthropic", DEFAULT_RATE_LIMITS["anthropic"]),
    ("openrouter", DEFAULT_RATE_LIMITS["openrouter"]),
    ("unknown", DEFAULT_RATE_LIMITS["default"]),
])
def test_provider_specific_limits(provider, limit):
    """Test rate limits for specific providers."""
    limiter = RateLimiter()
    
    # Should not exceed
    would_exceed, _ = limiter.check_rate_limit(provider, limit - 1)
    assert not would_exceed
    
    # Should exceed
    would_exceed, _ = limiter.check_rate_limit(provider, limit + 1)
    assert would_exceed


def test_cleanup_expired_entries():
    """Test cleanup of expired usage entries."""
    usage = TokenUsage(window_seconds=60)
    
    # Mock datetime.now() to test cleanup
    with patch('ra_aid.rate_limits.datetime') as mock_datetime:
        base_time = datetime.now()
        mock_datetime.now.return_value = base_time
        
        # Add initial usage
        usage.add_usage(100)
        assert usage.get_current_usage() == 100
        
        # Move time forward past window
        mock_datetime.now.return_value = base_time + timedelta(seconds=61)
        
        # Old usage should be cleaned up
        assert usage.get_current_usage() == 0
