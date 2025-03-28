
"""Unit tests for default_callback_handler.py."""

import time
from unittest.mock import patch, MagicMock
import pytest

from langchain_core.outputs import LLMResult
from ra_aid.callbacks.default_callback_handler import (
    DefaultCallbackHandler,
    MODEL_COSTS,
    DEFAULT_MODEL,
)


@pytest.fixture
def callback_handler():
    """Fixture providing a fresh DefaultCallbackHandler instance."""
    # Clear any existing singleton instance
    DefaultCallbackHandler._instances = {}
    return DefaultCallbackHandler()


def test_singleton_pattern(callback_handler):
    """Test that DefaultCallbackHandler follows singleton pattern."""
    handler2 = DefaultCallbackHandler()
    assert handler2 is callback_handler


def test_initial_state(callback_handler):
    """Test initial state of callback handler."""
    assert callback_handler.total_tokens == 0
    assert callback_handler.prompt_tokens == 0
    assert callback_handler.completion_tokens == 0
    assert callback_handler.successful_requests == 0
    assert callback_handler.total_cost == 0.0
    assert callback_handler.model_name == DEFAULT_MODEL
    assert callback_handler.session_totals == {
        "tokens": 0,
        "cost": 0.0,
        "duration": 0.0,
    }


def test_on_llm_start(callback_handler):
    """Test on_llm_start records request time."""
    callback_handler.on_llm_start({}, [])
    assert callback_handler._last_request_time is not None


def test_on_llm_end_no_token_usage(callback_handler):
    """Test on_llm_end with no token usage data."""
    mock_response = MagicMock(spec=LLMResult)
    mock_response.llm_output = {}
    
    callback_handler.on_llm_end(mock_response)
    assert callback_handler.total_tokens == 0


def test_on_llm_end_with_token_usage(callback_handler):
    """Test on_llm_end with token usage data."""
    mock_response = MagicMock(spec=LLMResult)
    mock_response.llm_output = {
        "token_usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
    }
    
    with patch("time.time", return_value=100.1):
        callback_handler._last_request_time = 100.0
        callback_handler.on_llm_end(mock_response)
    
    assert callback_handler.prompt_tokens == 100
    assert callback_handler.completion_tokens == 50
    assert callback_handler.total_tokens == 150
    assert callback_handler.session_totals["tokens"] == 150
    assert callback_handler.session_totals["duration"] == 0.1


@pytest.mark.parametrize("model_name,cost", [
    ("claude-3-7-sonnet-20250219", (100 * 0.000003) + (50 * 0.000015)),
    ("claude-3-opus-20240229", (100 * 0.000015) + (50 * 0.000075)),
    ("claude-2", (100 * 0.00001102) + (50 * 0.00003268)),
])
def test_cost_calculation(model_name, cost, callback_handler):
    """Test cost calculation for different models."""
    callback_handler.model_name = model_name
    
    mock_response = MagicMock(spec=LLMResult)
    mock_response.llm_output = {
        "token_usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
    }
    
    callback_handler.on_llm_end(mock_response)
    assert pytest.approx(callback_handler.total_cost) == cost
    assert pytest.approx(callback_handler.session_totals["cost"]) == cost


def test_unknown_model_no_cost(callback_handler):
    """Test that unknown models don't accumulate costs."""
    callback_handler.model_name = "unknown-model"
    
    mock_response = MagicMock(spec=LLMResult)
    mock_response.llm_output = {
        "token_usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
    }
    
    callback_handler.on_llm_end(mock_response)
    assert callback_handler.total_cost == 0.0


def test_reset_session_totals(callback_handler):
    """Test reset_session_totals clears session data."""
    mock_response = MagicMock(spec=LLMResult)
    mock_response.llm_output = {
        "token_usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
    }
    
    callback_handler.on_llm_end(mock_response)
    callback_handler.reset_session_totals()
    
    assert callback_handler.session_totals == {
        "tokens": 0,
        "cost": 0.0,
        "duration": 0.0,
    }
    # Verify global totals remain
    assert callback_handler.total_tokens == 150


def test_get_stats(callback_handler):
    """Test get_stats returns correct data structure."""
    mock_response = MagicMock(spec=LLMResult)
    mock_response.llm_output = {
        "token_usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
    }
    
    callback_handler.on_llm_end(mock_response)
    stats = callback_handler.get_stats()
    
    assert stats["total_tokens"] == 150
    assert stats["prompt_tokens"] == 100
    assert stats["completion_tokens"] == 50
    assert pytest.approx(stats["total_cost"]) == (100 * 0.000003) + (50 * 0.000015)
    assert stats["successful_requests"] == 1
    assert stats["model_name"] == DEFAULT_MODEL
    assert isinstance(stats["session_totals"], dict)
