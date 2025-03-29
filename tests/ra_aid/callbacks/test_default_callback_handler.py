"""Unit tests for default_callback_handler.py."""

from unittest.mock import patch, MagicMock
import pytest
from decimal import Decimal

from langchain_core.outputs import LLMResult
from ra_aid.callbacks.default_callback_handler import (
    DefaultCallbackHandler,
)
from ra_aid.config import DEFAULT_MODEL


@pytest.fixture(autouse=True)
def mock_repositories():
    """Mock repository getters to prevent database access."""
    with (
        patch(
            "ra_aid.callbacks.default_callback_handler.get_trajectory_repository"
        ) as mock_traj_repo,
        patch(
            "ra_aid.callbacks.default_callback_handler.get_session_repository"
        ) as mock_session_repo,
    ):
        # Configure the mocks if needed, e.g., return a MagicMock instance
        mock_traj_repo.return_value = MagicMock()
        # Mock get_current_session_record to return None or a mock session
        mock_session_instance = MagicMock()
        mock_session_instance.get_id.return_value = 123  # Example session ID
        mock_session_repo.return_value.get_current_session_record.return_value = (
            mock_session_instance
        )
        yield mock_traj_repo, mock_session_repo


@pytest.fixture
def callback_handler(mock_repositories):  # Ensure mocks are active
    """Fixture providing a fresh DefaultCallbackHandler instance."""
    # Clear any existing singleton instance
    DefaultCallbackHandler._instances = {}
    # Provide the required model_name argument
    handler = DefaultCallbackHandler(model_name=DEFAULT_MODEL)
    # Reset all state before the test runs
    handler.reset_all_totals()
    # Re-mock session ID after reset_all_totals clears it
    handler.session_totals["session_id"] = 123
    return handler


def test_singleton_pattern(callback_handler):
    """Test that DefaultCallbackHandler follows singleton pattern."""
    # Pass model_name when calling again to allow re-initialization
    handler2 = DefaultCallbackHandler(model_name=DEFAULT_MODEL)
    assert handler2 is callback_handler
    # Verify re-initialization happened (optional check)
    assert handler2.model_name == DEFAULT_MODEL


def test_initial_state(callback_handler):
    """Test initial state of callback handler."""
    assert callback_handler.total_tokens == 0
    assert callback_handler.prompt_tokens == 0
    assert callback_handler.completion_tokens == 0
    assert callback_handler.successful_requests == 0
    assert callback_handler.total_cost == Decimal("0.0")
    assert callback_handler.model_name == DEFAULT_MODEL
    # Update assertion to match the full initial structure and use Decimal
    assert callback_handler.session_totals == {
        "cost": Decimal("0.0"),
        "tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "session_id": 123,  # From mock_repositories fixture
        "duration": 0.0,
    }


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
    # Use pytest.approx for float comparison
    assert callback_handler.session_totals["duration"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    "model_name,cost",
    [
        ("claude-3-7-sonnet-20250219", (100 * 0.000003) + (50 * 0.000015)),
        ("claude-3-opus-20240229", (100 * 0.000015) + (50 * 0.000075)),
        ("claude-2", (100 * 0.00001102) + (50 * 0.00003268)),
    ],
)
def test_cost_calculation(model_name, cost, callback_handler):
    """Test cost calculation for different models."""
    # Re-initialize costs after changing model name
    callback_handler._initialize(model_name=model_name)
    assert callback_handler.model_name == model_name  # Verify model name change

    mock_response = MagicMock(spec=LLMResult)
    mock_response.llm_output = {
        "token_usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
    }

    # Mock time for duration calculation consistency
    with patch("time.time", return_value=100.1):
        callback_handler._last_request_time = 100.0
        callback_handler.on_llm_end(mock_response)

    # Compare Decimal with Decimal using pytest.approx
    expected_cost_decimal = Decimal(str(cost))
    assert callback_handler.total_cost == pytest.approx(expected_cost_decimal)


def test_unknown_model_no_cost(callback_handler):
    """Test that unknown models don't accumulate costs."""
    # Set model name and initialize costs for it
    # State is already reset by the fixture
    callback_handler._initialize(model_name="unknown-model")
    assert callback_handler.model_name == "unknown-model"

    mock_response = MagicMock(spec=LLMResult)
    mock_response.llm_output = {
        "token_usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
    }

    # Mock time for duration calculation consistency
    with patch("time.time", return_value=100.1):
        callback_handler._last_request_time = 100.0
        callback_handler.on_llm_end(mock_response)

    assert callback_handler.total_cost == Decimal("0.0")
    assert callback_handler.session_totals["cost"] == Decimal("0.0")


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

    # First call to on_llm_end
    callback_handler.on_llm_end(mock_response)
    
    # Second call to on_llm_end
    # Mock time for duration calculation consistency for the second call
    with patch("time.time", return_value=100.1):
        callback_handler._last_request_time = 100.0
        callback_handler.on_llm_end(mock_response)
    
    # Get totals before reset to verify later
    cost_before_reset = callback_handler.total_cost
    tokens_before_reset = callback_handler.total_tokens
    
    # Call reset_session_totals
    callback_handler.reset_session_totals()

    # Check session totals after reset
    assert callback_handler.session_totals == {
        "cost": Decimal("0.0"),  # Expect Decimal after reset
        "tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "session_id": 123,  # session_id is PRESERVED by reset_session_totals
        "duration": 0.0,
    }
    # Verify last message totals remain (these are instance attributes, not reset by session reset)
    assert callback_handler.total_tokens == tokens_before_reset  # Should be 150 from the second on_llm_end call
    assert callback_handler.total_cost == cost_before_reset  # Should be cost from the second on_llm_end call

    # Now test reset_all_totals
    callback_handler.reset_all_totals()
    assert callback_handler.total_tokens == 0
    assert callback_handler.total_cost == Decimal("0.0")
    assert DefaultCallbackHandler.cumulative_total_tokens == 0
    assert callback_handler.session_totals["cost"] == Decimal("0.0")
    assert callback_handler.session_totals["tokens"] == 0


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

    # Costs are initialized by the fixture's reset_all_totals call

    # Mock time for duration calculation consistency
    with patch("time.time", return_value=100.1):
        callback_handler._last_request_time = 100.0
        callback_handler.on_llm_end(mock_response)

    stats = callback_handler.get_stats()

    # Calculate expected cost using Decimal
    expected_cost = Decimal("100") * Decimal("0.000003") + Decimal("50") * Decimal(
        "0.000015"
    )

    assert stats["total_tokens"] == 150  # Last message tokens
    assert stats["prompt_tokens"] == 100  # Last message prompt tokens
    assert stats["completion_tokens"] == 50  # Last message completion tokens
    assert stats["total_cost"] == pytest.approx(expected_cost)  # Cumulative cost
    assert stats["successful_requests"] == 1
    assert stats["model_name"] == DEFAULT_MODEL

    # Check cumulative tokens
    assert stats["cumulative_tokens"]["total"] == 150
    assert stats["cumulative_tokens"]["prompt"] == 100
    assert stats["cumulative_tokens"]["completion"] == 50

    # Check session totals within stats
    assert isinstance(stats["session_totals"], dict)
    assert stats["session_totals"]["tokens"] == 150
    assert stats["session_totals"]["input_tokens"] == 100
    assert stats["session_totals"]["output_tokens"] == 50
    assert stats["session_totals"]["cost"] == pytest.approx(expected_cost)
    assert stats["session_totals"]["duration"] == pytest.approx(0.1)
    assert stats["session_totals"]["session_id"] == 123  # From mock
