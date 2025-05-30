"""Unit tests for limit checking functionality in default_callback_handler.py."""

import pytest
from unittest.mock import patch, MagicMock
from decimal import Decimal

from langchain_core.outputs import LLMResult
from ra_aid.callbacks.default_callback_handler import DefaultCallbackHandler
from ra_aid.config import DEFAULT_MODEL


@pytest.fixture(autouse=True)
def mock_repositories():
    """Mock repository getters to prevent database access."""
    with (
        patch("ra_aid.callbacks.default_callback_handler.get_trajectory_repository") as mock_traj_repo,
        patch("ra_aid.callbacks.default_callback_handler.get_session_repository") as mock_session_repo,
        patch("ra_aid.callbacks.default_callback_handler.get_config_repository") as mock_config_repo,
    ):
        # Configure the mocks
        mock_traj_repo.return_value = MagicMock()
        
        mock_session_instance = MagicMock()
        mock_session_instance.get_id.return_value = 123
        mock_session_repo.return_value.get_current_session_record.return_value = mock_session_instance
        
        # Mock config repository
        mock_config_repo.return_value = MagicMock()
        
        yield mock_traj_repo, mock_session_repo, mock_config_repo


@pytest.fixture
def callback_handler(mock_repositories):
    """Fixture providing a fresh DefaultCallbackHandler instance."""
    DefaultCallbackHandler._instances = {}
    handler = DefaultCallbackHandler(model_name=DEFAULT_MODEL)
    handler.reset_all_totals()
    handler.session_totals["session_id"] = 123
    return handler


def test_check_limits_no_limits_set(callback_handler):
    """Test _check_limits returns None when no limits are configured."""
    # Mock config to return None for all limits
    callback_handler.config_repo.get.side_effect = lambda key, default=None: None
    
    result = callback_handler._check_limits()
    assert result is None


def test_check_limits_cost_exceeded(callback_handler):
    """Test _check_limits detects cost limit exceeded."""
    # Set up config mock
    def config_get(key, default=None):
        if key == "max_cost":
            return 0.01  # $0.01 limit
        elif key == "max_tokens":
            return None
        elif key == "exit_at_limit":
            return False
        return default
    
    callback_handler.config_repo.get.side_effect = config_get
    
    # Set session cost above limit
    callback_handler.session_totals["cost"] = Decimal("0.02")  # $0.02
    callback_handler.session_totals["tokens"] = 100
    
    result = callback_handler._check_limits()
    assert result is not None
    limit_type, current_val, limit_val, exit_at_limit = result
    assert limit_type == "cost"
    assert current_val == 0.02
    assert limit_val == 0.01
    assert exit_at_limit is False


def test_check_limits_tokens_exceeded(callback_handler):
    """Test _check_limits detects token limit exceeded."""
    # Set up config mock
    def config_get(key, default=None):
        if key == "max_cost":
            return None
        elif key == "max_tokens":
            return 1000  # 1000 token limit
        elif key == "exit_at_limit":
            return True
        return default
    
    callback_handler.config_repo.get.side_effect = config_get
    
    # Set session tokens above limit
    callback_handler.session_totals["cost"] = Decimal("0.005")
    callback_handler.session_totals["tokens"] = 1500  # Above limit
    
    result = callback_handler._check_limits()
    assert result is not None
    limit_type, current_val, limit_val, exit_at_limit = result
    assert limit_type == "tokens"
    assert current_val == 1500
    assert limit_val == 1000
    assert exit_at_limit is True


def test_check_limits_cost_priority_over_tokens(callback_handler):
    """Test that cost limit is checked before token limit."""
    # Set up config mock with both limits exceeded
    def config_get(key, default=None):
        if key == "max_cost":
            return 0.01
        elif key == "max_tokens":
            return 1000
        elif key == "exit_at_limit":
            return False
        return default
    
    callback_handler.config_repo.get.side_effect = config_get
    
    # Set both above limits
    callback_handler.session_totals["cost"] = Decimal("0.02")
    callback_handler.session_totals["tokens"] = 1500
    
    result = callback_handler._check_limits()
    assert result is not None
    limit_type, current_val, limit_val, exit_at_limit = result
    # Cost should be checked first
    assert limit_type == "cost"


def test_record_limit_reached(callback_handler):
    """Test _record_limit_reached creates trajectory record."""
    callback_handler._record_limit_reached("cost", 0.02, 0.01, False)
    
    # Verify trajectory repo was called
    callback_handler.trajectory_repo.create.assert_called_once_with(
        record_type="limit_reached",
        session_id=123,
        step_data={
            "limit_type": "cost",
            "current_value": 0.02,
            "limit_value": 0.01,
            "exit_at_limit": False
        }
    )


def test_format_limit_message_cost(callback_handler):
    """Test _format_limit_message for cost limits."""
    message = callback_handler._format_limit_message("cost", 0.025, 0.01)
    assert "Cost limit exceeded: $0.025000 >= $0.010000. Continue anyway?" == message


def test_format_limit_message_tokens(callback_handler):
    """Test _format_limit_message for token limits."""
    message = callback_handler._format_limit_message("tokens", 1500, 1000)
    assert "Token limit exceeded: 1,500 >= 1,000. Continue anyway?" == message


@patch('sys.exit')
@patch('ra_aid.callbacks.default_callback_handler.Confirm.ask')
def test_limit_exceeded_user_continues(mock_confirm, mock_exit, callback_handler):
    """Test limit exceeded with user choosing to continue."""
    # Set up config mock
    def config_get(key, default=None):
        if key == "max_cost":
            return 0.01
        elif key == "max_tokens":
            return None
        elif key == "exit_at_limit":
            return False
        return default
    
    callback_handler.config_repo.get.side_effect = config_get
    mock_confirm.return_value = True  # User chooses to continue
    
    # Create mock response with tokens that will exceed cost limit
    mock_response = MagicMock(spec=LLMResult)
    mock_response.llm_output = {
        "token_usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500,
        }
    }
    
    # Set initial session totals near the limit
    callback_handler.session_totals["cost"] = Decimal("0.009")
    
    with patch("time.time", return_value=100.1):
        callback_handler._last_request_time = 100.0
        callback_handler.on_llm_end(mock_response)
    
    # Verify user was prompted and didn't exit
    mock_confirm.assert_called_once()
    mock_exit.assert_not_called()


@patch('sys.exit')
@patch('ra_aid.callbacks.default_callback_handler.Confirm.ask')
def test_limit_exceeded_user_exits(mock_confirm, mock_exit, callback_handler):
    """Test limit exceeded with user choosing to exit."""
    # Set up config mock
    def config_get(key, default=None):
        if key == "max_cost":
            return 0.01
        elif key == "max_tokens":
            return None
        elif key == "exit_at_limit":
            return False
        return default
    
    callback_handler.config_repo.get.side_effect = config_get
    mock_confirm.return_value = False  # User chooses to exit
    
    # Create mock response with tokens that will exceed cost limit
    mock_response = MagicMock(spec=LLMResult)
    mock_response.llm_output = {
        "token_usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500,
        }
    }
    
    # Set initial session totals near the limit
    callback_handler.session_totals["cost"] = Decimal("0.009")
    
    with patch("time.time", return_value=100.1):
        callback_handler._last_request_time = 100.0
        callback_handler.on_llm_end(mock_response)
    
    # Verify user was prompted and system exited
    mock_confirm.assert_called_once()
    mock_exit.assert_called_once_with(0)


@patch('sys.exit')
@patch('builtins.print')
def test_limit_exceeded_auto_exit(mock_print, mock_exit, callback_handler):
    """Test limit exceeded with auto-exit enabled."""
    # Set up config mock with auto-exit
    def config_get(key, default=None):
        if key == "max_cost":
            return 0.01
        elif key == "max_tokens":
            return None
        elif key == "exit_at_limit":
            return True  # Auto-exit enabled
        return default
    
    callback_handler.config_repo.get.side_effect = config_get
    
    # Create mock response with tokens that will exceed cost limit
    mock_response = MagicMock(spec=LLMResult)
    mock_response.llm_output = {
        "token_usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500,
        }
    }
    
    # Set initial session totals near the limit
    callback_handler.session_totals["cost"] = Decimal("0.009")
    
    with patch("time.time", return_value=100.1):
        callback_handler._last_request_time = 100.0
        callback_handler.on_llm_end(mock_response)
    
    # Verify auto-exit occurred without user prompt
    mock_print.assert_called_once()
    mock_exit.assert_called_once_with(0)


def test_check_limits_zero_limits_ignored(callback_handler):
    """Test that zero values for limits are treated as disabled."""
    # Set up config mock with zero limits
    def config_get(key, default=None):
        if key == "max_cost":
            return 0.0  # Zero should be ignored
        elif key == "max_tokens":
            return 0  # Zero should be ignored
        elif key == "exit_at_limit":
            return False
        return default
    
    callback_handler.config_repo.get.side_effect = config_get
    
    # Set session values above zero
    callback_handler.session_totals["cost"] = Decimal("0.02")
    callback_handler.session_totals["tokens"] = 1500
    
    result = callback_handler._check_limits()
    assert result is None  # No limits should trigger


def test_check_limits_exact_match_triggers(callback_handler):
    """Test that exact limit matches trigger the limit check."""
    # Set up config mock
    def config_get(key, default=None):
        if key == "max_cost":
            return 0.01
        elif key == "max_tokens":
            return None
        elif key == "exit_at_limit":
            return False
        return default
    
    callback_handler.config_repo.get.side_effect = config_get
    
    # Set session cost exactly at limit
    callback_handler.session_totals["cost"] = Decimal("0.01")  # Exactly at limit
    callback_handler.session_totals["tokens"] = 100
    
    result = callback_handler._check_limits()
    assert result is not None
    limit_type, current_val, limit_val, exit_at_limit = result
    assert limit_type == "cost"
    assert current_val == 0.01
    assert limit_val == 0.01
