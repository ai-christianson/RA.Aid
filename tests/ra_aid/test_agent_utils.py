"""Unit tests for agent_utils.py."""

from typing import Any, Dict, Literal
from unittest.mock import Mock, patch, MagicMock

import litellm
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ra_aid.agent_context import (
    agent_context,
)
from ra_aid.agent_utils import (
    AgentState,
    create_agent,
    is_anthropic_claude,
)
from ra_aid.anthropic_token_limiter import (
    get_model_token_limit,
    state_modifier,
)
from ra_aid.models_params import DEFAULT_TOKEN_LIMIT, models_params
from ra_aid.database.repositories.config_repository import (
    ConfigRepositoryManager,
    get_config_repository,
    config_repo_var,
)


@pytest.fixture
def mock_model():
    """Fixture providing a mock LLM model."""
    model = Mock(spec=BaseChatModel)
    return model


@pytest.fixture
def mock_config_repository():
    """Mock the ConfigRepository to avoid database operations during tests"""
    with patch(
        "ra_aid.database.repositories.config_repository.config_repo_var"
    ) as mock_repo_var:
        # Setup a mock repository
        mock_repo = MagicMock()

        # Create a dictionary to simulate config
        config = {}

        # Setup get method to return config values
        def get_config(key, default=None):
            return config.get(key, default)

        mock_repo.get.side_effect = get_config

        # Setup get_all method to return all config values
        mock_repo.get_all.return_value = config

        # Setup set method to update config values
        def set_config(key, value):
            config[key] = value

        mock_repo.set.side_effect = set_config

        # Setup update method to update multiple config values
        def update_config(update_dict):
            config.update(update_dict)

        mock_repo.update.side_effect = update_config

        # Make the mock context var return our mock repo
        mock_repo_var.get.return_value = mock_repo

        yield mock_repo


@pytest.fixture(autouse=True)
def mock_trajectory_repository():
    """Mock the TrajectoryRepository to avoid database operations during tests"""
    with patch(
        "ra_aid.database.repositories.trajectory_repository.trajectory_repo_var"
    ) as mock_repo_var:
        # Setup a mock repository
        mock_repo = MagicMock()

        # Setup create method to return a mock trajectory
        def mock_create(**kwargs):
            mock_trajectory = MagicMock()
            mock_trajectory.id = 1
            return mock_trajectory

        mock_repo.create.side_effect = mock_create

        # Make the mock context var return our mock repo
        mock_repo_var.get.return_value = mock_repo

        yield mock_repo


@pytest.fixture(autouse=True)
def mock_human_input_repository():
    """Mock the HumanInputRepository to avoid database operations during tests"""
    with patch(
        "ra_aid.database.repositories.human_input_repository.human_input_repo_var"
    ) as mock_repo_var:
        # Setup a mock repository
        mock_repo = MagicMock()

        # Setup get_most_recent_id method to return a dummy ID
        mock_repo.get_most_recent_id.return_value = 1

        # Make the mock context var return our mock repo
        mock_repo_var.get.return_value = mock_repo

        yield mock_repo


def test_create_agent_anthropic(mock_model, mock_config_repository):
    """Test create_agent with Anthropic Claude model."""
    mock_config_repository.update({"provider": "anthropic", "model": "claude-2"})

    with (
        patch("ra_aid.agent_utils.create_react_agent") as mock_react,
        patch("ra_aid.anthropic_token_limiter.state_modifier") as mock_state_modifier,
    ):
        mock_react.return_value = "react_agent"
        agent = create_agent(mock_model, [])

        assert agent == "react_agent"
        mock_react.assert_called_once_with(
            mock_model,
            [],
            interrupt_after=["tools"],
            version="v2",
            state_modifier=mock_react.call_args[1]["state_modifier"],
            name="React",
        )


def test_create_agent_openai(mock_model, mock_config_repository):
    """Test create_agent with OpenAI model."""
    mock_config_repository.update({"provider": "openai", "model": "gpt-4"})

    with patch("ra_aid.agent_utils.CiaynAgent") as mock_ciayn:
        mock_ciayn.return_value = "ciayn_agent"
        agent = create_agent(mock_model, [])

        assert agent == "ciayn_agent"
        mock_ciayn.assert_called_once_with(
            mock_model,
            [],
            max_tokens=models_params["openai"]["gpt-4"]["token_limit"],
            config={"provider": "openai", "model": "gpt-4"},
        )


def test_create_agent_no_token_limit(mock_model, mock_config_repository):
    """Test create_agent when no token limit is found."""
    mock_config_repository.update({"provider": "unknown", "model": "unknown-model"})

    with patch("ra_aid.agent_utils.CiaynAgent") as mock_ciayn:
        mock_ciayn.return_value = "ciayn_agent"
        agent = create_agent(mock_model, [])

        assert agent == "ciayn_agent"
        mock_ciayn.assert_called_once_with(
            mock_model,
            [],
            max_tokens=DEFAULT_TOKEN_LIMIT,
            config={"provider": "unknown", "model": "unknown-model"},
        )


def test_create_agent_missing_config(mock_model, mock_config_repository):
    """Test create_agent with missing configuration."""
    mock_config_repository.update({"provider": "openai"})

    with patch("ra_aid.agent_utils.CiaynAgent") as mock_ciayn:
        mock_ciayn.return_value = "ciayn_agent"
        agent = create_agent(mock_model, [])

        assert agent == "ciayn_agent"
        mock_ciayn.assert_called_once_with(
            mock_model,
            [],
            max_tokens=DEFAULT_TOKEN_LIMIT,
            config={"provider": "openai"},
        )


@pytest.fixture
def mock_messages():
    """Fixture providing mock message objects."""

    return [
        SystemMessage(content="System prompt"),
        HumanMessage(content="Human message 1"),
        AIMessage(content="AI response 1"),
        HumanMessage(content="Human message 2"),
        AIMessage(content="AI response 2"),
    ]


# This test has been moved to test_anthropic_token_limiter.py


def test_create_agent_with_checkpointer(mock_model, mock_config_repository):
    """Test create_agent with checkpointer argument."""
    mock_config_repository.update({"provider": "openai", "model": "gpt-4"})
    mock_checkpointer = Mock()

    with patch("ra_aid.agent_utils.CiaynAgent") as mock_ciayn:
        mock_ciayn.return_value = "ciayn_agent"
        agent = create_agent(mock_model, [], checkpointer=mock_checkpointer)

        assert agent == "ciayn_agent"
        mock_ciayn.assert_called_once_with(
            mock_model,
            [],
            max_tokens=models_params["openai"]["gpt-4"]["token_limit"],
            config={"provider": "openai", "model": "gpt-4"},
        )


def test_create_agent_anthropic_token_limiting_enabled(
    mock_model, mock_config_repository
):
    """Test create_agent sets up token limiting for Claude models when enabled."""
    mock_config_repository.update(
        {
            "provider": "anthropic",
            "model": "claude-2",
            "limit_tokens": True,
        }
    )

    with (
        patch("ra_aid.agent_utils.create_react_agent") as mock_react,
        patch("ra_aid.anthropic_token_limiter.get_model_token_limit") as mock_limit,
    ):
        mock_react.return_value = "react_agent"
        mock_limit.return_value = 100000

        agent = create_agent(mock_model, [])

        assert agent == "react_agent"
        args = mock_react.call_args
        assert "state_modifier" in args[1]
        assert callable(args[1]["state_modifier"])


def test_create_agent_anthropic_token_limiting_disabled(
    mock_model, mock_config_repository
):
    """Test create_agent doesn't set up token limiting for Claude models when disabled."""
    mock_config_repository.update(
        {
            "provider": "anthropic",
            "model": "claude-2",
            "limit_tokens": False,
        }
    )

    with (
        patch("ra_aid.agent_utils.create_react_agent") as mock_react,
        patch("ra_aid.anthropic_token_limiter.get_model_token_limit") as mock_limit,
    ):
        mock_react.return_value = "react_agent"
        mock_limit.return_value = 100000

        agent = create_agent(mock_model, [])

        assert agent == "react_agent"
        mock_react.assert_called_once_with(
            mock_model, [], interrupt_after=["tools"], version="v2", name="React"
        )


# These tests have been moved to test_anthropic_token_limiter.py


# New tests for private helper methods in agent_utils.py


def test_setup_and_restore_interrupt_handling():
    import signal

    from ra_aid.agent_utils import (
        _request_interrupt,
        _restore_interrupt_handling,
        _setup_interrupt_handling,
    )

    original_handler = signal.getsignal(signal.SIGINT)
    handler = _setup_interrupt_handling()
    # Verify the SIGINT handler is set to _request_interrupt
    assert signal.getsignal(signal.SIGINT) == _request_interrupt
    _restore_interrupt_handling(handler)
    # Verify the SIGINT handler is restored to the original
    assert signal.getsignal(signal.SIGINT) == original_handler


def test_agent_context_depth():
    from ra_aid.agent_context import agent_context, get_depth

    # Test depth with nested contexts
    assert get_depth() == 0  # No context
    with agent_context() as ctx1:
        assert get_depth() == 0  # Root context has depth 0
        assert ctx1.depth == 0

        with agent_context() as ctx2:
            assert get_depth() == 1  # Nested context has depth 1
            assert ctx2.depth == 1

            with agent_context() as ctx3:
                assert get_depth() == 2  # Doubly nested context has depth 2
                assert ctx3.depth == 2


def test_run_agent_stream(monkeypatch, mock_config_repository):
    from ra_aid.agent_utils import _run_agent_stream

    # Create a simple state class with a next property
    class State:
        def __init__(self):
            self.next = None

    # Create a dummy agent that yields one chunk and has a get_state method
    class DummyAgent:
        def stream(self, input_data, cfg: dict):
            yield {"content": "chunk1"}

        def get_state(self, state_config=None):
            # Return an object with a next property set to None
            return State()

    dummy_agent = DummyAgent()
    # Set flags so that _run_agent_stream will reset them
    with agent_context() as ctx:
        ctx.plan_completed = True
        ctx.task_completed = True
        ctx.completion_message = "existing"

    call_flag = {"called": False}

    def fake_print_agent_output(
        chunk: Dict[str, Any], agent_type: Literal["CiaynAgent", "React"], cost_cb=None
    ):
        call_flag["called"] = True

    monkeypatch.setattr(
        "ra_aid.agent_utils.print_agent_output", fake_print_agent_output
    )
    _run_agent_stream(dummy_agent, [HumanMessage("dummy prompt")])
    assert call_flag["called"]

    with agent_context() as ctx:
        assert ctx.plan_completed is False
        assert ctx.task_completed is False
        assert ctx.completion_message == ""


def test_execute_test_command_wrapper(monkeypatch):
    from ra_aid.agent_utils import _execute_test_command_wrapper

    # Patch execute_test_command to return a testable tuple
    def fake_execute(config, orig, tests, auto):
        return (True, "new prompt", auto, tests + 1)

    monkeypatch.setattr("ra_aid.agent_utils.execute_test_command", fake_execute)
    result = _execute_test_command_wrapper("orig", {}, 0, False)
    assert result == (True, "new prompt", False, 1)


def test_handle_api_error_valueerror():
    import pytest

    from ra_aid.agent_utils import _handle_api_error

    # ValueError not containing "code" or rate limit phrases should be re-raised
    with pytest.raises(ValueError):
        _handle_api_error(ValueError("some unrelated error"), 0, 5, 1)

    # ValueError with "429" should be handled without raising
    _handle_api_error(ValueError("error code 429"), 0, 5, 1)

    # ValueError with "rate limit" phrase should be handled without raising
    _handle_api_error(ValueError("hit rate limit"), 0, 5, 1)

    # ValueError with "too many requests" phrase should be handled without raising
    _handle_api_error(ValueError("too many requests, try later"), 0, 5, 1)

    # ValueError with "quota exceeded" phrase should be handled without raising
    _handle_api_error(ValueError("quota exceeded for this month"), 0, 5, 1)


def test_handle_api_error_status_code():
    from ra_aid.agent_utils import _handle_api_error

    # Error with status_code=429 attribute should be handled without raising
    error_with_status = Exception("Rate limited")
    error_with_status.status_code = 429
    _handle_api_error(error_with_status, 0, 5, 1)

    # Error with http_status=429 attribute should be handled without raising
    error_with_http_status = Exception("Too many requests")
    error_with_http_status.http_status = 429
    _handle_api_error(error_with_http_status, 0, 5, 1)


def test_handle_api_error_rate_limit_phrases():
    from ra_aid.agent_utils import _handle_api_error

    # Generic exception with "rate limit" phrase should be handled without raising
    _handle_api_error(Exception("You have exceeded your rate limit"), 0, 5, 1)

    # Generic exception with "too many requests" phrase should be handled without raising
    _handle_api_error(Exception("Too many requests, please slow down"), 0, 5, 1)

    # Generic exception with "quota exceeded" phrase should be handled without raising
    _handle_api_error(Exception("API quota exceeded for this billing period"), 0, 5, 1)

    # Generic exception with "rate" and "limit" separate but in message should be handled
    _handle_api_error(Exception("You hit the rate at which we limit requests"), 0, 5, 1)


def test_handle_api_error_max_retries():
    import pytest

    from ra_aid.agent_utils import _handle_api_error

    # When attempt reaches max retries, a RuntimeError should be raised
    with pytest.raises(RuntimeError):
        _handle_api_error(Exception("error code 429"), 4, 5, 1)


def test_handle_api_error_retry(monkeypatch):
    import time

    from ra_aid.agent_utils import _handle_api_error

    # Patch time.monotonic and time.sleep to simulate immediate delay expiration
    fake_time = [0]

    def fake_monotonic():
        fake_time[0] += 0.5
        return fake_time[0]

    monkeypatch.setattr(time, "monotonic", fake_monotonic)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    # Should not raise error when attempt is lower than max retries
    _handle_api_error(Exception("error code 429"), 0, 5, 1)


def test_is_anthropic_claude():
    """Test is_anthropic_claude function with various configurations."""
    # Test Anthropic provider cases
    assert is_anthropic_claude({"provider": "anthropic", "model": "claude-2"})
    assert is_anthropic_claude({"provider": "ANTHROPIC", "model": "claude-instant"})
    assert not is_anthropic_claude({"provider": "anthropic", "model": "gpt-4"})

    # Test OpenRouter provider cases
    assert is_anthropic_claude(
        {"provider": "openrouter", "model": "anthropic/claude-2"}
    )
    assert is_anthropic_claude(
        {"provider": "openrouter", "model": "anthropic/claude-instant"}
    )
    assert not is_anthropic_claude({"provider": "openrouter", "model": "openai/gpt-4"})

    # Test edge cases
    assert not is_anthropic_claude({})  # Empty config
    assert not is_anthropic_claude({"provider": "anthropic"})  # Missing model
    assert not is_anthropic_claude({"model": "claude-2"})  # Missing provider
    assert not is_anthropic_claude(
        {"provider": "other", "model": "claude-2"}
    )  # Wrong provider


def test_run_agent_with_retry_checks_crash_status(monkeypatch, mock_config_repository):
    """Test that run_agent_with_retry checks for crash status at the beginning of each iteration."""
    from ra_aid.agent_context import agent_context, mark_agent_crashed
    from ra_aid.agent_utils import run_agent_with_retry

    # Setup mocks for dependencies to isolate our test
    dummy_agent = Mock()

    # Track function calls
    mock_calls = {"run_agent_stream": 0}

    def mock_run_agent_stream(*args, **kwargs):
        mock_calls["run_agent_stream"] += 1

    def mock_setup_interrupt_handling():
        return None

    def mock_restore_interrupt_handling(handler):
        pass

    def mock_is_crashed():
        return ctx.is_crashed() if ctx else False

    def mock_get_crash_message():
        return ctx.agent_crashed_message if ctx and ctx.is_crashed() else None

    # Apply mocks
    monkeypatch.setattr("ra_aid.agent_utils._run_agent_stream", mock_run_agent_stream)
    monkeypatch.setattr(
        "ra_aid.agent_utils._setup_interrupt_handling", mock_setup_interrupt_handling
    )
    monkeypatch.setattr(
        "ra_aid.agent_utils._restore_interrupt_handling",
        mock_restore_interrupt_handling,
    )
    monkeypatch.setattr("ra_aid.agent_utils.check_interrupt", lambda: None)

    # First, run without a crash - agent should be run
    with agent_context() as ctx:
        monkeypatch.setattr("ra_aid.agent_context.is_crashed", mock_is_crashed)
        monkeypatch.setattr(
            "ra_aid.agent_context.get_crash_message", mock_get_crash_message
        )
        result = run_agent_with_retry(dummy_agent, "test prompt", {})
        assert mock_calls["run_agent_stream"] == 1

    # Reset call counter
    mock_calls["run_agent_stream"] = 0

    # Now run with a crash - agent should not be run
    with agent_context() as ctx:
        mark_agent_crashed("Test crash message")
        monkeypatch.setattr("ra_aid.agent_context.is_crashed", mock_is_crashed)
        monkeypatch.setattr(
            "ra_aid.agent_context.get_crash_message", mock_get_crash_message
        )
        result = run_agent_with_retry(dummy_agent, "test prompt", {})
        # Verify _run_agent_stream was not called
        assert mock_calls["run_agent_stream"] == 0
        # Verify the result contains the crash message
        assert "Agent has crashed: Test crash message" in result


def test_run_agent_with_retry_handles_badrequest_error(
    monkeypatch, mock_config_repository
):
    """Test that run_agent_with_retry properly handles BadRequestError as unretryable."""
    from ra_aid.agent_context import agent_context, is_crashed
    from ra_aid.agent_utils import run_agent_with_retry
    from ra_aid.exceptions import ToolExecutionError

    # Setup mocks
    dummy_agent = Mock()

    # Track function calls and simulate BadRequestError
    run_count = [0]

    def mock_run_agent_stream(*args, **kwargs):
        run_count[0] += 1
        if run_count[0] == 1:
            # First call throws a 400 BadRequestError
            raise ToolExecutionError("400 Bad Request: Invalid input")
        # If it's called again, it should run normally

    def mock_setup_interrupt_handling():
        return None

    def mock_restore_interrupt_handling(handler):
        pass

    def mock_mark_agent_crashed(message):
        ctx.agent_has_crashed = True
        ctx.agent_crashed_message = message

    def mock_is_crashed():
        return ctx.is_crashed() if ctx else False

    # Apply mocks
    monkeypatch.setattr("ra_aid.agent_utils._run_agent_stream", mock_run_agent_stream)
    monkeypatch.setattr(
        "ra_aid.agent_utils._setup_interrupt_handling", mock_setup_interrupt_handling
    )
    monkeypatch.setattr(
        "ra_aid.agent_utils._restore_interrupt_handling",
        mock_restore_interrupt_handling,
    )
    monkeypatch.setattr("ra_aid.agent_utils.check_interrupt", lambda: None)

    with agent_context() as ctx:
        monkeypatch.setattr(
            "ra_aid.agent_context.mark_agent_crashed", mock_mark_agent_crashed
        )
        monkeypatch.setattr("ra_aid.agent_context.is_crashed", mock_is_crashed)

        result = run_agent_with_retry(dummy_agent, "test prompt", {})
        # Verify the agent was only run once and not retried
        assert run_count[0] == 1
        # Verify the result contains the crash message
        assert "Agent has crashed: Unretryable error" in result
        # Verify the agent is marked as crashed
        assert is_crashed()


def test_run_agent_with_retry_handles_api_badrequest_error(
    monkeypatch, mock_config_repository
):
    """Test that run_agent_with_retry properly handles API BadRequestError as unretryable."""
    # Import APIError from anthropic module and patch it on the agent_utils module

    from ra_aid.agent_context import agent_context, is_crashed
    from ra_aid.agent_utils import run_agent_with_retry

    # Setup mocks
    dummy_agent = Mock()

    # Track function calls and simulate BadRequestError
    run_count = [0]

    # Create a mock APIError class that simulates Anthropic's APIError
    class MockAPIError(Exception):
        pass

    def mock_run_agent_stream(*args, **kwargs):
        run_count[0] += 1
        if run_count[0] == 1:
            # First call throws a 400 Bad Request APIError
            mock_error = MockAPIError("400 Bad Request")
            mock_error.__class__.__name__ = (
                "APIError"  # Make it look like Anthropic's APIError
            )
            raise mock_error
        # If it's called again, it should run normally

    def mock_setup_interrupt_handling():
        return None

    def mock_restore_interrupt_handling(handler):
        pass

    def mock_mark_agent_crashed(message):
        ctx.agent_has_crashed = True
        ctx.agent_crashed_message = message

    def mock_is_crashed():
        return ctx.is_crashed() if ctx else False

    # Apply mocks
    monkeypatch.setattr("ra_aid.agent_utils._run_agent_stream", mock_run_agent_stream)
    monkeypatch.setattr(
        "ra_aid.agent_utils._setup_interrupt_handling", mock_setup_interrupt_handling
    )
    monkeypatch.setattr(
        "ra_aid.agent_utils._restore_interrupt_handling",
        mock_restore_interrupt_handling,
    )
    monkeypatch.setattr("ra_aid.agent_utils.check_interrupt", lambda: None)
    monkeypatch.setattr("ra_aid.agent_utils._handle_api_error", lambda *args: None)
    monkeypatch.setattr("ra_aid.agent_utils.APIError", MockAPIError)
    monkeypatch.setattr(
        "ra_aid.agent_context.mark_agent_crashed", mock_mark_agent_crashed
    )
    monkeypatch.setattr("ra_aid.agent_context.is_crashed", mock_is_crashed)

    with agent_context() as ctx:
        result = run_agent_with_retry(dummy_agent, "test prompt", {})
        # Verify the agent was only run once and not retried
        assert run_count[0] == 1
        # Verify the result contains the crash message
        assert "Agent has crashed: Unretryable API error" in result
        # Verify the agent is marked as crashed
        assert is_crashed()


def test_handle_api_error_resource_exhausted():
    from google.api_core.exceptions import ResourceExhausted
    from ra_aid.agent_utils import _handle_api_error

    # ResourceExhausted exception should be handled without raising
    resource_exhausted_error = ResourceExhausted(
        "429 Resource has been exhausted (e.g. check quota)."
    )
    _handle_api_error(resource_exhausted_error, 0, 5, 1)
