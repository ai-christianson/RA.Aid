"""Utility functions for working with agents."""

import signal
import sys
import threading
import time
from typing import Any, Dict, List, Literal, Optional
import uuid

from langgraph.graph.graph import CompiledGraph
from ra_aid.callbacks.anthropic_callback_handler import (
    AnthropicCallbackHandler,
)
from ra_aid.model_detection import model_name_has_claude, should_use_react_agent, get_model_name_from_chat_model


from anthropic import APIError, APITimeoutError, InternalServerError, RateLimitError
from openai import RateLimitError as OpenAIRateLimitError
from litellm.exceptions import RateLimitError as LiteLLMRateLimitError
from google.api_core.exceptions import ResourceExhausted
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

from ra_aid.agent_context import (
    agent_context,
    is_completed,
    reset_completion_flags,
    should_exit,
)
from ra_aid.agent_backends.ciayn_agent import CiaynAgent
from ra_aid.agents_alias import RAgents
from ra_aid.config import DEFAULT_MAX_TEST_CMD_RETRIES, DEFAULT_MODEL
from ra_aid.console.formatting import cpm, print_error
from ra_aid.console.output import print_agent_output
from ra_aid.exceptions import (
    AgentInterrupt,
    FallbackToolExecutionError,
    ToolExecutionError,
)
from ra_aid.fallback_handler import FallbackHandler
from ra_aid.logging_config import get_logger
from ra_aid.models_params import (
    DEFAULT_TOKEN_LIMIT,
)
from ra_aid.tools.handle_user_defined_test_cmd_execution import execute_test_command
from ra_aid.database.repositories.human_input_repository import (
    get_human_input_repository,
)
from ra_aid.database.repositories.trajectory_repository import (
    get_trajectory_repository,
)
from ra_aid.database.repositories.config_repository import get_config_repository
from ra_aid.anthropic_token_limiter import (
    sonnet_35_state_modifier,
    state_modifier,
    get_model_token_limit,
)


logger = get_logger(__name__)


def build_agent_kwargs(
    checkpointer: Optional[Any] = None,
    model: BaseChatModel = None,
    max_input_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Build kwargs dictionary for agent creation.

    Args:
        checkpointer: Optional memory checkpointer
        model: The language model to use for token counting
        max_input_tokens: Optional token limit for the model

    Returns:
        Dictionary of kwargs for agent creation
    """
    agent_kwargs = {
        "version": "v2",
    }

    if checkpointer is not None:
        agent_kwargs["checkpointer"] = checkpointer

    # Use repository method to check if token limiting is enabled
    limit_tokens = get_config_repository().get("limit_tokens", True)
    model_name = get_model_name_from_chat_model(model)

    if limit_tokens and model is not None and model_name_has_claude(model_name):

        def wrapped_state_modifier(state: AgentState) -> list[BaseMessage]:
            model_name = get_model_name_from_chat_model(model)

            if any(
                pattern in model_name
                for pattern in ["claude-3.5", "claude3.5", "claude-3-5"]
            ):
                return sonnet_35_state_modifier(
                    state, max_input_tokens=max_input_tokens
                )

            return state_modifier(state, model, max_input_tokens=max_input_tokens)

        agent_kwargs["state_modifier"] = wrapped_state_modifier

    # Important for anthropic callback handler to determine the correct model name given the agent
    agent_kwargs["name"] = model_name

    return agent_kwargs




def create_agent(
    model: BaseChatModel,
    tools: List[Any],
    *,
    checkpointer: Any = None,
    agent_type: str = "default",
):
    """Create a react agent with the given configuration.

    Args:
        model: The LLM model to use
        tools: List of tools to provide to the agent
        checkpointer: Optional memory checkpointer
        agent_type: Type of agent to create (default: "default")

    Returns:
        The created agent instance

    Token limiting helps prevent context window overflow by trimming older messages
    while preserving system messages. It can be disabled by setting
    limit_tokens config value to False using get_config_repository().set('limit_tokens', False).
    """
    try:
        # Try to get config from repository for production use
        try:
            # Get only the necessary config values
            provider = get_config_repository().get("provider", "anthropic")
            model_name = get_config_repository().get("model", "")

            logger.debug(
                "Creating agent with config values: provider='%s', model='%s'",
                provider,
                model_name,
            )

            # Create minimal config dict with only needed values
            config = {
                "provider": provider,
            }
            # Only add model key if it has a value (to match test expectations)
            if model_name:
                config["model"] = model_name
        except RuntimeError:
            # In tests, this may fail because the repository isn't set up
            # So we'll use default values
            config = {}

        max_input_tokens = (
            get_model_token_limit(config, agent_type, model) or DEFAULT_TOKEN_LIMIT
        )

        use_react_agent = should_use_react_agent(model)

        if use_react_agent:
            logger.debug(
                "Using create_react_agent to instantiate agent based on model capabilities."
            )
            cpm("Using ReAct Agent")
            agent_kwargs = build_agent_kwargs(checkpointer, model, max_input_tokens)
            return create_react_agent(
                model, tools, interrupt_after=["tools"], **agent_kwargs
            )
        else:
            cpm("Using Ciayn Agent")
            logger.debug("Using CiaynAgent agent instance based on model capabilities.")
            return CiaynAgent(model, tools, max_tokens=max_input_tokens, config=config)

    except Exception as e:
        # Default to REACT agent if provider/model detection fails
        logger.warning(f"Failed to detect model type: {e}. Defaulting to REACT agent.")

        # Get only needed values for get_model_token_limit
        provider = get_config_repository().get("provider", "anthropic")
        model_name = get_config_repository().get("model", "")

        # Create config with only needed keys
        config = {"provider": provider}
        if model_name:
            config["model"] = model_name

        max_input_tokens = get_model_token_limit(config, agent_type, model)
        agent_kwargs = build_agent_kwargs(checkpointer, model, max_input_tokens)
        return create_react_agent(
            model, tools, interrupt_after=["tools"], **agent_kwargs
        )


_CONTEXT_STACK = []
_INTERRUPT_CONTEXT = None
_FEEDBACK_MODE = False


def _request_interrupt(signum, frame):
    global _INTERRUPT_CONTEXT
    if _CONTEXT_STACK:
        _INTERRUPT_CONTEXT = _CONTEXT_STACK[-1]

    if _FEEDBACK_MODE:
        print()
        print(" 👋 Bye!")
        print()
        sys.exit(0)


class InterruptibleSection:
    def __enter__(self):
        _CONTEXT_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _CONTEXT_STACK.remove(self)


def check_interrupt():
    if _CONTEXT_STACK and _INTERRUPT_CONTEXT is _CONTEXT_STACK[-1]:
        raise AgentInterrupt("Interrupt requested")


# New helper functions for run_agent_with_retry refactoring
def _setup_interrupt_handling():
    if threading.current_thread() is threading.main_thread():
        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _request_interrupt)
        return original_handler
    return None


def _restore_interrupt_handling(original_handler):
    if original_handler and threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, original_handler)


def reset_agent_completion_flags():
    """Reset completion flags in the current context."""
    reset_completion_flags()


def _execute_test_command_wrapper(original_prompt, config, test_attempts, auto_test):
    # For backwards compatibility, allow passing of config directly
    # No need to get config from repository as it's passed in
    return execute_test_command(config, original_prompt, test_attempts, auto_test)


def _handle_api_error(e, attempt, max_retries, base_delay):
    # 1. Check if this is a ValueError with 429 code or rate limit phrases
    if isinstance(e, ValueError):
        error_str = str(e).lower()
        rate_limit_phrases = [
            "429",
            "rate limit",
            "too many requests",
            "quota exceeded",
        ]
        if "code" not in error_str and not any(
            phrase in error_str for phrase in rate_limit_phrases
        ):
            raise e

    # 2. Check for status_code or http_status attribute equal to 429
    if hasattr(e, "status_code") and e.status_code == 429:
        pass  # This is a rate limit error, continue with retry logic
    elif hasattr(e, "http_status") and e.http_status == 429:
        pass  # This is a rate limit error, continue with retry logic
    # 3. Check for rate limit phrases in error message
    elif isinstance(e, Exception) and not isinstance(e, ValueError):
        error_str = str(e).lower()
        if not any(
            phrase in error_str
            for phrase in ["rate limit", "too many requests", "quota exceeded", "429"]
        ) and not ("rate" in error_str and "limit" in error_str):
            # This doesn't look like a rate limit error, but we'll still retry other API errors
            pass

    # Apply common retry logic for all identified errors
    if attempt == max_retries - 1:
        logger.error("Max retries reached, failing: %s", str(e))
        raise RuntimeError(f"Max retries ({max_retries}) exceeded. Last error: {e}")

    logger.warning("API error (attempt %d/%d): %s", attempt + 1, max_retries, str(e))
    delay = base_delay * (2**attempt)
    error_message = f"Encountered {e.__class__.__name__}: {e}. Retrying in {delay}s... (Attempt {attempt+1}/{max_retries})"

    trajectory_repo = get_trajectory_repository()
    human_input_id = get_human_input_repository().get_most_recent_id()

    trajectory_repo.create(
        step_data={
            "error_message": error_message,
            "display_title": "Error",
        },
        record_type="error",
        human_input_id=human_input_id,
        is_error=True,
        error_message=error_message,
    )

    print_error(error_message)
    start = time.monotonic()
    while time.monotonic() - start < delay:
        check_interrupt()
        time.sleep(0.1)


def get_agent_type(agent: RAgents) -> Literal["CiaynAgent", "React"]:
    """
    Determines the type of the agent.
    Returns "CiaynAgent" if agent is an instance of CiaynAgent, otherwise "React".
    """

    if isinstance(agent, CiaynAgent):
        return "CiaynAgent"
    else:
        return "React"


def init_fallback_handler(agent: RAgents, tools: List[Any]):
    """
    Initialize fallback handler if agent is of type "React" and experimental_fallback_handler is enabled; otherwise return None.
    """
    if not get_config_repository().get("experimental_fallback_handler", False):
        return None
    agent_type = get_agent_type(agent)
    if agent_type == "React":
        # Create a dict with only the necessary config values for the FallbackHandler
        fallback_tool_model_limit = get_config_repository().get(
            "fallback_tool_model_limit", None
        )
        retry_fallback_count = get_config_repository().get("retry_fallback_count", None)
        provider = get_config_repository().get("provider", "anthropic")
        model = get_config_repository().get("model", "")

        config_for_fallback = {
            "fallback_tool_model_limit": fallback_tool_model_limit,
            "retry_fallback_count": retry_fallback_count,
            "provider": provider,
            "model": model,
        }

        return FallbackHandler(config_for_fallback, tools)
    return None


def _handle_fallback_response(
    error: ToolExecutionError,
    fallback_handler: Optional[FallbackHandler],
    agent: RAgents,
    msg_list: list,
) -> None:
    """
    Handle fallback response by invoking fallback_handler and updating msg_list.
    """
    if not fallback_handler:
        return
    fallback_response = fallback_handler.handle_failure(error, agent, msg_list)
    agent_type = get_agent_type(agent)
    if fallback_response and agent_type == "React":
        msg_list_response = [HumanMessage(str(msg)) for msg in fallback_response]
        msg_list.extend(msg_list_response)


def initialize_callback_handler(agent: RAgents):
    """
    Initialize the callback handler for token tracking.

    Args:
        agent: The agent instance to extract model information from

    Returns:
        tuple: (callback_handler, stream_config) - The callback handler and updated stream config
    """
    config = get_config_repository()
    stream_config_dict = config.deep_copy().to_dict()
    cb = None

    if not config.get("track_cost", True):
        logger.debug("Cost tracking is disabled, skipping callback handler")
        return cb, stream_config_dict

    # Only supporting anthropic ReAct Agent for now
    if not isinstance(agent, CompiledGraph):
        return cb, stream_config_dict

    model_name = DEFAULT_MODEL

    if agent is not None and hasattr(agent, "name"):
        model_name = agent.name
    else:
        logger.warning(
            "Agent is None or has no name attribute - the agent name is needed to determine enablement of callback handler."
        )

    # Always use the callback handler regardless of model name
    logger.debug(f"Using callback handler for model {model_name}")

    cb = AnthropicCallbackHandler(model_name)

    # Add callback to callbacks list in the dictionary
    if "callbacks" not in stream_config_dict:
        stream_config_dict["callbacks"] = []
    stream_config_dict["callbacks"].append(cb)

    return cb, stream_config_dict


def _prepare_state_config(config: Dict[str, Any]):
    """
    Prepare the state configuration for agent.get_state().

    Args:
        config: The configuration dictionary to update

    Returns:
        dict: The updated configuration dictionary
    """
    config_repo = get_config_repository()
    thread_id = config_repo.get("thread_id", str(uuid.uuid4()))

    if "configurable" not in config:
        config["configurable"] = {}

    config["configurable"]["thread_id"] = thread_id
    return config


def _get_agent_state(agent: RAgents, state_config: Dict[str, Any]):
    """
    Safely retrieve the agent state.

    Args:
        agent: The agent instance
        state_config: The state configuration

    Returns:
        The agent state

    Raises:
        Exception: If there's an error retrieving the agent state
    """
    try:
        state = agent.get_state(state_config)
        logger.debug("Agent state retrieved: %s", state)
        return state
    except Exception as e:
        logger.error(
            "Error retrieving agent state with state_config %s: %s", state_config, e
        )
        raise


def _run_agent_stream(agent: RAgents, msg_list: list[BaseMessage]):
    """
    Streams agent output while handling completion and interruption.

    For each chunk, it logs the output, calls check_interrupt(), prints agent output,
    and then checks if is_completed() or should_exit() are true. If so, it resets completion
    flags and returns. After finishing a stream iteration (i.e. the for-loop over chunks),
    the function retrieves the agent's state. If the state indicates further steps (i.e. state.next is non-empty),
    it resumes execution via agent.invoke(None, config); otherwise, it exits the loop.

    This function adheres to the latest LangGraph best practices (as of March 2025) for handling
    human-in-the-loop interruptions using interrupt_after=["tools"].
    """
    _cb, stream_config = initialize_callback_handler(agent)
    stream_config = _prepare_state_config(stream_config)

    while True:
        logger.debug("Using stream_config for agent.stream(): %s", stream_config)
        for chunk in agent.stream({"messages": msg_list}, stream_config):
            logger.debug("Agent output: %s", chunk)
            check_interrupt()
            agent_type = get_agent_type(agent)
            print_agent_output(chunk, agent_type)

            if is_completed() or should_exit():
                reset_completion_flags()
                return True

        logger.debug("Stream iteration ended; checking agent state for continuation.")

        state = _get_agent_state(agent, stream_config)

        if state.next:
            logger.debug(f"Continuing execution with state.next: {state.next}")
            agent.invoke(None, stream_config)
            continue
        else:
            logger.debug("No continuation indicated in state; exiting stream loop.")
            break

    return True


def run_agent_with_retry(
    agent: RAgents,
    prompt: str,
    fallback_handler: Optional[FallbackHandler] = None,
) -> Optional[str]:
    """Run an agent with retry logic for API errors."""
    logger.debug("Running agent with prompt length: %d", len(prompt))
    original_handler = _setup_interrupt_handling()
    max_retries = 20
    base_delay = 1
    test_attempts = 0

    _max_test_retries = get_config_repository().get(
        "max_test_cmd_retries", DEFAULT_MAX_TEST_CMD_RETRIES
    )
    auto_test = get_config_repository().get("auto_test", False)

    # Create run_config with only the values needed by execute_test_command
    original_prompt = prompt
    msg_list = [HumanMessage(content=prompt)]

    # Get all values needed for run_config
    test_cmd = get_config_repository().get("test_cmd", None)
    max_test_cmd_retries = get_config_repository().get(
        "max_test_cmd_retries", DEFAULT_MAX_TEST_CMD_RETRIES
    )
    test_cmd_timeout = get_config_repository().get("test_cmd_timeout", None)

    run_config = {
        "test_cmd": test_cmd,
        "max_test_cmd_retries": max_test_cmd_retries,
        "test_cmd_timeout": test_cmd_timeout,
        "auto_test": auto_test,
    }

    # Create a new agent context for this run
    with InterruptibleSection(), agent_context() as ctx:
        try:
            for attempt in range(max_retries):
                logger.debug("Attempt %d/%d", attempt + 1, max_retries)
                check_interrupt()

                # Check if the agent has crashed before attempting to run it
                from ra_aid.agent_context import get_crash_message, is_crashed

                if is_crashed():
                    crash_message = get_crash_message()
                    logger.error("Agent has crashed: %s", crash_message)
                    return f"Agent has crashed: {crash_message}"

                try:
                    _run_agent_stream(agent, msg_list)
                    if fallback_handler and hasattr(
                        fallback_handler, "reset_fallback_handler"
                    ):
                        fallback_handler.reset_fallback_handler()
                    should_break, prompt, auto_test, test_attempts = (
                        _execute_test_command_wrapper(
                            original_prompt, run_config, test_attempts, auto_test
                        )
                    )
                    if should_break:
                        break
                    if prompt != original_prompt:
                        continue

                    logger.debug("Agent run completed successfully")
                    return "Agent run completed successfully"
                except ToolExecutionError as e:
                    # Check if this is a BadRequestError (HTTP 400) which is unretryable
                    error_str = str(e).lower()
                    if "400" in error_str or "bad request" in error_str:
                        from ra_aid.agent_context import mark_agent_crashed

                        crash_message = f"Unretryable error: {str(e)}"
                        mark_agent_crashed(crash_message)
                        logger.error("Agent has crashed: %s", crash_message)
                        return f"Agent has crashed: {crash_message}"

                    _handle_fallback_response(e, fallback_handler, agent, msg_list)
                    continue
                except FallbackToolExecutionError as e:
                    msg_list.append(
                        SystemMessage(f"FallbackToolExecutionError:{str(e)}")
                    )
                except (KeyboardInterrupt, AgentInterrupt):
                    raise
                except (
                    InternalServerError,
                    APITimeoutError,
                    RateLimitError,
                    OpenAIRateLimitError,
                    LiteLLMRateLimitError,
                    ResourceExhausted,
                    APIError,
                    ValueError,
                ) as e:
                    # Check if this is a BadRequestError (HTTP 400) which is unretryable
                    error_str = str(e).lower()
                    if (
                        "400" in error_str or "bad request" in error_str
                    ) and isinstance(e, APIError):
                        from ra_aid.agent_context import mark_agent_crashed

                        crash_message = f"Unretryable API error: {str(e)}"
                        mark_agent_crashed(crash_message)
                        logger.error("Agent has crashed: %s", crash_message)
                        return f"Agent has crashed: {crash_message}"

                    _handle_api_error(e, attempt, max_retries, base_delay)
        finally:
            _restore_interrupt_handling(original_handler)
