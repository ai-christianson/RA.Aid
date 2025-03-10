"""Utility functions for working with agents."""

import inspect
import os
import signal
import sys
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Sequence

from ra_aid.callbacks.anthropic_callback_handler import AnthropicCallbackHandler


import litellm
from anthropic import APIError, APITimeoutError, InternalServerError, RateLimitError
from openai import RateLimitError as OpenAIRateLimitError
from litellm.exceptions import RateLimitError as LiteLLMRateLimitError
from google.api_core.exceptions import ResourceExhausted
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from litellm import get_model_info
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ra_aid.agent_context import (
    agent_context,
    get_depth,
    is_completed,
    reset_completion_flags,
    should_exit,
)
from ra_aid.agent_backends.ciayn_agent import CiaynAgent
from ra_aid.agents_alias import RAgents
from ra_aid.config import DEFAULT_MAX_TEST_CMD_RETRIES, DEFAULT_RECURSION_LIMIT
from ra_aid.console.formatting import print_error, print_stage_header
from ra_aid.console.output import print_agent_output
from ra_aid.exceptions import (
    AgentInterrupt,
    FallbackToolExecutionError,
    ToolExecutionError,
)
from ra_aid.fallback_handler import FallbackHandler
from ra_aid.logging_config import get_logger
from ra_aid.llm import initialize_expert_llm
from ra_aid.models_params import DEFAULT_TOKEN_LIMIT, models_params
from ra_aid.text.processing import process_thinking_content
from ra_aid.project_info import (
    display_project_status,
    format_project_info,
    get_project_info,
)
from ra_aid.prompts.expert_prompts import (
    EXPERT_PROMPT_SECTION_IMPLEMENTATION,
    EXPERT_PROMPT_SECTION_PLANNING,
    EXPERT_PROMPT_SECTION_RESEARCH,
)
from ra_aid.prompts.human_prompts import (
    HUMAN_PROMPT_SECTION_IMPLEMENTATION,
    HUMAN_PROMPT_SECTION_PLANNING,
    HUMAN_PROMPT_SECTION_RESEARCH,
)
from ra_aid.prompts.implementation_prompts import IMPLEMENTATION_PROMPT
from ra_aid.prompts.common_prompts import NEW_PROJECT_HINTS
from ra_aid.prompts.planning_prompts import PLANNING_PROMPT
from ra_aid.prompts.reasoning_assist_prompt import (
    REASONING_ASSIST_PROMPT_PLANNING,
    REASONING_ASSIST_PROMPT_IMPLEMENTATION,
    REASONING_ASSIST_PROMPT_RESEARCH,
)
from ra_aid.prompts.research_prompts import (
    RESEARCH_ONLY_PROMPT,
    RESEARCH_PROMPT,
)
from ra_aid.prompts.web_research_prompts import (
    WEB_RESEARCH_PROMPT,
    WEB_RESEARCH_PROMPT_SECTION_CHAT,
    WEB_RESEARCH_PROMPT_SECTION_PLANNING,
    WEB_RESEARCH_PROMPT_SECTION_RESEARCH,
)
from ra_aid.tool_configs import (
    get_implementation_tools,
    get_planning_tools,
    get_research_tools,
    get_web_research_tools,
)
from ra_aid.tools.handle_user_defined_test_cmd_execution import execute_test_command
from ra_aid.database.repositories.key_fact_repository import get_key_fact_repository
from ra_aid.database.repositories.key_snippet_repository import (
    get_key_snippet_repository,
)
from ra_aid.database.repositories.human_input_repository import (
    get_human_input_repository,
)
from ra_aid.database.repositories.research_note_repository import (
    get_research_note_repository,
)
from ra_aid.database.repositories.work_log_repository import get_work_log_repository
from ra_aid.model_formatters import format_key_facts_dict
from ra_aid.model_formatters.key_snippets_formatter import format_key_snippets_dict
from ra_aid.model_formatters.research_notes_formatter import format_research_notes_dict
from ra_aid.tools.memory import (
    get_related_files,
    log_work_event,
)
from ra_aid.database.repositories.config_repository import get_config_repository
from ra_aid.env_inv_context import get_env_inv

console = Console()

logger = get_logger(__name__)

# Import repositories using get_* functions
from ra_aid.database.repositories.key_fact_repository import get_key_fact_repository


@tool
def output_markdown_message(message: str) -> str:
    """Outputs a message to the user, optionally prompting for input."""
    console.print(Panel(Markdown(message.strip()), title="🤖 Assistant"))
    return "Message output."


def estimate_messages_tokens(messages: Sequence[BaseMessage]) -> int:
    """Helper function to estimate total tokens in a sequence of messages.

    Args:
        messages: Sequence of messages to count tokens for

    Returns:
        Total estimated token count
    """
    if not messages:
        return 0

    estimate_tokens = CiaynAgent._estimate_tokens
    return sum(estimate_tokens(msg) for msg in messages)


def state_modifier(
    state: AgentState, max_input_tokens: int = DEFAULT_TOKEN_LIMIT
) -> list[BaseMessage]:
    """Given the agent state and max_tokens, return a trimmed list of messages.

    Args:
        state: The current agent state containing messages
        max_tokens: Maximum number of tokens to allow (default: DEFAULT_TOKEN_LIMIT)

    Returns:
        list[BaseMessage]: Trimmed list of messages that fits within token limit
    """
    messages = state["messages"]

    if not messages:
        return []

    first_message = messages[0]
    remaining_messages = messages[1:]
    first_tokens = estimate_messages_tokens([first_message])
    new_max_tokens = max_input_tokens - first_tokens

    trimmed_remaining = trim_messages(
        remaining_messages,
        token_counter=estimate_messages_tokens,
        max_tokens=new_max_tokens,
        strategy="last",
        allow_partial=False,
    )

    return [first_message] + trimmed_remaining


def get_model_token_limit(
    config: Dict[str, Any], agent_type: Literal["default", "research", "planner"]
) -> Optional[int]:
    """Get the token limit for the current model configuration based on agent type.

    Returns:
        Optional[int]: The token limit if found, None otherwise
    """
    try:
        # Try to get config from repository for production use
        try:
            config_from_repo = get_config_repository().get_all()
            # If we succeeded, use the repository config instead of passed config
            config = config_from_repo
        except RuntimeError:
            # In tests, this may fail because the repository isn't set up
            # So we'll use the passed config directly
            pass
        if agent_type == "research":
            provider = config.get("research_provider", "") or config.get("provider", "")
            model_name = config.get("research_model", "") or config.get("model", "")
        elif agent_type == "planner":
            provider = config.get("planner_provider", "") or config.get("provider", "")
            model_name = config.get("planner_model", "") or config.get("model", "")
        else:
            provider = config.get("provider", "")
            model_name = config.get("model", "")

        try:
            provider_model = model_name if not provider else f"{provider}/{model_name}"
            model_info = get_model_info(provider_model)
            max_input_tokens = model_info.get("max_input_tokens")
            if max_input_tokens:
                logger.debug(
                    f"Using litellm token limit for {model_name}: {max_input_tokens}"
                )
                return max_input_tokens
        except litellm.exceptions.NotFoundError:
            logger.debug(
                f"Model {model_name} not found in litellm, falling back to models_params"
            )
        except Exception as e:
            logger.debug(
                f"Error getting model info from litellm: {e}, falling back to models_params"
            )

        # Fallback to models_params dict
        # Normalize model name for fallback lookup (e.g. claude-2 -> claude2)
        normalized_name = model_name.replace("-", "")
        provider_tokens = models_params.get(provider, {})
        if normalized_name in provider_tokens:
            max_input_tokens = provider_tokens[normalized_name]["token_limit"]
            logger.debug(
                f"Found token limit for {provider}/{model_name}: {max_input_tokens}"
            )
        else:
            max_input_tokens = None
            logger.debug(f"Could not find token limit for {provider}/{model_name}")

        return max_input_tokens

    except Exception as e:
        logger.warning(f"Failed to get model token limit: {e}")
        return None


def build_agent_kwargs(
    checkpointer: Optional[Any] = None,
    max_input_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Build kwargs dictionary for agent creation.

    Args:
        checkpointer: Optional memory checkpointer
        config: Optional configuration dictionary
        token_limit: Optional token limit for the model

    Returns:
        Dictionary of kwargs for agent creation
    """
    agent_kwargs = {
        "version": "v2",
    }

    if checkpointer is not None:
        agent_kwargs["checkpointer"] = checkpointer

    config = get_config_repository().get_all()
    if config.get("limit_tokens", True) and is_anthropic_claude(config):

        def wrapped_state_modifier(state: AgentState) -> list[BaseMessage]:
            return state_modifier(state, max_input_tokens=max_input_tokens)

        agent_kwargs["state_modifier"] = wrapped_state_modifier

    return agent_kwargs


def is_anthropic_claude(config: Dict[str, Any]) -> bool:
    """Check if the provider and model name indicate an Anthropic Claude model.

    Args:
        config: Configuration dictionary containing provider and model information

    Returns:
        bool: True if this is an Anthropic Claude model
    """
    # For backwards compatibility, allow passing of config directly
    provider = config.get("provider", "")
    model_name = config.get("model", "")
    result = (
        provider.lower() == "anthropic"
        and model_name
        and "claude" in model_name.lower()
    ) or (
        provider.lower() == "openrouter"
        and model_name.lower().startswith("anthropic/claude-")
    )
    return result


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
        config: Optional configuration dictionary containing settings like:
            - limit_tokens (bool): Whether to apply token limiting (default: True)
            - provider (str): The LLM provider name
            - model (str): The model name

    Returns:
        The created agent instance

    Token limiting helps prevent context window overflow by trimming older messages
    while preserving system messages. It can be disabled by setting
    config['limit_tokens'] = False.
    """
    try:
        # Try to get config from repository for production use
        try:
            config_from_repo = get_config_repository().get_all()
            # If we succeeded, use the repository config instead of passed config
            config = config_from_repo
        except RuntimeError:
            # In tests, this may fail because the repository isn't set up
            # So we'll use the passed config directly
            pass
        max_input_tokens = (
            get_model_token_limit(config, agent_type) or DEFAULT_TOKEN_LIMIT
        )

        # Use REACT agent for Anthropic Claude models, otherwise use CIAYN
        if is_anthropic_claude(config):
            logger.debug("Using create_react_agent to instantiate agent.")
            agent_kwargs = build_agent_kwargs(checkpointer, max_input_tokens)
            return create_react_agent(
                model, tools, interrupt_after=["tools"], **agent_kwargs
            )
        else:
            logger.debug("Using CiaynAgent agent instance")
            return CiaynAgent(model, tools, max_tokens=max_input_tokens, config=config)

    except Exception as e:
        # Default to REACT agent if provider/model detection fails
        logger.warning(f"Failed to detect model type: {e}. Defaulting to REACT agent.")
        config = get_config_repository().get_all()
        max_input_tokens = get_model_token_limit(config, agent_type)
        agent_kwargs = build_agent_kwargs(checkpointer, max_input_tokens)
        return create_react_agent(
            model, tools, interrupt_after=["tools"], **agent_kwargs
        )


from ra_aid.agents.research_agent import run_research_agent, run_web_research_agent
from ra_aid.agents.implementation_agent import run_task_implementation_agent


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
    print_error(
        f"Encountered {e.__class__.__name__}: {e}. Retrying in {delay}s... (Attempt {attempt+1}/{max_retries})"
    )
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
        return FallbackHandler(get_config_repository().get_all(), tools)
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
    config = get_config_repository().get_all()
    stream_config = config.copy()

    cb = None
    if is_anthropic_claude(config):
        model_name = config.get("model", "")
        full_model_name = model_name
        cb = AnthropicCallbackHandler(full_model_name)

        if "callbacks" not in stream_config:
            stream_config["callbacks"] = []
        stream_config["callbacks"].append(cb)

    while True:
        for chunk in agent.stream({"messages": msg_list}, stream_config):
            logger.debug("Agent output: %s", chunk)
            check_interrupt()
            agent_type = get_agent_type(agent)
            print_agent_output(chunk, agent_type)

            if is_completed() or should_exit():
                reset_completion_flags()
                if cb:
                    logger.debug(f"AnthropicCallbackHandler:\n{cb}")
                return True

        logger.debug("Stream iteration ended; checking agent state for continuation.")

        # Prepare state configuration, ensuring 'configurable' is present.
        state_config = get_config_repository().get_all().copy()
        if "configurable" not in state_config:
            logger.debug(
                "Key 'configurable' not found in config; adding it as an empty dict."
            )
            state_config["configurable"] = {}
        logger.debug("Using state_config for agent.get_state(): %s", state_config)

        try:
            state = agent.get_state(state_config)
            logger.debug("Agent state retrieved: %s", state)
        except Exception as e:
            logger.error(
                "Error retrieving agent state with state_config %s: %s", state_config, e
            )
            raise

        if state.next:
            logger.debug(
                "State indicates continuation (state.next: %s); resuming execution.",
                state.next,
            )
            agent.invoke(None, stream_config)
            continue
        else:
            logger.debug("No continuation indicated in state; exiting stream loop.")
            break

    if cb:
        logger.debug(f"AnthropicCallbackHandler:\n{cb}")
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
    original_prompt = prompt
    msg_list = [HumanMessage(content=prompt)]
    run_config = get_config_repository().get_all()

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
                    if fallback_handler:
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
