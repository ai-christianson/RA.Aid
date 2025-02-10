"""Core agent functionality used by both agent_utils and tools/agent."""

import signal
import sys
import threading
import time
import uuid
from typing import Any, Optional

import litellm
from langchain_core.language_models import BaseChatModel
from litellm import get_model_info
from langchain_core.messages import BaseMessage, HumanMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ra_aid.agents.ciayn_agent import CiaynAgent
from ra_aid.console.formatting import print_error
from ra_aid.console.output import print_agent_output
from ra_aid.exceptions import AgentInterrupt
from ra_aid.logging_config import get_logger
from ra_aid.models_params import DEFAULT_TOKEN_LIMIT, models_params
from ra_aid.tools.handle_user_defined_test_cmd_execution import execute_test_command
from ra_aid.tools.memory import _global_memory, get_memory_value

console = Console()
logger = get_logger(__name__)

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


def create_agent(
    model: BaseChatModel,
    tools: list[Any],
    *,
    checkpointer: Any = None,
    agent_type: str = "default",
) -> Any:
    """Create a react agent with the given configuration.

    Args:
        model: The LLM model to use
        tools: List of tools to provide to the agent
        checkpointer: Optional memory checkpointer
        agent_type: Type of agent being created (default, research, planner)

    Returns:
        The created agent instance
    """
    try:
        config = _global_memory.get("config", {})
        max_input_tokens = get_model_token_limit(config, agent_type) or DEFAULT_TOKEN_LIMIT

        # Use REACT agent for Anthropic Claude models, otherwise use CIAYN
        if is_anthropic_claude(config):
            logger.debug("Using create_react_agent to instantiate agent.")
            agent_kwargs = build_agent_kwargs(checkpointer, config, max_input_tokens)
            return create_react_agent(model, tools, **agent_kwargs)
        else:
            logger.debug("Using CiaynAgent agent instance")
            return CiaynAgent(model, tools, max_tokens=max_input_tokens)

    except Exception as e:
        # Default to REACT agent if provider/model detection fails
        logger.warning(f"Failed to detect model type: {e}. Defaulting to REACT agent.")
        config = _global_memory.get("config", {})
        max_input_tokens = get_model_token_limit(config, agent_type)
        agent_kwargs = build_agent_kwargs(checkpointer, config, max_input_tokens)
        return create_react_agent(model, tools, **agent_kwargs)


def run_agent_with_retry(agent, prompt: str, config: dict) -> Optional[str]:
    """Run an agent with retry logic for API errors."""
    logger.debug("Running agent with prompt length: %d", len(prompt))
    original_handler = None
    if threading.current_thread() is threading.main_thread():
        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _request_interrupt)

    max_retries = 20
    base_delay = 1
    test_attempts = 0
    _max_test_retries = config.get("max_test_cmd_retries", 10)
    auto_test = config.get("auto_test", False)
    original_prompt = prompt

    with InterruptibleSection():
        try:
            # Track agent execution depth
            current_depth = _global_memory.get("agent_depth", 0)
            _global_memory["agent_depth"] = current_depth + 1

            for attempt in range(max_retries):
                logger.debug("Attempt %d/%d", attempt + 1, max_retries)
                check_interrupt()
                try:
                    for chunk in agent.stream(
                        {"messages": [HumanMessage(content=prompt)]}, config
                    ):
                        logger.debug("Agent output: %s", chunk)
                        check_interrupt()
                        print_agent_output(chunk)
                        if _global_memory["plan_completed"]:
                            _global_memory["plan_completed"] = False
                            _global_memory["task_completed"] = False
                            _global_memory["completion_message"] = ""
                            break
                        if _global_memory["task_completed"]:
                            _global_memory["task_completed"] = False
                            _global_memory["completion_message"] = ""
                            break

                    # Execute test command if configured
                    should_break, prompt, auto_test, test_attempts = execute_test_command(
                        config, original_prompt, test_attempts, auto_test
                    )
                    if should_break:
                        break
                    if prompt != original_prompt:
                        continue

                    logger.debug("Agent run completed successfully")
                    return "Agent run completed successfully"
                except (KeyboardInterrupt, AgentInterrupt):
                    raise
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error("Max retries reached, failing: %s", str(e))
                        raise RuntimeError(
                            f"Max retries ({max_retries}) exceeded. Last error: {e}"
                        )
                    logger.warning(
                        "API error (attempt %d/%d): %s",
                        attempt + 1,
                        max_retries,
                        str(e),
                    )
                    delay = base_delay * (2**attempt)
                    print_error(
                        f"Encountered {e.__class__.__name__}: {e}. Retrying in {delay}s... (Attempt {attempt+1}/{max_retries})"
                    )
                    start = time.monotonic()
                    while time.monotonic() - start < delay:
                        check_interrupt()
                        time.sleep(0.1)
        finally:
            # Reset depth tracking
            _global_memory["agent_depth"] = _global_memory.get("agent_depth", 1) - 1

            if original_handler and threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGINT, original_handler)


def get_model_token_limit(
    config: dict[str, Any], agent_type: str
) -> Optional[int]:
    """Get the token limit for the current model configuration based on agent type."""
    try:
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
            return max_input_tokens
        
        logger.debug(f"Could not find token limit for {provider}/{model_name}")
        return None

    except Exception as e:
        logger.warning(f"Failed to get model token limit: {e}")
        return None


def build_agent_kwargs(
    checkpointer: Optional[Any] = None,
    config: dict[str, Any] = None,
    max_input_tokens: Optional[int] = None,
) -> dict[str, Any]:
    """Build kwargs dictionary for agent creation."""
    agent_kwargs = {}

    if checkpointer is not None:
        agent_kwargs["checkpointer"] = checkpointer

    if config.get("limit_tokens", True) and is_anthropic_claude(config):

        def wrapped_state_modifier(state: AgentState) -> list[BaseMessage]:
            return state_modifier(state, max_input_tokens=max_input_tokens)

        agent_kwargs["state_modifier"] = wrapped_state_modifier

    return agent_kwargs


def is_anthropic_claude(config: dict[str, Any]) -> bool:
    """Check if the provider and model name indicate an Anthropic Claude model."""
    provider = config.get("provider", "")
    model_name = config.get("model", "")
    return (
        provider.lower() == "anthropic"
        and model_name
        and "claude" in model_name.lower()
    )


def state_modifier(
    state: AgentState, max_input_tokens: int = DEFAULT_TOKEN_LIMIT
) -> list[BaseMessage]:
    """Given the agent state and max_tokens, return a trimmed list of messages."""
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


def estimate_messages_tokens(messages: list[BaseMessage]) -> int:
    """Helper function to estimate total tokens in a sequence of messages."""
    if not messages:
        return 0

    estimate_tokens = CiaynAgent._estimate_tokens
    return sum(estimate_tokens(msg) for msg in messages)