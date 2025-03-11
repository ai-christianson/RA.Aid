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
<<<<<<< HEAD
        )


def run_research_agent(
    base_task_or_query: str,
    model,
    *,
    expert_enabled: bool = False,
    research_only: bool = False,
    hil: bool = False,
    web_research_enabled: bool = False,
    memory: Optional[Any] = None,
    thread_id: Optional[str] = None,
    console_message: Optional[str] = None,
) -> Optional[str]:
    """Run a research agent with the given configuration.

    Args:
        base_task_or_query: The main task or query for research
        model: The LLM model to use
        expert_enabled: Whether expert mode is enabled
        research_only: Whether this is a research-only task
        hil: Whether human-in-the-loop mode is enabled
        web_research_enabled: Whether web research is enabled
        memory: Optional memory instance to use
        config: Optional configuration dictionary
        thread_id: Optional thread ID (defaults to new UUID)
        console_message: Optional message to display before running

    Returns:
        Optional[str]: The completion message if task completed successfully

    Example:
        result = run_research_agent(
            "Research Python async patterns",
            model,
            expert_enabled=True,
            research_only=True
        )
    """
    thread_id = thread_id or str(uuid.uuid4())
    logger.debug("Starting research agent with thread_id=%s", thread_id)
    logger.debug(
        "Research configuration: expert=%s, research_only=%s, hil=%s, web=%s",
        expert_enabled,
        research_only,
        hil,
        web_research_enabled,
    )

    if memory is None:
        memory = MemorySaver()

    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    working_directory = os.getcwd()

    # Get the last human input, if it exists
    base_task = base_task_or_query
    try:
        human_input_repository = get_human_input_repository()
        recent_inputs = human_input_repository.get_recent(1)
        if recent_inputs and len(recent_inputs) > 0:
            last_human_input = recent_inputs[0].content
            base_task = (
                f"<last human input>{last_human_input}</last human input>\n{base_task}"
            )
    except RuntimeError as e:
        logger.error(f"Failed to access human input repository: {str(e)}")
        # Continue without appending last human input

    try:
        key_facts = format_key_facts_dict(get_key_fact_repository().get_facts_dict())
    except RuntimeError as e:
        logger.error(f"Failed to access key fact repository: {str(e)}")
        key_facts = ""
    key_snippets = format_key_snippets_dict(
        get_key_snippet_repository().get_snippets_dict()
    )
    related_files = get_related_files()

    try:
        project_info = get_project_info(".", file_limit=2000)
        formatted_project_info = format_project_info(project_info)
    except Exception as e:
        logger.warning(f"Failed to get project info: {e}")
        formatted_project_info = ""

    tools = get_research_tools(
        research_only=research_only,
        expert_enabled=expert_enabled,
        human_interaction=hil,
        web_research_enabled=get_config_repository().get("web_research_enabled", False),
    )

    # Get model info for reasoning assistance configuration
    provider = get_config_repository().get("provider", "")
    model_name = get_config_repository().get("model", "")

    # Get model configuration to check for reasoning_assist_default
    model_config = {}
    provider_models = models_params.get(provider, {})
    if provider_models and model_name in provider_models:
        model_config = provider_models[model_name]

    # Check if reasoning assist is explicitly enabled/disabled
    force_assistance = get_config_repository().get("force_reasoning_assistance", False)
    disable_assistance = get_config_repository().get(
        "disable_reasoning_assistance", False
    )
    if force_assistance:
        reasoning_assist_enabled = True
    elif disable_assistance:
        reasoning_assist_enabled = False
    else:
        # Fall back to model default
        reasoning_assist_enabled = model_config.get("reasoning_assist_default", False)

    logger.debug("Reasoning assist enabled: %s", reasoning_assist_enabled)
    expert_guidance = ""

    # Get research note information for reasoning assistance
    try:
        research_notes = format_research_notes_dict(
            get_research_note_repository().get_notes_dict()
        )
    except Exception as e:
        logger.warning(f"Failed to get research notes: {e}")
        research_notes = ""

    # If reasoning assist is enabled, make a one-off call to the expert model
    if reasoning_assist_enabled:
        try:
            logger.info(
                "Reasoning assist enabled for model %s, getting expert guidance",
                model_name,
            )

            # Collect tool descriptions
            tool_metadata = []
            from ra_aid.tools.reflection import get_function_info as get_tool_info

            for tool in tools:
                try:
                    tool_info = get_tool_info(tool.func)
                    name = tool.func.__name__
                    description = inspect.getdoc(tool.func)
                    tool_metadata.append(f"Tool: {name}\nDescription: {description}\n")
                except Exception as e:
                    logger.warning(f"Error getting tool info for {tool}: {e}")

            # Format tool metadata
            formatted_tool_metadata = "\n".join(tool_metadata)

            # Initialize expert model
            expert_model = initialize_expert_llm(provider, model_name)

            # Format the reasoning assist prompt
            reasoning_assist_prompt = REASONING_ASSIST_PROMPT_RESEARCH.format(
                current_date=current_date,
                working_directory=working_directory,
                base_task=base_task,
                key_facts=key_facts,
                key_snippets=key_snippets,
                research_notes=research_notes,
                related_files=related_files,
                env_inv=get_env_inv(),
                tool_metadata=formatted_tool_metadata,
            )

            # Show the reasoning assist query in a panel
            console.print(
                Panel(
                    Markdown(
                        "Consulting with the reasoning model on the best research approach."
                    ),
                    title="📝 Thinking about research strategy...",
                    border_style="yellow",
                )
            )

            logger.debug("Invoking expert model for reasoning assist")
            # Make the call to the expert model
            response = expert_model.invoke(reasoning_assist_prompt)

            # Check if the model supports think tags
            supports_think_tag = model_config.get("supports_think_tag", False)
            supports_thinking = model_config.get("supports_thinking", False)

            # Get response content, handling if it's a list (for Claude thinking mode)
            content = None

            if hasattr(response, "content"):
                content = response.content
            else:
                # Fallback if content attribute is missing
                content = str(response)

            # Process content based on its type
            if isinstance(content, list):
                # Handle structured thinking mode (e.g., Claude 3.7)
                thinking_content = None
                response_text = None

                # Process each item in the list
                for item in content:
                    if isinstance(item, dict):
                        # Extract thinking content
                        if item.get("type") == "thinking" and "thinking" in item:
                            thinking_content = item["thinking"]
                            logger.debug("Found structured thinking content")
                        # Extract response text
                        elif item.get("type") == "text" and "text" in item:
                            response_text = item["text"]
                            logger.debug("Found structured response text")

                # Display thinking content in a separate panel if available
                if thinking_content and get_config_repository().get(
                    "show_thoughts", False
                ):
                    logger.debug(
                        f"Displaying structured thinking content ({len(thinking_content)} chars)"
                    )
                    console.print(
                        Panel(
                            Markdown(thinking_content),
                            title="💭 Expert Thinking",
                            border_style="yellow",
                        )
                    )

                # Use response_text if available, otherwise fall back to joining
                if response_text:
                    content = response_text
                else:
                    # Fallback: join list items if structured extraction failed
                    logger.debug(
                        "No structured response text found, joining list items"
                    )
                    content = "\n".join(str(item) for item in content)
            elif supports_think_tag or supports_thinking:
                # Process thinking content using the centralized function
                content, _ = process_thinking_content(
                    content=content,
                    supports_think_tag=supports_think_tag,
                    supports_thinking=supports_thinking,
                    panel_title="💭 Expert Thinking",
                    panel_style="yellow",
                    logger=logger,
                )

            # Display the expert guidance in a panel
            console.print(
                Panel(
                    Markdown(content),
                    title="Research Strategy Guidance",
                    border_style="blue",
                )
            )

            # Use the content as expert guidance
            expert_guidance = (
                content + "\n\nCONSULT WITH THE EXPERT FREQUENTLY DURING RESEARCH"
            )

            logger.info("Received expert guidance for research")
        except Exception as e:
            logger.error("Error getting expert guidance for research: %s", e)
            expert_guidance = ""

    agent = create_agent(model, tools, checkpointer=memory, agent_type="research")

    expert_section = EXPERT_PROMPT_SECTION_RESEARCH if expert_enabled else ""
    human_section = HUMAN_PROMPT_SECTION_RESEARCH if hil else ""
    web_research_section = (
        WEB_RESEARCH_PROMPT_SECTION_RESEARCH
        if get_config_repository().get("web_research_enabled")
        else ""
    )

    # Prepare expert guidance section if expert guidance is available
    expert_guidance_section = ""
    if expert_guidance:
        expert_guidance_section = f"""<expert guidance>
{expert_guidance}
</expert guidance>"""

    # Format research notes if available
    # We get research notes earlier for reasoning assistance

    # Get environment inventory information

    prompt = (RESEARCH_ONLY_PROMPT if research_only else RESEARCH_PROMPT).format(
        current_date=current_date,
        working_directory=working_directory,
        base_task=base_task,
        research_only_note=(
            ""
            if research_only
            else " Only request implementation if the user explicitly asked for changes to be made."
        ),
        expert_section=expert_section,
        human_section=human_section,
        web_research_section=web_research_section,
        key_facts=key_facts,
        work_log=get_work_log_repository().format_work_log(),
        key_snippets=key_snippets,
        related_files=related_files,
        project_info=formatted_project_info,
        new_project_hints=NEW_PROJECT_HINTS if project_info.is_new else "",
        env_inv=get_env_inv(),
        expert_guidance_section=expert_guidance_section,
    )

    config = get_config_repository().get_all()
    recursion_limit = config.get("recursion_limit", DEFAULT_RECURSION_LIMIT)
    run_config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }
    run_config.update(config)

    try:
        if console_message:
            console.print(
                Panel(Markdown(console_message), title="🔬 Looking into it...")
            )

        if project_info:
            display_project_status(project_info)

        if agent is not None:
            logger.debug("Research agent created successfully")
            none_or_fallback_handler = init_fallback_handler(agent, tools)
            _result = run_agent_with_retry(agent, prompt, none_or_fallback_handler)
            if _result:
                # Log research completion
                log_work_event(f"Completed research phase for: {base_task_or_query}")
            return _result
        else:
            logger.debug("No model provided, running web research tools directly")
            return run_web_research_agent(
                base_task_or_query,
                model=None,
                expert_enabled=expert_enabled,
                hil=hil,
                web_research_enabled=web_research_enabled,
                memory=memory,
                thread_id=thread_id,
                console_message=console_message,
            )
    except (KeyboardInterrupt, AgentInterrupt):
        raise
    except Exception as e:
        logger.error("Research agent failed: %s", str(e), exc_info=True)
        raise


def run_web_research_agent(
    query: str,
    model,
    *,
    expert_enabled: bool = False,
    hil: bool = False,
    web_research_enabled: bool = False,
    memory: Optional[Any] = None,
    thread_id: Optional[str] = None,
    console_message: Optional[str] = None,
) -> Optional[str]:
    """Run a web research agent with the given configuration.

    Args:
        query: The mainquery for web research
        model: The LLM model to use
        expert_enabled: Whether expert mode is enabled
        hil: Whether human-in-the-loop mode is enabled
        web_research_enabled: Whether web research is enabled
        memory: Optional memory instance to use
        config: Optional configuration dictionary
        thread_id: Optional thread ID (defaults to new UUID)
        console_message: Optional message to display before running

    Returns:
        Optional[str]: The completion message if task completed successfully

    Example:
        result = run_web_research_agent(
            "Research latest Python async patterns",
            model,
            expert_enabled=True
        )
    """
    thread_id = thread_id or str(uuid.uuid4())
    logger.debug("Starting web research agent with thread_id=%s", thread_id)
    logger.debug(
        "Web research configuration: expert=%s, hil=%s, web=%s",
        expert_enabled,
        hil,
        web_research_enabled,
    )

    if memory is None:
        memory = MemorySaver()

    if thread_id is None:
        thread_id = str(uuid.uuid4())

    tools = get_web_research_tools(expert_enabled=expert_enabled)

    agent = create_agent(model, tools, checkpointer=memory, agent_type="research")

    expert_section = EXPERT_PROMPT_SECTION_RESEARCH if expert_enabled else ""
    human_section = HUMAN_PROMPT_SECTION_RESEARCH if hil else ""

    try:
        key_facts = format_key_facts_dict(get_key_fact_repository().get_facts_dict())
    except RuntimeError as e:
        logger.error(f"Failed to access key fact repository: {str(e)}")
        key_facts = ""
    try:
        key_snippets = format_key_snippets_dict(
            get_key_snippet_repository().get_snippets_dict()
        )
    except RuntimeError as e:
        logger.error(f"Failed to access key snippet repository: {str(e)}")
        key_snippets = ""
    related_files = get_related_files()

    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    working_directory = os.getcwd()

    # Get environment inventory information

    prompt = WEB_RESEARCH_PROMPT.format(
        current_date=current_date,
        working_directory=working_directory,
        web_research_query=query,
        expert_section=expert_section,
        human_section=human_section,
        key_facts=key_facts,
        work_log=get_work_log_repository().format_work_log(),
        key_snippets=key_snippets,
        related_files=related_files,
        env_inv=get_env_inv(),
    )

    config = get_config_repository().get_all()

    recursion_limit = config.get("recursion_limit", DEFAULT_RECURSION_LIMIT)
    run_config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }
    if config:
        run_config.update(config)

    try:
        if console_message:
            console.print(Panel(Markdown(console_message), title="🔬 Researching..."))

        logger.debug("Web research agent completed successfully")
        none_or_fallback_handler = init_fallback_handler(agent, tools)
        _result = run_agent_with_retry(agent, prompt, none_or_fallback_handler)
        if _result:
            # Log web research completion
            log_work_event(f"Completed web research phase for: {query}")
        return _result

    except (KeyboardInterrupt, AgentInterrupt):
        raise
    except Exception as e:
        logger.error("Web research agent failed: %s", str(e), exc_info=True)
        raise


def run_planning_agent(
    base_task: str,
    model,
    *,
    expert_enabled: bool = False,
    hil: bool = False,
    memory: Optional[Any] = None,
    thread_id: Optional[str] = None,
) -> Optional[str]:
    """Run a planning agent to create implementation plans.

    Args:
        base_task: The main task to plan implementation for
        model: The LLM model to use
        expert_enabled: Whether expert mode is enabled
        hil: Whether human-in-the-loop mode is enabled
        memory: Optional memory instance to use
        thread_id: Optional thread ID (defaults to new UUID)

    Returns:
        Optional[str]: The completion message if planning completed successfully
    """
    thread_id = thread_id or str(uuid.uuid4())
    logger.debug("Starting planning agent with thread_id=%s", thread_id)
    logger.debug("Planning configuration: expert=%s, hil=%s", expert_enabled, hil)

    if memory is None:
        memory = MemorySaver()

    if thread_id is None:
        thread_id = str(uuid.uuid4())

    # Get latest project info
    try:
        project_info = get_project_info(".")
        formatted_project_info = format_project_info(project_info)
    except Exception as e:
        logger.warning("Failed to get project info: %s", str(e))
        formatted_project_info = "Project info unavailable"

    tools = get_planning_tools(
        expert_enabled=expert_enabled,
        web_research_enabled=get_config_repository().get("web_research_enabled", False),
    )

    # Get model configuration
    provider = get_config_repository().get("provider", "")
    model_name = get_config_repository().get("model", "")
    logger.debug("Checking for reasoning_assist_default on %s/%s", provider, model_name)

    # Get model configuration to check for reasoning_assist_default
    model_config = {}
    provider_models = models_params.get(provider, {})
    if provider_models and model_name in provider_models:
        model_config = provider_models[model_name]

    # Check if reasoning assist is explicitly enabled/disabled
    force_assistance = get_config_repository().get("force_reasoning_assistance", False)
    disable_assistance = get_config_repository().get(
        "disable_reasoning_assistance", False
    )

    if force_assistance:
        reasoning_assist_enabled = True
    elif disable_assistance:
        reasoning_assist_enabled = False
    else:
        # Fall back to model default
        reasoning_assist_enabled = model_config.get("reasoning_assist_default", False)

    logger.debug("Reasoning assist enabled: %s", reasoning_assist_enabled)

    # Get all the context information (used both for normal planning and reasoning assist)
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    working_directory = os.getcwd()

    # Make sure key_facts is defined before using it
    try:
        key_facts = format_key_facts_dict(get_key_fact_repository().get_facts_dict())
    except RuntimeError as e:
        logger.error(f"Failed to access key fact repository: {str(e)}")
        key_facts = ""

    # Make sure key_snippets is defined before using it
    try:
        key_snippets = format_key_snippets_dict(
            get_key_snippet_repository().get_snippets_dict()
        )
    except RuntimeError as e:
        logger.error(f"Failed to access key snippet repository: {str(e)}")
        key_snippets = ""

    # Get formatted research notes using repository
    try:
        repository = get_research_note_repository()
        notes_dict = repository.get_notes_dict()
        formatted_research_notes = format_research_notes_dict(notes_dict)
    except RuntimeError as e:
        logger.error(f"Failed to access research note repository: {str(e)}")
        formatted_research_notes = ""

    # Get related files
    related_files = "\n".join(get_related_files())

    # Get environment inventory information
    env_inv = get_env_inv()

    # Display the planning stage header before any reasoning assistance
    print_stage_header("Planning Stage")

    # Initialize expert guidance section
    expert_guidance = ""

    # If reasoning assist is enabled, make a one-off call to the expert model
    if reasoning_assist_enabled:
        try:
            logger.info(
                "Reasoning assist enabled for model %s, getting expert guidance",
                model_name,
            )

            # Collect tool descriptions
            tool_metadata = []
            from ra_aid.tools.reflection import get_function_info as get_tool_info

            for tool in tools:
                try:
                    tool_info = get_tool_info(tool.func)
                    name = tool.func.__name__
                    description = inspect.getdoc(tool.func)
                    tool_metadata.append(f"Tool: {name}\nDescription: {description}\n")
                except Exception as e:
                    logger.warning(f"Error getting tool info for {tool}: {e}")

            # Format tool metadata
            formatted_tool_metadata = "\n".join(tool_metadata)

            # Initialize expert model
            expert_model = initialize_expert_llm(provider, model_name)

            # Format the reasoning assist prompt
            reasoning_assist_prompt = REASONING_ASSIST_PROMPT_PLANNING.format(
                current_date=current_date,
                working_directory=working_directory,
                base_task=base_task,
                key_facts=key_facts,
                key_snippets=key_snippets,
                research_notes=formatted_research_notes,
                related_files=related_files,
                env_inv=env_inv,
                tool_metadata=formatted_tool_metadata,
            )

            # Show the reasoning assist query in a panel
            console.print(
                Panel(
                    Markdown(
                        "Consulting with the reasoning model on the best way to do this."
                    ),
                    title="📝 Thinking about the plan...",
                    border_style="yellow",
                )
            )

            logger.debug("Invoking expert model for reasoning assist")
            # Make the call to the expert model
            response = expert_model.invoke(reasoning_assist_prompt)

            # Check if the model supports think tags
            supports_think_tag = model_config.get("supports_think_tag", False)
            supports_thinking = model_config.get("supports_thinking", False)

            # Get response content, handling if it's a list (for Claude thinking mode)
            content = None

            if hasattr(response, "content"):
                content = response.content
            else:
                # Fallback if content attribute is missing
                content = str(response)

            # Process content based on its type
            if isinstance(content, list):
                # Handle structured thinking mode (e.g., Claude 3.7)
                thinking_content = None
                response_text = None

                # Process each item in the list
                for item in content:
                    if isinstance(item, dict):
                        # Extract thinking content
                        if item.get("type") == "thinking" and "thinking" in item:
                            thinking_content = item["thinking"]
                            logger.debug("Found structured thinking content")
                        # Extract response text
                        elif item.get("type") == "text" and "text" in item:
                            response_text = item["text"]
                            logger.debug("Found structured response text")

                # Display thinking content in a separate panel if available
                if thinking_content and get_config_repository().get(
                    "show_thoughts", False
                ):
                    logger.debug(
                        f"Displaying structured thinking content ({len(thinking_content)} chars)"
                    )
                    console.print(
                        Panel(
                            Markdown(thinking_content),
                            title="💭 Expert Thinking",
                            border_style="yellow",
                        )
                    )

                # Use response_text if available, otherwise fall back to joining
                if response_text:
                    content = response_text
                else:
                    # Fallback: join list items if structured extraction failed
                    logger.debug(
                        "No structured response text found, joining list items"
                    )
                    content = "\n".join(str(item) for item in content)
            elif supports_think_tag or supports_thinking:
                # Process thinking content using the centralized function
                content, _ = process_thinking_content(
                    content=content,
                    supports_think_tag=supports_think_tag,
                    supports_thinking=supports_thinking,
                    panel_title="💭 Expert Thinking",
                    panel_style="yellow",
                    logger=logger,
                )

            # Display the expert guidance in a panel
            console.print(
                Panel(
                    Markdown(content), title="Reasoning Guidance", border_style="blue"
                )
            )

            # Use the content as expert guidance
            expert_guidance = (
                content + "\n\nCONSULT WITH THE EXPERT FREQUENTLY ON THIS TASK"
            )

            logger.info("Received expert guidance for planning")
        except Exception as e:
            logger.error("Error getting expert guidance for planning: %s", e)
            expert_guidance = ""

    agent = create_agent(model, tools, checkpointer=memory, agent_type="planner")

    expert_section = EXPERT_PROMPT_SECTION_PLANNING if expert_enabled else ""
    human_section = HUMAN_PROMPT_SECTION_PLANNING if hil else ""
    web_research_section = (
        WEB_RESEARCH_PROMPT_SECTION_PLANNING
        if get_config_repository().get("web_research_enabled", False)
        else ""
    )

    # Prepare expert guidance section if expert guidance is available
    expert_guidance_section = ""
    if expert_guidance:
        expert_guidance_section = f"""<expert guidance>
{expert_guidance}
</expert guidance>"""

    planning_prompt = PLANNING_PROMPT.format(
        current_date=current_date,
        working_directory=working_directory,
        expert_section=expert_section,
        human_section=human_section,
        web_research_section=web_research_section,
        base_task=base_task,
        project_info=formatted_project_info,
        research_notes=formatted_research_notes,
        related_files=related_files,
        key_facts=key_facts,
        key_snippets=key_snippets,
        work_log=get_work_log_repository().format_work_log(),
        research_only_note=(
            ""
            if get_config_repository().get("research_only", False)
            else " Only request implementation if the user explicitly asked for changes to be made."
        ),
        env_inv=env_inv,
        expert_guidance_section=expert_guidance_section,
    )

    config_values = get_config_repository().get_all()
    recursion_limit = get_config_repository().get(
        "recursion_limit", DEFAULT_RECURSION_LIMIT
    )
    run_config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }
    run_config.update(config_values)

    try:
        logger.debug("Planning agent completed successfully")
        none_or_fallback_handler = init_fallback_handler(agent, tools)
        _result = run_agent_with_retry(agent, planning_prompt, none_or_fallback_handler)
        if _result:
            # Log planning completion
            log_work_event(f"Completed planning phase for: {base_task}")
        return _result
    except (KeyboardInterrupt, AgentInterrupt):
        raise
    except Exception as e:
        logger.error("Planning agent failed: %s", str(e), exc_info=True)
        raise


def run_task_implementation_agent(
    base_task: str,
    tasks: list,
    task: str,
    plan: str,
    related_files: list,
    model,
    *,
    expert_enabled: bool = False,
    web_research_enabled: bool = False,
    memory: Optional[Any] = None,
    thread_id: Optional[str] = None,
) -> Optional[str]:
    """Run an implementation agent for a specific task.

    Args:
        base_task: The main task being implemented
        tasks: List of tasks to implement
        plan: The implementation plan
        related_files: List of related files
        model: The LLM model to use
        expert_enabled: Whether expert mode is enabled
        web_research_enabled: Whether web research is enabled
        memory: Optional memory instance to use
        thread_id: Optional thread ID (defaults to new UUID)

    Returns:
        Optional[str]: The completion message if task completed successfully
    """
    thread_id = thread_id or str(uuid.uuid4())
    logger.debug("Starting implementation agent with thread_id=%s", thread_id)
    logger.debug(
        "Implementation configuration: expert=%s, web=%s",
        expert_enabled,
        web_research_enabled,
    )
    logger.debug("Task details: base_task=%s, current_task=%s", base_task, task)
    logger.debug("Related files: %s", related_files)

    if memory is None:
        memory = MemorySaver()

    if thread_id is None:
        thread_id = str(uuid.uuid4())

    tools = get_implementation_tools(
        expert_enabled=expert_enabled,
        web_research_enabled=get_config_repository().get("web_research_enabled", False),
    )

    agent = create_agent(model, tools, checkpointer=memory, agent_type="planner")

    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    working_directory = os.getcwd()

    # Make sure key_facts is defined before using it
    try:
        key_facts = format_key_facts_dict(get_key_fact_repository().get_facts_dict())
    except RuntimeError as e:
        logger.error(f"Failed to access key fact repository: {str(e)}")
        key_facts = ""

    # Get formatted research notes using repository
    try:
        repository = get_research_note_repository()
        notes_dict = repository.get_notes_dict()
        formatted_research_notes = format_research_notes_dict(notes_dict)
    except RuntimeError as e:
        logger.error(f"Failed to access research note repository: {str(e)}")
        formatted_research_notes = ""

    # Get latest project info
    try:
        project_info = get_project_info(".")
        formatted_project_info = format_project_info(project_info)
    except Exception as e:
        logger.warning("Failed to get project info: %s", str(e))
        formatted_project_info = "Project info unavailable"

    # Get environment inventory information
    env_inv = get_env_inv()

    # Get model configuration to check for reasoning_assist_default
    provider = get_config_repository().get("provider", "")
    model_name = get_config_repository().get("model", "")
    logger.debug("Checking for reasoning_assist_default on %s/%s", provider, model_name)

    model_config = {}
    provider_models = models_params.get(provider, {})
    if provider_models and model_name in provider_models:
        model_config = provider_models[model_name]

    # Check if reasoning assist is explicitly enabled/disabled
    force_assistance = get_config_repository().get("force_reasoning_assistance", False)
    disable_assistance = get_config_repository().get(
        "disable_reasoning_assistance", False
    )

    if force_assistance:
        reasoning_assist_enabled = True
    elif disable_assistance:
        reasoning_assist_enabled = False
    else:
        # Fall back to model default
        reasoning_assist_enabled = model_config.get("reasoning_assist_default", False)

    logger.debug("Reasoning assist enabled: %s", reasoning_assist_enabled)

    # Initialize implementation guidance section
    implementation_guidance_section = ""

    # If reasoning assist is enabled, make a one-off call to the expert model
    if reasoning_assist_enabled:
        try:
            logger.info(
                "Reasoning assist enabled for model %s, getting implementation guidance",
                model_name,
            )

            # Collect tool descriptions
            tool_metadata = []
            from ra_aid.tools.reflection import get_function_info as get_tool_info

            for tool in tools:
                try:
                    tool_info = get_tool_info(tool.func)
                    name = tool.func.__name__
                    description = inspect.getdoc(tool.func)
                    tool_metadata.append(
                        f"Tool: {name}\\nDescription: {description}\\n"
                    )
                except Exception as e:
                    logger.warning(f"Error getting tool info for {tool}: {e}")

            # Format tool metadata
            formatted_tool_metadata = "\\n".join(tool_metadata)

            # Initialize expert model
            expert_model = initialize_expert_llm(provider, model_name)

            # Format the reasoning assist prompt for implementation
            reasoning_assist_prompt = REASONING_ASSIST_PROMPT_IMPLEMENTATION.format(
                current_date=current_date,
                working_directory=working_directory,
                task=task,
                key_facts=key_facts,
                key_snippets=format_key_snippets_dict(
                    get_key_snippet_repository().get_snippets_dict()
                ),
                research_notes=formatted_research_notes,
                related_files="\\n".join(related_files),
                env_inv=env_inv,
                tool_metadata=formatted_tool_metadata,
            )

            # Show the reasoning assist query in a panel
            console.print(
                Panel(
                    Markdown(
                        "Consulting with the reasoning model on the best implementation approach."
                    ),
                    title="📝 Thinking about implementation...",
                    border_style="yellow",
                )
            )

            logger.debug("Invoking expert model for implementation reasoning assist")
            # Make the call to the expert model
            response = expert_model.invoke(reasoning_assist_prompt)

            # Check if the model supports think tags
            supports_think_tag = model_config.get("supports_think_tag", False)
            supports_thinking = model_config.get("supports_thinking", False)

            # Process response content
            content = None

            if hasattr(response, "content"):
                content = response.content
            else:
                # Fallback if content attribute is missing
                content = str(response)

            # Process the response content using the centralized function
            content, extracted_thinking = process_thinking_content(
                content=content,
                supports_think_tag=supports_think_tag,
                supports_thinking=supports_thinking,
                panel_title="💭 Implementation Thinking",
                panel_style="yellow",
                logger=logger,
            )

            # Display the implementation guidance in a panel
            console.print(
                Panel(
                    Markdown(content),
                    title="Implementation Guidance",
                    border_style="blue",
                )
            )

            # Format the implementation guidance section for the prompt
            implementation_guidance_section = f"""<implementation guidance>
{content}
</implementation guidance>"""

            logger.info("Received implementation guidance")
        except Exception as e:
            logger.error("Error getting implementation guidance: %s", e)
            implementation_guidance_section = ""

    prompt = IMPLEMENTATION_PROMPT.format(
        current_date=current_date,
        working_directory=working_directory,
        base_task=base_task,
        task=task,
        tasks=tasks,
        plan=plan,
        related_files=related_files,
        key_facts=key_facts,
        key_snippets=format_key_snippets_dict(
            get_key_snippet_repository().get_snippets_dict()
        ),
        research_notes=formatted_research_notes,
        work_log=get_work_log_repository().format_work_log(),
        expert_section=EXPERT_PROMPT_SECTION_IMPLEMENTATION if expert_enabled else "",
        human_section=(
            HUMAN_PROMPT_SECTION_IMPLEMENTATION
            if get_config_repository().get("hil", False)
            else ""
        ),
        web_research_section=(
            WEB_RESEARCH_PROMPT_SECTION_CHAT
            if get_config_repository().get("web_research_enabled", False)
            else ""
        ),
        env_inv=env_inv,
        project_info=formatted_project_info,
        implementation_guidance_section=implementation_guidance_section,
    )

    config_values = get_config_repository().get_all()
    recursion_limit = get_config_repository().get(
        "recursion_limit", DEFAULT_RECURSION_LIMIT
    )
    run_config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }
    run_config.update(config_values)

    try:
        logger.debug("Implementation agent completed successfully")
        none_or_fallback_handler = init_fallback_handler(agent, tools)
        _result = run_agent_with_retry(agent, prompt, none_or_fallback_handler)
        if _result:
            # Log task implementation completion
            log_work_event(f"Completed implementation of task: {task}")
        return _result
    except (KeyboardInterrupt, AgentInterrupt):
        raise
    except Exception as e:
        logger.error("Implementation agent failed: %s", str(e), exc_info=True)
        raise
=======
        )


from ra_aid.agents.research_agent import run_research_agent, run_web_research_agent
from ra_aid.agents.implementation_agent import run_task_implementation_agent
>>>>>>> @{-1}


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
            print_agent_output(chunk, agent_type, cb)

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
