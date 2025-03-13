"""
Research agent implementation.

This module provides functionality for running a research agent to investigate tasks
and queries. The agent can perform both general research and web-specific research
tasks, with options for expert guidance and human-in-the-loop collaboration.
"""

import inspect
import os
import uuid
from datetime import datetime
from typing import Any, Optional

from langchain_core.messages import SystemMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ra_aid.agent_context import agent_context, is_completed, reset_completion_flags, should_exit
# Import agent_utils functions at runtime to avoid circular imports
from ra_aid import agent_utils
from ra_aid.console.formatting import print_error
from ra_aid.database.repositories.key_fact_repository import get_key_fact_repository
from ra_aid.database.repositories.key_snippet_repository import get_key_snippet_repository
from ra_aid.database.repositories.human_input_repository import get_human_input_repository
from ra_aid.database.repositories.research_note_repository import get_research_note_repository
from ra_aid.database.repositories.config_repository import get_config_repository
from ra_aid.database.repositories.work_log_repository import get_work_log_repository
from ra_aid.env_inv_context import get_env_inv
from ra_aid.exceptions import AgentInterrupt
from ra_aid.llm import initialize_expert_llm
from ra_aid.logging_config import get_logger
from ra_aid.model_formatters import format_key_facts_dict
from ra_aid.model_formatters.key_snippets_formatter import format_key_snippets_dict
from ra_aid.model_formatters.research_notes_formatter import format_research_notes_dict
from ra_aid.text.processing import process_thinking_content
from ra_aid.models_params import models_params
from ra_aid.project_info import display_project_status, format_project_info, get_project_info
from ra_aid.prompts.expert_prompts import EXPERT_PROMPT_SECTION_RESEARCH
from ra_aid.prompts.human_prompts import HUMAN_PROMPT_SECTION_RESEARCH
from ra_aid.prompts.research_prompts import RESEARCH_ONLY_PROMPT, RESEARCH_PROMPT
from ra_aid.prompts.reasoning_assist_prompt import REASONING_ASSIST_PROMPT_RESEARCH
from ra_aid.prompts.web_research_prompts import (
    WEB_RESEARCH_PROMPT,
    WEB_RESEARCH_PROMPT_SECTION_RESEARCH,
)
from ra_aid.prompts.common_prompts import NEW_PROJECT_HINTS
from ra_aid.tool_configs import get_research_tools, get_web_research_tools
from ra_aid.tools.memory import get_related_files, log_work_event

logger = get_logger(__name__)
console = Console()


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
        from langgraph.checkpoint.memory import MemorySaver
        memory = MemorySaver()

    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    working_directory = os.getcwd()

    # Get the last human input, if it exists
    base_task = base_task_or_query
    try:
        human_input_repository = get_human_input_repository()
        most_recent_id = human_input_repository.get_most_recent_id()
        if most_recent_id is not None:
            recent_input = human_input_repository.get(most_recent_id)
            if recent_input and recent_input.content != base_task_or_query:
                last_human_input = recent_input.content
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
    provider = get_config_repository().get("expert_provider", "")
    model_name = get_config_repository().get("expert_model", "")

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
                    tool_metadata.append(f"Tool: {tool_info}\nDescription: {description}\n")
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
                project_info=formatted_project_info,
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

    agent = agent_utils.create_agent(model, tools, checkpointer=memory, agent_type="research")

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
</expert guidance>
YOU MUST FOLLOW THE EXPERT'S GUIDANCE OR ELSE BE TERMINATED!
"""

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
    recursion_limit = config.get("recursion_limit", 100)
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
            none_or_fallback_handler = agent_utils.init_fallback_handler(agent, tools)
            _result = agent_utils.run_agent_with_retry(agent, prompt, none_or_fallback_handler)
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
        from langgraph.checkpoint.memory import MemorySaver
        memory = MemorySaver()

    if thread_id is None:
        thread_id = str(uuid.uuid4())

    tools = get_web_research_tools(expert_enabled=expert_enabled)

    agent = agent_utils.create_agent(model, tools, checkpointer=memory, agent_type="research")

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

    recursion_limit = config.get("recursion_limit", 100)
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
        none_or_fallback_handler = agent_utils.init_fallback_handler(agent, tools)
        _result = agent_utils.run_agent_with_retry(agent, prompt, none_or_fallback_handler)
        if _result:
            # Log web research completion
            log_work_event(f"Completed web research phase for: {query}")
        return _result

    except (KeyboardInterrupt, AgentInterrupt):
        raise
    except Exception as e:
        logger.error("Web research agent failed: %s", str(e), exc_info=True)
        raise