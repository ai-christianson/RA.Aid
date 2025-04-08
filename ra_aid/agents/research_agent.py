
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

from rich.console import Console

# Import agent_utils functions at runtime to avoid circular imports
from ra_aid import agent_utils
from ra_aid.console.formatting import cpm
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
from ra_aid.project_info import (
    display_project_status,
    format_project_info,
    get_project_info,
)
from ra_aid.prompts.expert_prompts import EXPERT_PROMPT_SECTION_RESEARCH
from ra_aid.prompts.human_prompts import HUMAN_PROMPT_SECTION_RESEARCH
from ra_aid.prompts.research_prompts import RESEARCH_ONLY_PROMPT, RESEARCH_PROMPT
from ra_aid.prompts.reasoning_assist_prompt import REASONING_ASSIST_PROMPT_RESEARCH
from ra_aid.prompts.web_research_prompts import (
    WEB_RESEARCH_PROMPT,
    WEB_RESEARCH_PROMPT_SECTION_RESEARCH,
)
from ra_aid.prompts.custom_tools_prompts import DEFAULT_CUSTOM_TOOLS_PROMPT
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
    logger.info(
        f"[{thread_id}] Starting research agent. Task: '{base_task_or_query[:50]}...'"
    )
    logger.info(
        f"[{thread_id}] Config: expert={expert_enabled}, research_only={research_only}, "
        f"hil={hil}, web_research={web_research_enabled}"
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
                logger.debug(f"[{thread_id}] Appending last human input to base task.")
                base_task = f"<last human input>{last_human_input}</last human input>\n{base_task}"
    except RuntimeError as e:
        logger.error(f"[{thread_id}] Failed to access human input repository: {str(e)}")
        # Continue without appending last human input

    try:
        key_facts = format_key_facts_dict(get_key_fact_repository().get_facts_dict())
        logger.debug(f"[{thread_id}] Retrieved {len(key_facts)} chars of key facts.")
    except RuntimeError as e:
        logger.error(f"[{thread_id}] Failed to access key fact repository: {str(e)}")
        key_facts = ""

    try:
        key_snippets = format_key_snippets_dict(
            get_key_snippet_repository().get_snippets_dict()
        )
        logger.debug(
            f"[{thread_id}] Retrieved {len(key_snippets)} chars of key snippets."
        )
    except RuntimeError as e:
        logger.error(f"[{thread_id}] Failed to access key snippet repository: {str(e)}")
        key_snippets = ""

    try:
        related_files_list = get_related_files()
        related_files = "\n".join(related_files_list)
        logger.debug(
            f"[{thread_id}] Retrieved {len(related_files_list)} related files."
        )
    except Exception as e:
        logger.warning(f"[{thread_id}] Failed to get related files: {e}")
        related_files = ""

    try:
        project_info = get_project_info(".", file_limit=2000)
        formatted_project_info = format_project_info(project_info)
        logger.debug(
            f"[{thread_id}] Retrieved project info ({len(formatted_project_info)} chars)."
        )
    except Exception as e:
        logger.warning(f"[{thread_id}] Failed to get project info: {e}")
        formatted_project_info = ""

    tools = get_research_tools(
        research_only=research_only,
        expert_enabled=expert_enabled,
        human_interaction=hil,
        web_research_enabled=get_config_repository().get("web_research_enabled", False),
    )
    tool_names = [tool.func.__name__ for tool in tools]
    logger.debug(f"[{thread_id}] Tools selected for agent: {tool_names}")

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

    logger.debug(f"[{thread_id}] Reasoning assist enabled: {reasoning_assist_enabled}")
    expert_guidance = ""

    # Get research note information for reasoning assistance
    try:
        research_notes = format_research_notes_dict(
            get_research_note_repository().get_notes_dict()
        )
        logger.debug(
            f"[{thread_id}] Retrieved {len(research_notes)} chars of research notes."
        )
    except Exception as e:
        logger.warning(f"[{thread_id}] Failed to get research notes: {e}")
        research_notes = ""

    # If reasoning assist is enabled, make a one-off call to the expert model
    if reasoning_assist_enabled:
        try:
            logger.info(
                f"[{thread_id}] Reasoning assist enabled for model {model_name}, getting expert guidance"
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
                        f"Tool: {tool_info}\nDescription: {description}\n"
                    )
                except Exception as e:
                    logger.warning(f"[{thread_id}] Error getting tool info for {tool}: {e}")

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
            cpm(
                "Consulting with the reasoning model on the best research approach.",
                title="📝 Thinking about research strategy...",
                border_style="yellow",
            )

            logger.debug(f"[{thread_id}] Invoking expert model for reasoning assist")
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
                            logger.debug(f"[{thread_id}] Found structured thinking content")
                        # Extract response text
                        elif item.get("type") == "text" and "text" in item:
                            response_text = item["text"]
                            logger.debug(f"[{thread_id}] Found structured response text")

                # Display thinking content in a separate panel if available
                if thinking_content and get_config_repository().get(
                    "show_thoughts", False
                ):
                    logger.debug(
                        f"[{thread_id}] Displaying structured thinking content ({len(thinking_content)} chars)"
                    )
                    cpm(
                        thinking_content,
                        title="💭 Expert Thinking",
                        border_style="yellow",
                    )

                # Use response_text if available, otherwise fall back to joining
                if response_text:
                    content = response_text
                else:
                    # Fallback: join list items if structured extraction failed
                    logger.debug(
                        f"[{thread_id}] No structured response text found, joining list items"
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
            cpm(content, title="Research Strategy Guidance", border_style="blue")

            # Use the content as expert guidance
            expert_guidance = (
                content + "\n\nCONSULT WITH THE EXPERT FREQUENTLY DURING RESEARCH"
            )

            logger.info(f"[{thread_id}] Received expert guidance for research")
        except Exception as e:
            logger.error(f"[{thread_id}] Error getting expert guidance for research: {e}")
            expert_guidance = ""

    logger.debug(f"[{thread_id}] Creating research agent with model: {model}")
    agent = agent_utils.create_agent(
        model, tools, checkpointer=memory, agent_type="research"
    )

    if agent:
        logger.info(f"[{thread_id}] Research agent created successfully.")
    else:
        logger.info(f"[{thread_id}] Research agent creation returned None (likely web research only).")


    expert_section = EXPERT_PROMPT_SECTION_RESEARCH if expert_enabled else ""
    human_section = HUMAN_PROMPT_SECTION_RESEARCH if hil else ""
    web_research_section = (
        WEB_RESEARCH_PROMPT_SECTION_RESEARCH
        if get_config_repository().get("web_research_enabled")
        else ""
    )
    custom_tools_section = (
        DEFAULT_CUSTOM_TOOLS_PROMPT
        if get_config_repository().get("custom_tools_enabled", False)
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
        custom_tools_section=custom_tools_section,
        key_facts=key_facts,
        work_log=get_work_log_repository().format_work_log(),
        key_snippets=key_snippets,
        related_files=related_files,
        project_info=formatted_project_info,
        new_project_hints=NEW_PROJECT_HINTS if project_info.is_new else "",
        env_inv=get_env_inv(),
        expert_guidance_section=expert_guidance_section,
    )

    # Log the prompt, trimming if it's too long
    prompt_log_limit = 1000
    trimmed_prompt = (
        prompt[:prompt_log_limit] + "... (trimmed)"
        if len(prompt) > prompt_log_limit
        else prompt
    )
    logger.debug(f"[{thread_id}] Prompt for agent:\n{trimmed_prompt}")

    recursion_limit = get_config_repository().get("recursion_limit", 100)
    run_config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }
    # Update with necessary config values
    run_config["show_cost"] = get_config_repository().get("show_cost", False)
    run_config["valid_providers"] = get_config_repository().get("valid_providers", [])

    try:
        if console_message:
            cpm(console_message, title="🔬 Looking into it...")

        if project_info:
            display_project_status(project_info)

        if agent is not None:
            logger.debug(f"[{thread_id}] Invoking research agent...")
            none_or_fallback_handler = agent_utils.init_fallback_handler(agent, tools)
            _result = agent_utils.run_agent_with_retry(
                agent, prompt, none_or_fallback_handler
            )
            if _result:
                # Log research completion
                logger.info(f"[{thread_id}] Research agent completed successfully.")
                log_work_event(f"Completed research phase for: {base_task_or_query}")
            else:
                 logger.info(f"[{thread_id}] Research agent finished without returning a final message.")
            return _result
        else:
            logger.debug(f"[{thread_id}] No agent created, running web research tools directly")
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
        logger.info(f"[{thread_id}] Research agent interrupted.")
        raise
    except Exception as e:
        logger.error(f"[{thread_id}] Research agent failed: {str(e)}", exc_info=True)
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
    logger.debug(f"[{thread_id}] Starting web research agent.")
    logger.debug(
        f"[{thread_id}] Web research configuration: expert={expert_enabled}, hil={hil}, web={web_research_enabled}"
    )

    if memory is None:
        from langgraph.checkpoint.memory import MemorySaver

        memory = MemorySaver()

    if thread_id is None:
        thread_id = str(uuid.uuid4())

    tools = get_web_research_tools(expert_enabled=expert_enabled)

    agent = agent_utils.create_agent(
        model, tools, checkpointer=memory, agent_type="research"
    )

    expert_section = EXPERT_PROMPT_SECTION_RESEARCH if expert_enabled else ""
    human_section = HUMAN_PROMPT_SECTION_RESEARCH if hil else ""

    try:
        key_facts = format_key_facts_dict(get_key_fact_repository().get_facts_dict())
    except RuntimeError as e:
        logger.error(f"[{thread_id}] Failed to access key fact repository: {str(e)}")
        key_facts = ""
    try:
        key_snippets = format_key_snippets_dict(
            get_key_snippet_repository().get_snippets_dict()
        )
    except RuntimeError as e:
        logger.error(f"[{thread_id}] Failed to access key snippet repository: {str(e)}")
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
        related_files="\n".join(related_files),
        env_inv=get_env_inv(),
    )

    recursion_limit = get_config_repository().get("recursion_limit", 100)
    run_config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }
    # Update with necessary config values
    run_config["show_cost"] = get_config_repository().get("show_cost", False)
    run_config["valid_providers"] = get_config_repository().get("valid_providers", [])

    try:
        if console_message:
            cpm(console_message, title="🔬 Researching...")

        logger.debug(f"[{thread_id}] Invoking web research agent.")
        none_or_fallback_handler = agent_utils.init_fallback_handler(agent, tools)
        _result = agent_utils.run_agent_with_retry(
            agent, prompt, none_or_fallback_handler
        )
        if _result:
            # Log web research completion
            logger.info(f"[{thread_id}] Web research agent completed successfully.")
            log_work_event(f"Completed web research phase for: {query}")
        else:
            logger.info(f"[{thread_id}] Web research agent finished without returning a final message.")
        return _result

    except (KeyboardInterrupt, AgentInterrupt):
        logger.info(f"[{thread_id}] Web research agent interrupted.")
        raise
    except Exception as e:
        logger.error(f"[{thread_id}] Web research agent failed: {str(e)}", exc_info=True)
        raise
