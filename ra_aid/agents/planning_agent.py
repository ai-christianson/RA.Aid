"""
Planning agent implementation.

This module provides functionality for running a planning agent to create implementation 
plans. The agent can be configured with expert guidance and human-in-the-loop options.
"""

import inspect
import os
import uuid
from datetime import datetime
from typing import Any, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ra_aid.agent_context import agent_context, is_completed, reset_completion_flags, should_exit
# Import agent_utils functions at runtime to avoid circular imports
from ra_aid import agent_utils
from ra_aid.console.formatting import print_stage_header
from ra_aid.database.repositories.key_fact_repository import get_key_fact_repository
from ra_aid.database.repositories.key_snippet_repository import get_key_snippet_repository
from ra_aid.database.repositories.research_note_repository import get_research_note_repository
from ra_aid.database.repositories.config_repository import get_config_repository
from ra_aid.database.repositories.work_log_repository import get_work_log_repository
from ra_aid.database.repositories.trajectory_repository import get_trajectory_repository
from ra_aid.database.repositories.human_input_repository import get_human_input_repository
from ra_aid.env_inv_context import get_env_inv
from ra_aid.exceptions import AgentInterrupt
from ra_aid.llm import initialize_expert_llm
from ra_aid.logging_config import get_logger
from ra_aid.model_formatters import format_key_facts_dict
from ra_aid.model_formatters.key_snippets_formatter import format_key_snippets_dict
from ra_aid.model_formatters.research_notes_formatter import format_research_notes_dict
from ra_aid.text.processing import process_thinking_content
from ra_aid.models_params import models_params
from ra_aid.project_info import format_project_info, get_project_info
from ra_aid.prompts.expert_prompts import EXPERT_PROMPT_SECTION_PLANNING
from ra_aid.prompts.human_prompts import HUMAN_PROMPT_SECTION_PLANNING
from ra_aid.prompts.planning_prompts import PLANNING_PROMPT
from ra_aid.prompts.reasoning_assist_prompt import REASONING_ASSIST_PROMPT_PLANNING
from ra_aid.prompts.web_research_prompts import WEB_RESEARCH_PROMPT_SECTION_PLANNING
from ra_aid.prompts.custom_tools_prompts import DEFAULT_CUSTOM_TOOLS_PROMPT
from ra_aid.tool_configs import get_planning_tools
from ra_aid.tools.memory import get_related_files, log_work_event

logger = get_logger(__name__)
console = Console()


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
        from langgraph.checkpoint.memory import MemorySaver
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
    provider = get_config_repository().get("expert_provider", "")
    model_name = get_config_repository().get("expert_model", "")
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
    
    # Record stage transition in trajectory
    trajectory_repo = get_trajectory_repository()
    human_input_id = get_human_input_repository().get_most_recent_id()
    trajectory_repo.create(
        step_data={
            "stage": "planning_stage",
            "display_title": "Planning Stage",
        },
        record_type="stage_transition",
        human_input_id=human_input_id
    )

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
                project_info=formatted_project_info,
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

    agent = agent_utils.create_agent(model, tools, checkpointer=memory, agent_type="planner")

    expert_section = EXPERT_PROMPT_SECTION_PLANNING if expert_enabled else ""
    human_section = HUMAN_PROMPT_SECTION_PLANNING if hil else ""
    web_research_section = (
        WEB_RESEARCH_PROMPT_SECTION_PLANNING
        if get_config_repository().get("web_research_enabled", False)
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
</expert guidance>"""

    planning_prompt = PLANNING_PROMPT.format(
        current_date=current_date,
        working_directory=working_directory,
        expert_section=expert_section,
        human_section=human_section,
        web_research_section=web_research_section,
        custom_tools_section=custom_tools_section,
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

    recursion_limit = get_config_repository().get(
        "recursion_limit", 100
    )
    run_config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }
    # Update with necessary config values
    run_config["show_cost"] = get_config_repository().get("show_cost", False)
    run_config["valid_providers"] = get_config_repository().get("valid_providers", [])

    try:
        logger.debug("Planning agent completed successfully")
        none_or_fallback_handler = agent_utils.init_fallback_handler(agent, tools)
        _result = agent_utils.run_agent_with_retry(agent, planning_prompt, none_or_fallback_handler)
        if _result:
            # Log planning completion
            log_work_event(f"Completed planning phase for: {base_task}")
        return _result
    except (KeyboardInterrupt, AgentInterrupt):
        raise
    except Exception as e:
        logger.error("Planning agent failed: %s", str(e), exc_info=True)
        raise