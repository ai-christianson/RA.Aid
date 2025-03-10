"""
Implementation agent for executing specific implementation tasks.

This module provides functionality for running a task implementation agent 
to execute specific tasks based on a plan. The agent can be configured with 
expert guidance and web research options.
"""

import inspect
import os
import uuid
from datetime import datetime
from typing import Any, Optional, List

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ra_aid.agent_context import agent_context, is_completed, reset_completion_flags, should_exit
# Import agent_utils functions at runtime to avoid circular imports
from ra_aid import agent_utils
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
from ra_aid.models_params import models_params, DEFAULT_TOKEN_LIMIT
from ra_aid.project_info import format_project_info, get_project_info
from ra_aid.prompts.expert_prompts import EXPERT_PROMPT_SECTION_IMPLEMENTATION
from ra_aid.prompts.human_prompts import HUMAN_PROMPT_SECTION_IMPLEMENTATION
from ra_aid.prompts.implementation_prompts import IMPLEMENTATION_PROMPT
from ra_aid.prompts.reasoning_assist_prompt import REASONING_ASSIST_PROMPT_IMPLEMENTATION
from ra_aid.prompts.web_research_prompts import WEB_RESEARCH_PROMPT_SECTION_CHAT
from ra_aid.tool_configs import get_implementation_tools
from ra_aid.tools.memory import get_related_files, log_work_event
from ra_aid.text.processing import process_thinking_content

logger = get_logger(__name__)
console = Console()


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
        task: The current task to implement
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
        from langgraph.checkpoint.memory import MemorySaver
        memory = MemorySaver()

    if thread_id is None:
        thread_id = str(uuid.uuid4())

    tools = get_implementation_tools(
        expert_enabled=expert_enabled,
        web_research_enabled=get_config_repository().get("web_research_enabled", False),
    )

    agent = agent_utils.create_agent(model, tools, checkpointer=memory, agent_type="planner")

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
    provider = get_config_repository().get("expert_provider", "")
    model_name = get_config_repository().get("expert_model", "")
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
                        f"Tool: {name}\nDescription: {description}\n"
                    )
                except Exception as e:
                    logger.warning(f"Error getting tool info for {tool}: {e}")

            # Format tool metadata
            formatted_tool_metadata = "\n".join(tool_metadata)

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
                related_files="\n".join(related_files),
                env_inv=env_inv,
                tool_metadata=formatted_tool_metadata,
                project_info=formatted_project_info,
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
        "recursion_limit", 100
    )
    run_config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }
    run_config.update(config_values)

    try:
        logger.debug("Implementation agent completed successfully")
        none_or_fallback_handler = agent_utils.init_fallback_handler(agent, tools)
        _result = agent_utils.run_agent_with_retry(agent, prompt, none_or_fallback_handler)
        if _result:
            # Log task implementation completion
            log_work_event(f"Completed implementation of task: {task}")
        return _result
    except (KeyboardInterrupt, AgentInterrupt):
        raise
    except Exception as e:
        logger.error("Implementation agent failed: %s", str(e), exc_info=True)
        raise