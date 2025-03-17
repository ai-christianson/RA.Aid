"""Tools for spawning and managing sub-agents."""

from typing import Any, Dict, List, Union
import logging

from langchain_core.tools import tool
from langchain_text_splitters import markdown
from rich.console import Console

from ra_aid.agent_context import (
    get_completion_message,
    get_crash_message,
    get_depth,
    is_crashed,
    reset_completion_flags,
)
from ra_aid.config import DEFAULT_MODEL
from ra_aid.console.formatting import print_error, print_task_header
from ra_aid.database.repositories.human_input_repository import get_human_input_repository
from ra_aid.database.repositories.key_fact_repository import get_key_fact_repository
from ra_aid.database.repositories.key_snippet_repository import get_key_snippet_repository
from ra_aid.database.repositories.config_repository import get_config_repository
from ra_aid.database.repositories.trajectory_repository import get_trajectory_repository
from ra_aid.database.repositories.related_files_repository import get_related_files_repository
from ra_aid.database.repositories.research_note_repository import get_research_note_repository
from ra_aid.exceptions import AgentInterrupt
from ra_aid.model_formatters import format_key_facts_dict
from ra_aid.model_formatters.key_snippets_formatter import format_key_snippets_dict
from ra_aid.model_formatters.research_notes_formatter import format_research_notes_dict

from ra_aid.llm import initialize_llm
from .human import ask_human
from .memory import get_related_files, get_work_log

ResearchResult = Dict[str, Union[str, bool, Dict[int, Any], List[Any], None]]

CANCELLED_BY_USER_REASON = "The operation was explicitly cancelled by the user. This typically is an indication that the action requested was not aligned with the user request."

RESEARCH_AGENT_RECURSION_LIMIT = 3

console = Console()
logger = logging.getLogger(__name__)


@tool("request_research")
def request_research(query: str) -> ResearchResult:
    """Spawn a research-only agent to investigate the given query.

    This function creates a new research agent to investigate the given query. It includes
    recursion depth limiting to prevent infinite recursive research calls.

    Args:
        query: The research question or project description
    """
    # Initialize model from config
    model = initialize_llm(
        get_config_repository().get("provider", "anthropic"),
        get_config_repository().get("model", "claude-3-7-sonnet-20250219"),
        temperature=get_config_repository().get("temperature"),
    )

    # Check recursion depth
    current_depth = get_depth()
    if current_depth >= RESEARCH_AGENT_RECURSION_LIMIT:
        error_message = "Maximum research recursion depth reached"
        
        # Record error in trajectory
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
            error_message=error_message
        )
        
        print_error(error_message)
        try:
            key_facts = format_key_facts_dict(get_key_fact_repository().get_facts_dict())
        except RuntimeError as e:
            logger.error(f"Failed to access key fact repository: {str(e)}")
            key_facts = ""

        try:
            key_snippets = format_key_snippets_dict(get_key_snippet_repository().get_snippets_dict())
        except RuntimeError as e:
            logger.error(f"Failed to access key snippet repository: {str(e)}")
            key_snippets = ""
            
        return {
            "completion_message": "Research stopped - maximum recursion depth reached",
            "key_facts": key_facts,
            "related_files": get_related_files(),
            "research_notes": "",  # Empty for max depth exceeded case
            "key_snippets": key_snippets,
            "success": False,
            "reason": "max_depth_exceeded",
        }

    success = True
    reason = None

    try:
        # Run research agent
        from ..agents.research_agent import run_research_agent

        _result = run_research_agent(
            query,
            model,
            expert_enabled=True,
            research_only=True,
            hil=get_config_repository().get("hil", False),
            console_message=query,
        )
    except AgentInterrupt:
        print()
        response = ask_human.invoke({"question": "Why did you interrupt me?"})
        success = False
        reason = response if response.strip() else CANCELLED_BY_USER_REASON
    except KeyboardInterrupt:
        raise
    except Exception as e:
        error_message = f"Error during research: {str(e)}"
        
        # Record error in trajectory
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
            error_message=error_message
        )
        
        print_error(error_message)
        success = False
        reason = f"error: {str(e)}"
    finally:
        # Get completion message if available
        completion_message = get_completion_message() or (
            "Task was completed successfully." if success else None
        )

        work_log = get_work_log()

        # Clear completion state
        reset_completion_flags()

    try:
        key_facts = format_key_facts_dict(get_key_fact_repository().get_facts_dict())
    except RuntimeError as e:
        logger.error(f"Failed to access key fact repository: {str(e)}")
        key_facts = ""
        
    try:
        key_snippets = format_key_snippets_dict(get_key_snippet_repository().get_snippets_dict())
    except RuntimeError as e:
        logger.error(f"Failed to access key snippet repository: {str(e)}")
        key_snippets = ""

    try:
        repository = get_research_note_repository()
        notes_dict = repository.get_notes_dict()
        formatted_research_notes = format_research_notes_dict(notes_dict)
    except RuntimeError as e:
        logger.error(f"Failed to access research note repository: {str(e)}")
        formatted_research_notes = ""
        
    response_data = {
        "completion_message": completion_message,
        "key_facts": key_facts,
        "related_files": get_related_files(),
        "research_notes": formatted_research_notes,
        "key_snippets": key_snippets,
        "success": success,
        "reason": reason,
    }
    if work_log is not None:
        response_data["work_log"] = work_log
    return response_data


@tool("request_web_research")
def request_web_research(query: str) -> ResearchResult:
    """Spawn a web research agent to investigate the given query using web search.

    Args:
        query: The research question or project description
    """
    # Initialize model from config
    model = initialize_llm(
        get_config_repository().get("provider", "anthropic"),
        get_config_repository().get("model", "claude-3-7-sonnet-20250219"),
        temperature=get_config_repository().get("temperature"),
    )

    success = True
    reason = None

    try:
        # Run web research agent
        from ..agents.research_agent import run_web_research_agent

        _result = run_web_research_agent(
            query,
            model,
            expert_enabled=True,
            hil=get_config_repository().get("hil", False),
            console_message=query,
        )
    except AgentInterrupt:
        print()
        response = ask_human.invoke({"question": "Why did you interrupt me?"})
        success = False
        reason = response if response.strip() else CANCELLED_BY_USER_REASON
    except KeyboardInterrupt:
        raise
    except Exception as e:
        error_message = f"Error during web research: {str(e)}"
        
        # Record error in trajectory
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
            error_message=error_message
        )
        
        print_error(error_message)
        success = False
        reason = f"error: {str(e)}"
    finally:
        # Get completion message if available
        completion_message = get_completion_message() or (
            "Task was completed successfully." if success else None
        )

        work_log = get_work_log()

        # Clear completion state
        reset_completion_flags()

    try:
        key_snippets = format_key_snippets_dict(get_key_snippet_repository().get_snippets_dict())
    except RuntimeError as e:
        logger.error(f"Failed to access key snippet repository: {str(e)}")
        key_snippets = ""

    try:
        repository = get_research_note_repository()
        notes_dict = repository.get_notes_dict()
        formatted_research_notes = format_research_notes_dict(notes_dict)
    except RuntimeError as e:
        logger.error(f"Failed to access research note repository: {str(e)}")
        formatted_research_notes = ""
        
    response_data = {
        "completion_message": completion_message,
        "key_snippets": key_snippets,
        "research_notes": formatted_research_notes,
        "success": success,
        "reason": reason,
    }
    if work_log is not None:
        response_data["work_log"] = work_log
    return response_data


@tool("request_research_and_implementation")
def request_research_and_implementation(query: str) -> Dict[str, Any]:
    """Spawn a research agent to investigate and implement the given query.

    If you are calling this on behalf of a user request, you must *faithfully*
    represent all info the user gave you, sometimes even to the point of repeating the user query verbatim.

    Args:
        query: The research question or project description
    """
    # Initialize model from config
    model = initialize_llm(
        get_config_repository().get("provider", "anthropic"),
        get_config_repository().get("model", "claude-3-7-sonnet-20250219"),
        temperature=get_config_repository().get("temperature"),
    )

    try:
        # Run research agent
        from ..agents.research_agent import run_research_agent

        _result = run_research_agent(
            query,
            model,
            expert_enabled=True,
            research_only=False,
            hil=get_config_repository().get("hil", False),
            console_message=query,
        )

        success = True
        reason = None
    except AgentInterrupt:
        print()
        response = ask_human.invoke({"question": "Why did you interrupt me?"})
        success = False
        reason = response if response.strip() else CANCELLED_BY_USER_REASON
    except KeyboardInterrupt:
        raise
    except Exception as e:
        console.print(f"\n[red]Error during research: {str(e)}[/red]")
        success = False
        reason = f"error: {str(e)}"

    # Get completion message if available
    completion_message = get_completion_message() or (
        "Task was completed successfully." if success else None
    )

    work_log = get_work_log()

    # Clear completion state
    reset_completion_flags()

    try:
        key_facts = format_key_facts_dict(get_key_fact_repository().get_facts_dict())
    except RuntimeError as e:
        logger.error(f"Failed to access key fact repository: {str(e)}")
        key_facts = ""
        
    try:
        key_snippets = format_key_snippets_dict(get_key_snippet_repository().get_snippets_dict())
    except RuntimeError as e:
        logger.error(f"Failed to access key snippet repository: {str(e)}")
        key_snippets = ""

    try:
        repository = get_research_note_repository()
        notes_dict = repository.get_notes_dict()
        formatted_research_notes = format_research_notes_dict(notes_dict)
    except RuntimeError as e:
        logger.error(f"Failed to access research note repository: {str(e)}")
        formatted_research_notes = ""
        
    response_data = {
        "completion_message": completion_message,
        "key_facts": key_facts,
        "related_files": get_related_files(),
        "research_notes": formatted_research_notes,
        "key_snippets": key_snippets,
        "success": success,
        "reason": reason,
    }
    if work_log is not None:
        response_data["work_log"] = work_log
    return response_data


@tool("request_task_implementation")
def request_task_implementation(task_spec: str) -> str:
    """Spawn an implementation agent to execute the given task.

    Task specs should have the requirements. Generally, the spec will not include any code.

    Args:
        task_spec: REQUIRED The full task specification (markdown format, typically one part of the overall plan)
    """
    # Initialize model from config
    model = initialize_llm(
        get_config_repository().get("provider", "anthropic"),
        get_config_repository().get("model",DEFAULT_MODEL),
        temperature=get_config_repository().get("temperature"),
    )

    # Get required parameters
    related_files = list(get_related_files_repository().get_all().values())

    try:
        print_task_header(task_spec)
        
        # Record task display in trajectory
        trajectory_repo = get_trajectory_repository()
        human_input_id = get_human_input_repository().get_most_recent_id()
        trajectory_repo.create(
            step_data={
                "task": task_spec,
                "display_title": "Task",
            },
            record_type="task_display",
            human_input_id=human_input_id
        )
        
        # Run implementation agent
        from ..agents.implementation_agent import run_task_implementation_agent

        reset_completion_flags()

        _result = run_task_implementation_agent(
            base_task="",  # No more base_task from global memory
            tasks=[],  # No more tasks from global memory
            task=task_spec,
            plan="",  # No more plan from global memory
            related_files=related_files,
            model=model,
            expert_enabled=True,
        )

        success = True
        reason = None
    except AgentInterrupt:
        print()
        response = ask_human.invoke({"question": "Why did you interrupt me?"})
        success = False
        reason = response if response.strip() else CANCELLED_BY_USER_REASON
    except KeyboardInterrupt:
        raise
    except Exception as e:
        error_message = f"Error during task implementation: {str(e)}"
        
        # Record error in trajectory
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
            error_message=error_message
        )
        
        print_error(error_message)
        success = False
        reason = f"error: {str(e)}"

    # Get completion message if available
    completion_message = get_completion_message() or (
        "Task was completed successfully." if success else None
    )

    # Get and reset work log if at root depth
    work_log = get_work_log()

    # Clear completion state
    reset_completion_flags()

    # Check if the agent has crashed
    agent_crashed = is_crashed()
    crash_message = get_crash_message() if agent_crashed else None

    try:
        key_facts = format_key_facts_dict(get_key_fact_repository().get_facts_dict())
    except RuntimeError as e:
        logger.error(f"Failed to access key fact repository: {str(e)}")
        key_facts = ""
        
    try:
        key_snippets = format_key_snippets_dict(get_key_snippet_repository().get_snippets_dict())
    except RuntimeError as e:
        logger.error(f"Failed to access key snippet repository: {str(e)}")
        key_snippets = ""
        
    response_data = {
        "key_facts": key_facts,
        "related_files": get_related_files(),
        "key_snippets": key_snippets,
        "completion_message": completion_message,
        "success": success and not agent_crashed,
        "reason": reason,
        "agent_crashed": agent_crashed,
        "crash_message": crash_message,
    }
    if work_log is not None:
        response_data["work_log"] = work_log

    # Convert the response data to a markdown string
    markdown_parts = []

    # Add header and completion message
    markdown_parts.append("# Task Implementation")
    if response_data.get("completion_message"):
        markdown_parts.append(
            f"\n## Completion Message\n\n{response_data['completion_message']}"
        )

    # Add crash information if applicable
    if response_data.get("agent_crashed"):
        markdown_parts.append(
            f"\n## ⚠️ Agent Crashed ⚠️\n\n**Error:** {response_data.get('crash_message', 'Unknown error')}"
        )

    # Add success status
    status = "Success" if response_data.get("success", False) else "Failed"
    reason_text = (
        f": {response_data.get('reason')}" if response_data.get("reason") else ""
    )
    markdown_parts.append(f"\n## Status\n\n**{status}**{reason_text}")

    # Add key facts
    if response_data.get("key_facts"):
        markdown_parts.append(f"\n## Key Facts\n\n{response_data['key_facts']}")

    # Add related files
    if response_data.get("related_files"):
        files_list = "\n".join([f"- {file}" for file in response_data["related_files"]])
        markdown_parts.append(f"\n## Related Files\n\n{files_list}")

    # Add key snippets
    if response_data.get("key_snippets"):
        markdown_parts.append(f"\n## Key Snippets\n\n{response_data['key_snippets']}")

    # Add work log
    if response_data.get("work_log"):
        markdown_parts.append(f"\n## Work Log\n\n{response_data['work_log']}")
        markdown_parts.append(
            "\n\nTHE ABOVE WORK HAS BEEN COMPLETED"
        )

    # Join all parts into a single markdown string
    markdown_output = "".join(markdown_parts)

    return markdown_output


@tool("request_implementation")
def request_implementation(task_spec: str) -> str:
    """Spawn a planning agent to create an implementation plan for the given task.

    Args:
        task_spec: The task specification to plan implementation for
    """
    # Initialize model from config
    model = initialize_llm(
        get_config_repository().get("provider", "anthropic"),
        get_config_repository().get("model", DEFAULT_MODEL),
        temperature=get_config_repository().get("temperature"),
    )

    try:
        # Run planning agent
        from ..agents import run_planning_agent

        reset_completion_flags()

        _result = run_planning_agent(
            task_spec,
            model,
            expert_enabled=True,
            hil=get_config_repository().get("hil", False),
        )

        success = True
        reason = None
    except AgentInterrupt:
        print()
        response = ask_human.invoke({"question": "Why did you interrupt me?"})
        success = False
        reason = response if response.strip() else CANCELLED_BY_USER_REASON
    except KeyboardInterrupt:
        raise
    except Exception as e:
        error_message = f"Error during planning: {str(e)}"
        
        # Record error in trajectory
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
            error_message=error_message
        )
        
        print_error(error_message)
        success = False
        reason = f"error: {str(e)}"

    # Get completion message if available
    completion_message = get_completion_message() or (
        "Task was completed successfully." if success else None
    )

    # Get and reset work log if at root depth
    work_log = get_work_log()

    # Clear completion state
    reset_completion_flags()

    # Check if the agent has crashed
    agent_crashed = is_crashed()
    crash_message = get_crash_message() if agent_crashed else None

    try:
        key_facts = format_key_facts_dict(get_key_fact_repository().get_facts_dict())
    except RuntimeError as e:
        logger.error(f"Failed to access key fact repository: {str(e)}")
        key_facts = ""
        
    try:
        key_snippets = format_key_snippets_dict(get_key_snippet_repository().get_snippets_dict())
    except RuntimeError as e:
        logger.error(f"Failed to access key snippet repository: {str(e)}")
        key_snippets = ""
        
    response_data = {
        "completion_message": completion_message,
        "key_facts": key_facts,
        "related_files": get_related_files(),
        "key_snippets": key_snippets,
        "success": success and not agent_crashed,
        "reason": reason,
        "agent_crashed": agent_crashed,
        "crash_message": crash_message,
    }
    if work_log is not None:
        response_data["work_log"] = work_log

    # Convert the response data to a markdown string
    markdown_parts = []

    # Add header and completion message
    markdown_parts.append("# Implementation Plan")
    if response_data.get("completion_message"):
        markdown_parts.append(
            f"\n## Completion Message\n\n{response_data['completion_message']}"
        )

    # Add crash information if applicable
    if response_data.get("agent_crashed"):
        markdown_parts.append(
            f"\n## ⚠️ Agent Crashed ⚠️\n\n**Error:** {response_data.get('crash_message', 'Unknown error')}"
        )

    # Add success status
    status = "Success" if response_data.get("success", False) else "Failed"
    reason_text = (
        f": {response_data.get('reason')}" if response_data.get("reason") else ""
    )
    markdown_parts.append(f"\n## Status\n\n**{status}**{reason_text}")

    # Add key facts
    if response_data.get("key_facts"):
        markdown_parts.append(f"\n## Key Facts\n\n{response_data['key_facts']}")

    # Add related files
    if response_data.get("related_files"):
        files_list = "\n".join([f"- {file}" for file in response_data["related_files"]])
        markdown_parts.append(f"\n## Related Files\n\n{files_list}")

    # Add key snippets
    if response_data.get("key_snippets"):
        markdown_parts.append(f"\n## Key Snippets\n\n{response_data['key_snippets']}")

    # Add work log
    if response_data.get("work_log"):
        markdown_parts.append(f"\n## Work Log\n\n{response_data['work_log']}")
        markdown_parts.append(
            "\n\nTHE ABOVE WORK HAS ALREADY BEEN COMPLETED --**DO NOT REQUEST IMPLEMENTATION OF IT AGAIN**"
        )

    # Join all parts into a single markdown string
    markdown_output = "".join(markdown_parts)

    return markdown_output
