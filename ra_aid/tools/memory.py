import os
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from ra_aid.utils.file_utils import is_binary_file
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from typing_extensions import TypedDict

from ra_aid.agent_context import (
    mark_plan_completed,
    mark_should_exit,
    mark_task_completed,
)
from ra_aid.database.repositories.key_fact_repository import get_key_fact_repository
from ra_aid.database.repositories.key_snippet_repository import get_key_snippet_repository
from ra_aid.database.repositories.human_input_repository import get_human_input_repository
from ra_aid.database.repositories.research_note_repository import get_research_note_repository
from ra_aid.database.repositories.trajectory_repository import get_trajectory_repository
from ra_aid.database.repositories.work_log_repository import get_work_log_repository
from ra_aid.model_formatters import key_snippets_formatter
from ra_aid.logging_config import get_logger

logger = get_logger(__name__)



class SnippetInfo(TypedDict):
    filepath: str
    line_number: int
    snippet: str
    description: Optional[str]


console = Console()

# Import repositories using the get_* functions
from ra_aid.database.repositories.key_fact_repository import get_key_fact_repository

# Import the related files repository
from ra_aid.database.repositories.related_files_repository import get_related_files_repository


@tool("emit_research_notes")
def emit_research_notes(notes: str) -> str:
    """Use this when you have completed your research to share your notes in markdown format.

    Keep your research notes information dense and no more than 300 words.

    Args:
        notes: REQUIRED The research notes to store
    """
    # Try to get the latest human input
    human_input_id = None
    try:
        human_input_repo = get_human_input_repository()
        human_input_id = human_input_repo.get_most_recent_id()
    except RuntimeError as e:
        logger.warning(f"No HumanInputRepository available: {str(e)}")
    except Exception as e:
        logger.warning(f"Failed to get recent human input: {str(e)}")
    
    try:
        # Create note in database using repository
        created_note = get_research_note_repository().create(notes, human_input_id=human_input_id)
        note_id = created_note.id
        
        # Format the note using the formatter
        from ra_aid.model_formatters.research_notes_formatter import format_research_note
        formatted_note = format_research_note(note_id, notes)
        
        # Record to trajectory before displaying panel
        try:
            trajectory_repo = get_trajectory_repository()
            trajectory_repo.create(
                tool_name="emit_research_notes",
                tool_parameters={"notes": notes},
                step_data={
                    "note_id": note_id,
                    "display_title": "Research Notes",
                },
                record_type="memory_operation",
                human_input_id=human_input_id
            )
        except RuntimeError as e:
            logger.warning(f"Failed to record trajectory: {str(e)}")
        
        # Display formatted note
        console.print(Panel(Markdown(formatted_note), title="🔍 Research Notes"))
        
        log_work_event(f"Stored research note #{note_id}.")
        
        # Check if we need to clean up notes (more than 30)
        try:
            all_notes = get_research_note_repository().get_all()
            if len(all_notes) > 30:
                # Trigger the research notes cleaner agent
                try:
                    from ra_aid.agents.research_notes_gc_agent import run_research_notes_gc_agent
                    run_research_notes_gc_agent()
                except Exception as e:
                    logger.error(f"Failed to run research notes cleaner: {str(e)}")
        except RuntimeError as e:
            logger.error(f"Failed to access research note repository: {str(e)}")
            
        return f"Research note #{note_id} stored."
    except RuntimeError as e:
        logger.error(f"Failed to access research note repository: {str(e)}")
        console.print(f"Error storing research note: {str(e)}", style="red")
        return "Failed to store research note."


@tool("emit_key_facts")
def emit_key_facts(facts: List[str]) -> str:
    """Store multiple key facts about the project or current task in global memory.

    Args:
        facts: List of key facts to store
    """
    results = []
    
    # Try to get the latest human input
    human_input_id = None
    try:
        human_input_repo = get_human_input_repository()
        human_input_id = human_input_repo.get_most_recent_id()
    except RuntimeError as e:
        logger.warning(f"No HumanInputRepository available: {str(e)}")
    except Exception as e:
        logger.warning(f"Failed to get recent human input: {str(e)}")
    
    for fact in facts:
        try:
            # Create fact in database using repository
            created_fact = get_key_fact_repository().create(fact, human_input_id=human_input_id)
            fact_id = created_fact.id
        except RuntimeError as e:
            logger.error(f"Failed to access key fact repository: {str(e)}")
            console.print(f"Error storing fact: {str(e)}", style="red")
            continue

        # Record to trajectory before displaying panel
        try:
            trajectory_repo = get_trajectory_repository()
            trajectory_repo.create(
                tool_name="emit_key_facts",
                tool_parameters={"facts": [fact]},
                step_data={
                    "fact_id": fact_id,
                    "fact": fact,
                    "display_title": f"Key Fact #{fact_id}",
                },
                record_type="memory_operation",
                human_input_id=human_input_id
            )
        except RuntimeError as e:
            logger.warning(f"Failed to record trajectory: {str(e)}")

        # Display panel with ID
        console.print(
            Panel(
                Markdown(fact),
                title=f"💡 Key Fact #{fact_id}",
                border_style="bright_cyan",
            )
        )

        # Add result message
        results.append(f"Stored fact #{fact_id}: {fact}")

    log_work_event(f"Stored {len(facts)} key facts.")
    
    # Check if we need to clean up facts (more than 30)
    try:
        all_facts = get_key_fact_repository().get_all()
        if len(all_facts) > 50:
            # Trigger the key facts cleaner agent
            try:
                from ra_aid.agents.key_facts_gc_agent import run_key_facts_gc_agent
                run_key_facts_gc_agent()
            except Exception as e:
                logger.error(f"Failed to run key facts cleaner: {str(e)}")
    except RuntimeError as e:
        logger.error(f"Failed to access key fact repository: {str(e)}")
    
    return "Facts stored."




@tool("emit_key_snippet")
def emit_key_snippet(snippet_info: SnippetInfo) -> str:
    """Store a single source code snippet in the database which represents key information.
    Automatically adds the filepath of the snippet to related files.

    This is for **existing**, or **just-written** files, not for things to be created in the future.

    ONLY emit snippets if they will be relevant to UPCOMING work.

    Focus on external interfaces and things that are very specific and relevant to UPCOMING work.

    SNIPPETS SHOULD TYPICALLY BE MULTIPLE LINES, NOT SINGLE LINES, NOT ENTIRE FILES.

    Args:
        snippet_info: Dict with keys:
                 - filepath: Path to the source file
                 - line_number: Line number where the snippet starts
                 - snippet: The source code snippet text
                 - description: Optional description of the significance
    """
    # Add filepath to related files
    emit_related_files.invoke({"files": [snippet_info["filepath"]]})

    # Try to get the latest human input
    human_input_id = None
    try:
        human_input_repo = get_human_input_repository()
        human_input_id = human_input_repo.get_most_recent_id()
    except RuntimeError as e:
        logger.warning(f"No HumanInputRepository available: {str(e)}")
    except Exception as e:
        logger.warning(f"Failed to get recent human input: {str(e)}")

    # Create a new key snippet in the database
    key_snippet = get_key_snippet_repository().create(
        filepath=snippet_info["filepath"],
        line_number=snippet_info["line_number"],
        snippet=snippet_info["snippet"],
        description=snippet_info["description"],
        human_input_id=human_input_id,
    )
    
    # Get the snippet ID from the database record
    snippet_id = key_snippet.id

    # Format display text as markdown
    display_text = [
        "**Source Location**:",
        f"- File: `{snippet_info['filepath']}`",
        f"- Line: `{snippet_info['line_number']}`",
        "",  # Empty line before code block
        "**Code**:",
        "```python",
        snippet_info["snippet"].rstrip(),  # Remove trailing whitespace
        "```",
    ]
    if snippet_info["description"]:
        display_text.extend(["", "**Description**:", snippet_info["description"]])

    # Record to trajectory before displaying panel
    try:
        trajectory_repo = get_trajectory_repository()
        trajectory_repo.create(
            tool_name="emit_key_snippet",
            tool_parameters={
                "snippet_info": {
                    "filepath": snippet_info["filepath"],
                    "line_number": snippet_info["line_number"],
                    "description": snippet_info["description"],
                    # Omit the full snippet content to avoid duplicating large text in the database
                    "snippet_length": len(snippet_info["snippet"])
                }
            },
            step_data={
                "snippet_id": snippet_id,
                "filepath": snippet_info["filepath"],
                "line_number": snippet_info["line_number"],
                "display_title": f"Key Snippet #{snippet_id}",
            },
            record_type="memory_operation",
            human_input_id=human_input_id
        )
    except RuntimeError as e:
        logger.warning(f"Failed to record trajectory: {str(e)}")

    # Display panel
    console.print(
        Panel(
            Markdown("\n".join(display_text)),
            title=f"📝 Key Snippet #{snippet_id}",
            border_style="bright_cyan",
        )
    )

    log_work_event(f"Stored code snippet #{snippet_id}.")
    
    # Check if we need to clean up snippets (more than 20)
    all_snippets = get_key_snippet_repository().get_all()
    if len(all_snippets) > 35:
        # Trigger the key snippets cleaner agent
        try:
            from ra_aid.agents.key_snippets_gc_agent import run_key_snippets_gc_agent
            run_key_snippets_gc_agent()
        except Exception as e:
            logger.error(f"Failed to run key snippets cleaner: {str(e)}")
    
    return f"Snippet #{snippet_id} stored."


@tool("one_shot_completed")
def one_shot_completed(message: str) -> str:
    """Signal that a one-shot task has been completed and execution should stop.

    Only call this if you have already **fully** completed the original request.

    Args:
        message: Completion message to display
    """
    mark_task_completed(message)
    
    # Record to trajectory before displaying panel
    human_input_id = None
    try:
        human_input_id = get_human_input_repository().get_most_recent_id()
        trajectory_repo = get_trajectory_repository()
        trajectory_repo.create(
            tool_name="one_shot_completed",
            tool_parameters={"message": message},
            step_data={
                "completion_message": message,
                "display_title": "Task Completed",
            },
            record_type="task_completion",
            human_input_id=human_input_id
        )
    except RuntimeError as e:
        logger.warning(f"Failed to record trajectory: {str(e)}")
    
    console.print(Panel(Markdown(message), title="✅ Task Completed"))
    log_work_event(f"Task completed:\n\n{message}")
    return "Completion noted."


@tool("task_completed")
def task_completed(message: str) -> str:
    """Mark the current task as completed with a completion message.

    Args:
        message: Message explaining how/why the task is complete
    """
    mark_task_completed(message)
    
    # Record to trajectory before displaying panel
    human_input_id = None
    try:
        human_input_id = get_human_input_repository().get_most_recent_id()
        trajectory_repo = get_trajectory_repository()
        trajectory_repo.create(
            tool_name="task_completed",
            tool_parameters={"message": message},
            step_data={
                "completion_message": message,
                "display_title": "Task Completed",
            },
            record_type="task_completion",
            human_input_id=human_input_id
        )
    except RuntimeError as e:
        logger.warning(f"Failed to record trajectory: {str(e)}")
    
    console.print(Panel(Markdown(message), title="✅ Task Completed"))
    log_work_event(f"Task completed:\n\n{message}")
    return "Completion noted."


@tool("plan_implementation_completed")
def plan_implementation_completed(message: str) -> str:
    """Mark the entire implementation plan as completed.

    Args:
        message: Message explaining how the implementation plan was completed
    """
    mark_should_exit(propagation_depth=1)
    mark_plan_completed(message)
    
    # Record to trajectory before displaying panel
    human_input_id = None
    try:
        human_input_id = get_human_input_repository().get_most_recent_id()
        trajectory_repo = get_trajectory_repository()
        trajectory_repo.create(
            tool_name="plan_implementation_completed",
            tool_parameters={"message": message},
            step_data={
                "completion_message": message,
                "display_title": "Plan Executed",
            },
            record_type="plan_completion",
            human_input_id=human_input_id
        )
    except RuntimeError as e:
        logger.warning(f"Failed to record trajectory: {str(e)}")
    
    console.print(Panel(Markdown(message), title="✅ Plan Executed"))
    log_work_event(f"Completed implementation:\n\n{message}")
    return "Plan completion noted."


def get_related_files() -> List[str]:
    """Get the current list of related files.

    Returns:
        List of formatted strings in the format 'ID#X path/to/file.py'
    """
    repo = get_related_files_repository()
    return repo.format_related_files()


@tool("emit_related_files")
def emit_related_files(files: List[str]) -> str:
    """Store multiple related files that tools should work with.

    Args:
        files: List of file paths to add
    """
    repo = get_related_files_repository()
    
    # Store the repository's ID counter value before adding any files
    try:
        initial_next_id = repo.get_next_id()
    except (AttributeError, TypeError):
        # Handle case where repo is mocked in tests
        initial_next_id = 0  # Use a safe default for mocked environments
    
    results = []
    added_files = []
    invalid_paths = []
    binary_files = []

    # Process files
    for file in files:
        # First check if path exists
        if not os.path.exists(file):
            invalid_paths.append(file)
            results.append(f"Error: Path '{file}' does not exist")
            continue

        # Then check if it's a directory
        if os.path.isdir(file):
            invalid_paths.append(file)
            results.append(f"Error: Path '{file}' is a directory, not a file")
            continue

        # Finally validate it's a regular file
        if not os.path.isfile(file):
            invalid_paths.append(file)
            results.append(f"Error: Path '{file}' exists but is not a regular file")
            continue

        # Check if it's a binary file
        if is_binary_file(file):
            binary_files.append(file)
            results.append(f"Skipped binary file: '{file}'")
            continue

        # Add file to repository
        file_id = repo.add_file(file)
        
        if file_id is not None:
            # Check if it's a truly new file (ID >= initial_next_id)
            try:
                is_truly_new = file_id >= initial_next_id
            except TypeError:
                # Handle case where file_id or initial_next_id is mocked in tests
                is_truly_new = True  # Default to True in test environments
            
            # Also check for duplicates within this function call
            is_duplicate_in_call = False
            for r in results:
                if r.startswith(f"File ID #{file_id}:"):
                    is_duplicate_in_call = True
                    break
                    
            # Only add to added_files if it's truly new AND not a duplicate in this call
            if is_truly_new and not is_duplicate_in_call:
                added_files.append((file_id, file))  # Keep original path for display
                
            results.append(f"File ID #{file_id}: {file}")

    # Record to trajectory before displaying panel for added files
    if added_files:
        files_added_md = "\n".join(f"- `{file}`" for id, file in added_files)
        md_content = f"**Files Noted:**\n{files_added_md}"
        
        human_input_id = None
        try:
            human_input_id = get_human_input_repository().get_most_recent_id()
            trajectory_repo = get_trajectory_repository()
            trajectory_repo.create(
                tool_name="emit_related_files",
                tool_parameters={"files": files},
                step_data={
                    "added_files": [file for _, file in added_files],
                    "added_file_ids": [file_id for file_id, _ in added_files],
                    "display_title": "Related Files Noted",
                },
                record_type="memory_operation",
                human_input_id=human_input_id
            )
        except RuntimeError as e:
            logger.warning(f"Failed to record trajectory: {str(e)}")
        
        console.print(
            Panel(
                Markdown(md_content),
                title="📁 Related Files Noted",
                border_style="green",
            )
        )

    # Record to trajectory before displaying panel for binary files
    if binary_files:
        binary_files_md = "\n".join(f"- `{file}`" for file in binary_files)
        md_content = f"**Binary Files Skipped:**\n{binary_files_md}"
        
        human_input_id = None
        try:
            human_input_id = get_human_input_repository().get_most_recent_id()
            trajectory_repo = get_trajectory_repository()
            trajectory_repo.create(
                tool_name="emit_related_files",
                tool_parameters={"files": files},
                step_data={
                    "binary_files": binary_files,
                    "display_title": "Binary Files Not Added",
                },
                record_type="memory_operation",
                human_input_id=human_input_id
            )
        except RuntimeError as e:
            logger.warning(f"Failed to record trajectory: {str(e)}")
        
        console.print(
            Panel(
                Markdown(md_content),
                title="⚠️ Binary Files Not Added",
                border_style="yellow",
            )
        )

    # Return summary message
    if binary_files:
        binary_files_list = ", ".join(f"'{file}'" for file in binary_files)
        return f"Files noted. Binary files skipped: {binary_files_list}"
    else:
        return "Files noted."


def log_work_event(event: str) -> str:
    """Add timestamped entry to work log.

    Internal function used to track major events during agent execution.
    Each entry is stored with an ISO format timestamp.

    Args:
        event: Description of the event to log

    Returns:
        Confirmation message

    Note:
        Entries can be retrieved with get_work_log() as markdown formatted text.
    """
    try:
        repo = get_work_log_repository()
        repo.add_entry(event)
        return f"Event logged: {event}"
    except RuntimeError as e:
        logger.error(f"Failed to access work log repository: {str(e)}")
        return f"Failed to log event: {str(e)}"





def get_work_log() -> str:
    """Return formatted markdown of work log entries.

    Returns:
        Markdown formatted text with timestamps as headings and events as content,
        or 'No work log entries' if the log is empty.

    Example:
        ## 2024-12-23T11:39:10

        Task #1 added: Create login form
    """
    try:
        repo = get_work_log_repository()
        return repo.format_work_log()
    except RuntimeError as e:
        logger.error(f"Failed to access work log repository: {str(e)}")
        return "No work log entries"


def reset_work_log() -> str:
    """Clear the work log.

    Returns:
        Confirmation message

    Note:
        This permanently removes all work log entries. The operation cannot be undone.
    """
    try:
        repo = get_work_log_repository()
        repo.clear()
        return "Work log cleared"
    except RuntimeError as e:
        logger.error(f"Failed to access work log repository: {str(e)}")
        return f"Failed to clear work log: {str(e)}"


@tool("deregister_related_files")
def deregister_related_files(file_ids: List[int]) -> str:
    """Delete multiple related files by their IDs.
    Silently skips any IDs that don't exist.

    Args:
        file_ids: List of file IDs to delete
    """
    repo = get_related_files_repository()
    results = []
    
    for file_id in file_ids:
        deleted_file = repo.remove_file(file_id)
        if deleted_file:
            success_msg = (
                f"Successfully removed related file #{file_id}: {deleted_file}"
            )
            console.print(
                Panel(
                    Markdown(success_msg),
                    title="File Reference Removed",
                    border_style="green",
                )
            )
            results.append(success_msg)

    return "Files noted."
