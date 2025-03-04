import os
from typing import Any, Dict, List, Optional

try:
    import magic
except ImportError:
    magic = None

from langchain_core.tools import tool
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
from ra_aid.model_formatters import key_snippets_formatter
from ra_aid.logging_config import get_logger

logger = get_logger(__name__)


class WorkLogEntry(TypedDict):
    timestamp: str
    event: str


class SnippetInfo(TypedDict):
    filepath: str
    line_number: int
    snippet: str
    description: Optional[str]


console = Console()

# Import repositories using the get_* functions
from ra_aid.database.repositories.key_fact_repository import get_key_fact_repository

# Global memory store
_global_memory: Dict[str, Any] = {
    "plans": [],
    "tasks": {},  # Dict[int, str] - ID to task mapping
    "task_id_counter": 1,  # Counter for generating unique task IDs
    "implementation_requested": False,
    "related_files": {},  # Dict[int, str] - ID to filepath mapping
    "related_file_id_counter": 1,  # Counter for generating unique file IDs
    "agent_depth": 0,
    "work_log": [],  # List[WorkLogEntry] - Timestamped work events
}


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
        recent_inputs = human_input_repo.get_recent(1)
        if recent_inputs and len(recent_inputs) > 0:
            human_input_id = recent_inputs[0].id
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


@tool("emit_plan")
def emit_plan(plan: str) -> str:
    """Store a plan step in global memory.

    Args:
        plan: The plan step to store (markdown format; be clear, complete, use newlines, and use as many tokens as you need)
    """
    _global_memory["plans"].append(plan)
    console.print(Panel(Markdown(plan), title="📋 Plan"))
    log_work_event(f"Added plan step:\n\n{plan}")
    return "Plan stored."


@tool("emit_task")
def emit_task(task: str) -> str:
    """Store a task in global memory.

    Args:
        task: The task to store
    """
    # Get and increment task ID
    task_id = _global_memory["task_id_counter"]
    _global_memory["task_id_counter"] += 1

    # Store task with ID
    _global_memory["tasks"][task_id] = task

    console.print(Panel(Markdown(task), title=f"✅ Task #{task_id}"))
    log_work_event(f"Task #{task_id} added:\n\n{task}")
    return f"Task #{task_id} stored."


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
        recent_inputs = human_input_repo.get_recent(1)
        if recent_inputs and len(recent_inputs) > 0:
            human_input_id = recent_inputs[0].id
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


@tool("delete_tasks")
def delete_tasks(task_ids: List[int]) -> str:
    """Delete multiple tasks from global memory by their IDs.
    Silently skips any IDs that don't exist.

    Args:
        task_ids: List of task IDs to delete
    """
    results = []
    for task_id in task_ids:
        if task_id in _global_memory["tasks"]:
            # Delete the task
            deleted_task = _global_memory["tasks"].pop(task_id)
            success_msg = f"Successfully deleted task #{task_id}: {deleted_task}"
            console.print(
                Panel(Markdown(success_msg), title="Task Deleted", border_style="green")
            )
            results.append(success_msg)

    log_work_event(f"Deleted tasks {task_ids}.")
    return "Tasks deleted."


@tool("request_implementation")
def request_implementation() -> str:
    """Request that implementation proceed after research/planning.
    Used to indicate the agent should move to implementation stage.

    Think carefully before requesting implementation.
      Do you need to request research subtasks first?
      Have you run relevant unit tests, if they exist, to get a baseline (this can be a subtask)?
      Do you need to crawl deeper to find all related files and symbols?
    """
    _global_memory["implementation_requested"] = True
    console.print(Panel("🚀 Implementation Requested", style="yellow", padding=0))
    log_work_event("Implementation requested.")
    return "Implementation requested."


@tool("emit_key_snippet")
def emit_key_snippet(snippet_info: SnippetInfo) -> str:
    """Store a single source code snippet in the database which represents key information.
    Automatically adds the filepath of the snippet to related files.

    This is for **existing**, or **just-written** files, not for things to be created in the future.

    ONLY emit snippets if they will be relevant to UPCOMING work.

    Focus on external interfaces and things that are very specific and relevant to UPCOMING work.

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
        recent_inputs = human_input_repo.get_recent(1)
        if recent_inputs and len(recent_inputs) > 0:
            human_input_id = recent_inputs[0].id
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



@tool("swap_task_order")
def swap_task_order(id1: int, id2: int) -> str:
    """Swap the order of two tasks in global memory by their IDs.

    Args:
        id1: First task ID
        id2: Second task ID
    """
    # Validate IDs are different
    if id1 == id2:
        return "Cannot swap task with itself"

    # Validate both IDs exist
    if id1 not in _global_memory["tasks"] or id2 not in _global_memory["tasks"]:
        return "Invalid task ID(s)"

    # Swap the tasks
    _global_memory["tasks"][id1], _global_memory["tasks"][id2] = (
        _global_memory["tasks"][id2],
        _global_memory["tasks"][id1],
    )

    # Display what was swapped
    console.print(
        Panel(
            Markdown(f"Swapped:\n- Task #{id1} ↔️ Task #{id2}"),
            title="🔄 Tasks Reordered",
            border_style="green",
        )
    )

    return "Tasks deleted."


@tool("one_shot_completed")
def one_shot_completed(message: str) -> str:
    """Signal that a one-shot task has been completed and execution should stop.

    Only call this if you have already **fully** completed the original request.

    Args:
        message: Completion message to display
    """
    if _global_memory.get("implementation_requested", False):
        return "Cannot complete in one shot - implementation was requested"

    mark_task_completed(message)
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
    console.print(Panel(Markdown(message), title="✅ Task Completed"))
    log_work_event(f"Task completed:\n\n{message}")
    return "Completion noted."


@tool("plan_implementation_completed")
def plan_implementation_completed(message: str) -> str:
    """Mark the entire implementation plan as completed.

    Args:
        message: Message explaining how the implementation plan was completed
    """
    mark_should_exit()
    mark_plan_completed(message)
    _global_memory["tasks"].clear()  # Clear task list when plan is completed
    _global_memory["task_id_counter"] = 1
    console.print(Panel(Markdown(message), title="✅ Plan Executed"))
    log_work_event(f"Completed implementation:\n\n{message}")
    return "Plan completion noted and task list cleared."


def get_related_files() -> List[str]:
    """Get the current list of related files.

    Returns:
        List of formatted strings in the format 'ID#X path/to/file.py'
    """
    files = _global_memory["related_files"]
    return [f"ID#{file_id} {filepath}" for file_id, filepath in sorted(files.items())]


@tool("emit_related_files")
def emit_related_files(files: List[str]) -> str:
    """Store multiple related files that tools should work with.

    Args:
        files: List of file paths to add
    """
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

        # Normalize the path
        normalized_path = os.path.abspath(file)

        # Check if normalized path already exists in values
        existing_id = None
        for fid, fpath in _global_memory["related_files"].items():
            if fpath == normalized_path:
                existing_id = fid
                break

        if existing_id is not None:
            # File exists, use existing ID
            results.append(f"File ID #{existing_id}: {file}")
        else:
            # New file, assign new ID
            file_id = _global_memory["related_file_id_counter"]
            _global_memory["related_file_id_counter"] += 1

            # Store normalized path with ID
            _global_memory["related_files"][file_id] = normalized_path
            added_files.append((file_id, file))  # Keep original path for display
            results.append(f"File ID #{file_id}: {file}")

    # Rich output - single consolidated panel for added files
    if added_files:
        files_added_md = "\n".join(f"- `{file}`" for id, file in added_files)
        md_content = f"**Files Noted:**\n{files_added_md}"
        console.print(
            Panel(
                Markdown(md_content),
                title="📁 Related Files Noted",
                border_style="green",
            )
        )

    # Display skipped binary files
    if binary_files:
        binary_files_md = "\n".join(f"- `{file}`" for file in binary_files)
        md_content = f"**Binary Files Skipped:**\n{binary_files_md}"
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
    from datetime import datetime

    entry = WorkLogEntry(timestamp=datetime.now().isoformat(), event=event)
    _global_memory["work_log"].append(entry)
    return f"Event logged: {event}"


def is_binary_file(filepath):
    """Check if a file is binary using magic library if available."""
    # First check if file is empty
    if os.path.getsize(filepath) == 0:
        return False  # Empty files are not binary
        
    if magic:
        try:
            mime = magic.from_file(filepath, mime=True)
            file_type = magic.from_file(filepath)

            # If MIME type starts with 'text/', it's likely a text file
            if mime.startswith("text/"):
                return False
                
            # Also consider 'application/x-python' and similar script types as text
            if any(mime.startswith(prefix) for prefix in ['application/x-python', 'application/javascript']):
                return False
                
            # Check for common text file descriptors
            text_indicators = ["text", "script", "xml", "json", "yaml", "markdown", "HTML"]
            if any(indicator.lower() in file_type.lower() for indicator in text_indicators):
                return False
                
            # If none of the text indicators are present, assume it's binary
            return True
        except Exception:
            return _is_binary_fallback(filepath)
    else:
        return _is_binary_fallback(filepath)


def _is_binary_fallback(filepath):
    """Fallback method to detect binary files without using magic."""
    try:
        # First check if file is empty
        if os.path.getsize(filepath) == 0:
            return False  # Empty files are not binary
            
        with open(filepath, "r", encoding="utf-8") as f:
            chunk = f.read(1024)

            # Check for null bytes which indicate binary content
            if "\0" in chunk:
                return True

            # If we can read it as text without errors, it's probably not binary
            return False
    except UnicodeDecodeError:
        # If we can't decode as UTF-8, it's likely binary
        return True


def get_work_log() -> str:
    """Return formatted markdown of work log entries.

    Returns:
        Markdown formatted text with timestamps as headings and events as content,
        or 'No work log entries' if the log is empty.

    Example:
        ## 2024-12-23T11:39:10

        Task #1 added: Create login form
    """
    if not _global_memory["work_log"]:
        return "No work log entries"

    entries = []
    for entry in _global_memory["work_log"]:
        entries.extend(
            [
                f"## {entry['timestamp']}",
                "",
                entry["event"],
                "",  # Blank line between entries
            ]
        )

    return "\n".join(entries).rstrip()  # Remove trailing newline


def reset_work_log() -> str:
    """Clear the work log.

    Returns:
        Confirmation message

    Note:
        This permanently removes all work log entries. The operation cannot be undone.
    """
    _global_memory["work_log"].clear()
    return "Work log cleared"


@tool("deregister_related_files")
def deregister_related_files(file_ids: List[int]) -> str:
    """Delete multiple related files from global memory by their IDs.
    Silently skips any IDs that don't exist.

    Args:
        file_ids: List of file IDs to delete
    """
    results = []
    for file_id in file_ids:
        if file_id in _global_memory["related_files"]:
            # Delete the file reference
            deleted_file = _global_memory["related_files"].pop(file_id)
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


def get_memory_value(key: str) -> str:
    """
    Get a value from global memory.
    
    Note: Key facts and key snippets are handled by their respective repository and formatter modules,
    and should be accessed directly using those instead of through this function.

    Different memory types return different formats:
    - For work_log: Returns formatted markdown with timestamps and events
    - For research_notes: Returns formatted markdown from repository
    - For other types: Returns newline-separated list of values

    Args:
        key: The key to get from memory

    Returns:
        String representation of the memory values
    """
    if key == "work_log":
        values = _global_memory.get(key, [])
        if not values:
            return ""
        entries = [f"## {entry['timestamp']}\n{entry['event']}" for entry in values]
        return "\n\n".join(entries)
    
    if key == "research_notes":
        # DEPRECATED: This method of accessing research notes is deprecated.
        # Use direct repository access instead:
        # from ra_aid.database.repositories.research_note_repository import get_research_note_repository
        # from ra_aid.model_formatters.research_notes_formatter import format_research_notes_dict
        # repository = get_research_note_repository()
        # notes_dict = repository.get_notes_dict()
        # formatted_notes = format_research_notes_dict(notes_dict)
        logger.warning("DEPRECATED: Accessing research notes via get_memory_value() is deprecated. "
                       "Use direct repository access with get_research_note_repository() instead.")
        try:
            # Import required modules for research notes
            from ra_aid.database.repositories.research_note_repository import get_research_note_repository
            from ra_aid.model_formatters.research_notes_formatter import format_research_notes_dict
            
            # Get notes from repository and format them
            repository = get_research_note_repository()
            notes_dict = repository.get_notes_dict()
            return format_research_notes_dict(notes_dict)
        except RuntimeError as e:
            logger.error(f"Failed to access research note repository: {str(e)}")
            return ""
        except Exception as e:
            logger.error(f"Error accessing research notes: {str(e)}")
            return ""

    # For other types (lists), join with newlines
    values = _global_memory.get(key, [])
    return "\n".join(str(v) for v in values)