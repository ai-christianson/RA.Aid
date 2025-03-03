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
from ra_aid.database.repositories.key_fact_repository import KeyFactRepository
from ra_aid.database.repositories.key_snippet_repository import KeySnippetRepository
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

# Initialize repository for key facts
key_fact_repository = KeyFactRepository()

# Initialize repository for key snippets
key_snippet_repository = KeySnippetRepository()

# Global memory store
_global_memory: Dict[str, Any] = {
    "research_notes": [],
    "plans": [],
    "tasks": {},  # Dict[int, str] - ID to task mapping
    "task_id_counter": 1,  # Counter for generating unique task IDs
    "key_facts": {},  # Dict[int, str] - ID to fact mapping (deprecated, using DB now)
    "key_fact_id_counter": 1,  # Counter for generating unique fact IDs (deprecated, using DB now)
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
    _global_memory["research_notes"].append(notes)
    console.print(Panel(Markdown(notes), title="🔍 Research Notes"))
    return "Research notes stored."


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
    for fact in facts:
        # Create fact in database using repository
        created_fact = key_fact_repository.create(fact)
        fact_id = created_fact.id

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
    all_facts = key_fact_repository.get_all()
    if len(all_facts) > 30:
        # Trigger the key facts cleaner agent
        try:
            from ra_aid.agents.key_facts_gc_agent import run_key_facts_gc_agent
            run_key_facts_gc_agent()
        except Exception as e:
            logger.error(f"Failed to run key facts cleaner: {str(e)}")
    
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

    # Create a new key snippet in the database
    key_snippet = key_snippet_repository.create(
        filepath=snippet_info["filepath"],
        line_number=snippet_info["line_number"],
        snippet=snippet_info["snippet"],
        description=snippet_info["description"],
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
    all_snippets = key_snippet_repository.get_all()
    if len(all_snippets) > 20:
        # Trigger the key snippets cleaner agent
        try:
            from ra_aid.agents.key_snippets_gc_agent import run_key_snippets_gc_agent
            run_key_snippets_gc_agent()
        except Exception as e:
            logger.error(f"Failed to run key snippets cleaner: {str(e)}")
    
    return f"Snippet #{snippet_id} stored."


@tool("delete_key_snippets")
def delete_key_snippets(snippet_ids: List[int]) -> str:
    """Delete multiple key snippets from the database by their IDs.
    Silently skips any IDs that don't exist.

    Args:
        snippet_ids: List of snippet IDs to delete
    """
    results = []
    for snippet_id in snippet_ids:
        # Get the snippet first to capture filepath for the message
        snippet = key_snippet_repository.get(snippet_id)
        if snippet:
            filepath = snippet.filepath
            # Delete from database
            success = key_snippet_repository.delete(snippet_id)
            if success:
                success_msg = f"Successfully deleted snippet #{snippet_id} from {filepath}"
                console.print(
                    Panel(
                        Markdown(success_msg), title="Snippet Deleted", border_style="green"
                    )
                )
                results.append(success_msg)

    log_work_event(f"Deleted snippets {snippet_ids}.")
    return "Snippets deleted."


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
    if magic:
        try:
            mime = magic.from_file(filepath, mime=True)
            file_type = magic.from_file(filepath)

            if not mime.startswith("text/"):
                return True

            if "ASCII text" in file_type:
                return False

            return True
        except Exception:
            return _is_binary_fallback(filepath)
    else:
        return _is_binary_fallback(filepath)


def _is_binary_fallback(filepath):
    """Fallback method to detect binary files without using magic."""
    try:
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
    
    Note: Key facts and key snippets are handled by their respective repositories 
    and formatter modules.

    Different memory types return different formats:
    - key_snippets: Returns formatted snippets with file path, line number and content
    - All other types: Returns newline-separated list of values

    Args:
        key: The key to get from memory

    Returns:
        String representation of the memory values:
        - For key_snippets: Formatted snippet blocks
        - For other types: One value per line
    """
    if key == "key_snippets":
        try:
            # Get snippets from repository
            snippets_dict = key_snippet_repository.get_snippets_dict()
            return key_snippets_formatter.format_key_snippets_dict(snippets_dict)
        except Exception as e:
            logger.error(f"Error retrieving key snippets: {str(e)}")
            return ""

    if key == "work_log":
        values = _global_memory.get(key, [])
        if not values:
            return ""
        entries = [f"## {entry['timestamp']}\n{entry['event']}" for entry in values]
        return "\n\n".join(entries)

    # For other types (lists), join with newlines
    values = _global_memory.get(key, [])
    return "\n".join(str(v) for v in values)