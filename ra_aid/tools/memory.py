import os
from typing import Any, Dict, List, Optional, Set, Union

from langchain_core.tools import tool
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from typing_extensions import TypedDict


class WorkLogEntry(TypedDict):
    timestamp: str
    event: str


class SnippetInfo(TypedDict):
    """Type definition for source code snippet information"""

    filepath: str
    line_number: int
    snippet: str
    description: Optional[str]


console = Console()

# Global memory store
_global_memory: Dict[
    str,
    Union[
        List[Any],
        Dict[int, str],
        Dict[int, SnippetInfo],
        int,
        Set[str],
        bool,
        str,
        int,
        List[WorkLogEntry],
    ],
] = {
    "research_notes": [],
    "plans": [],
    "tasks": {},  # Dict[int, str] - ID to task mapping
    "task_completed": False,  # Flag indicating if task is complete
    "completion_message": "",  # Message explaining completion
    "task_id_counter": 1,  # Counter for generating unique task IDs
    "key_facts": {},  # Dict[int, str] - ID to fact mapping
    "key_fact_id_counter": 1,  # Counter for generating unique fact IDs
    "key_snippets": {},  # Dict[int, SnippetInfo] - ID to snippet mapping
    "key_snippet_id_counter": 1,  # Counter for generating unique snippet IDs
    "implementation_requested": False,
    "related_files": {},  # Dict[int, str] - ID to filepath mapping
    "related_file_id_counter": 1,  # Counter for generating unique file IDs
    "plan_completed": False,
    "agent_depth": 0,
    "work_log": [],  # List[WorkLogEntry] - Timestamped work events
}


@tool("emit_research_notes")
def emit_research_notes(notes: str) -> str:
    """Store research notes in global memory.

    Args:
        notes: REQUIRED The research notes to store

    Returns:
        The stored notes
    """
    _global_memory["research_notes"].append(notes)
    console.print(Panel(Markdown(notes), title="🔍 Research Notes"))
    return notes


@tool("emit_plan")
def emit_plan(plan: str) -> str:
    """Store a plan step in global memory.

    Args:
        plan: The plan step to store (markdown format; be clear, complete, use newlines, and use as many tokens as you need)

    Returns:
        The stored plan
    """
    _global_memory["plans"].append(plan)
    console.print(Panel(Markdown(plan), title="📋 Plan"))
    log_work_event(f"Added plan step:\n\n{plan}")
    return plan


@tool("emit_task")
def emit_task(task: str) -> str:
    """Store a task in global memory.

    Args:
        task: The task to store

    Returns:
        String confirming task storage with ID number
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

    Returns:
        List of stored fact confirmation messages
    """
    results = []
    for fact in facts:
        # Get and increment fact ID
        fact_id = _global_memory["key_fact_id_counter"]
        _global_memory["key_fact_id_counter"] += 1

        # Store fact with ID
        _global_memory["key_facts"][fact_id] = fact

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
    return "Facts stored."


@tool("delete_key_facts")
def delete_key_facts(fact_ids: List[int]) -> str:
    """Delete multiple key facts from global memory by their IDs.
    Silently skips any IDs that don't exist.

    Args:
        fact_ids: List of fact IDs to delete

    Returns:
        List of success messages for deleted facts
    """
    results = []
    for fact_id in fact_ids:
        if fact_id in _global_memory["key_facts"]:
            # Delete the fact
            deleted_fact = _global_memory["key_facts"].pop(fact_id)
            success_msg = f"Successfully deleted fact #{fact_id}: {deleted_fact}"
            console.print(
                Panel(Markdown(success_msg), title="Fact Deleted", border_style="green")
            )
            results.append(success_msg)

    log_work_event(f"Deleted facts {fact_ids}.")
    return "Facts deleted."


@tool("delete_tasks")
def delete_tasks(task_ids: List[int]) -> str:
    """Delete multiple tasks from global memory by their IDs.
    Silently skips any IDs that don't exist.

    Args:
        task_ids: List of task IDs to delete

    Returns:
        Confirmation message
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

    Returns:
        Empty string
    """
    _global_memory["implementation_requested"] = True
    console.print(Panel("🚀 Implementation Requested", style="yellow", padding=0))
    log_work_event("Implementation requested.")
    return ""


@tool("emit_key_snippets")
def emit_key_snippets(snippets: List[SnippetInfo]) -> str:
    """Store multiple key source code snippets in global memory.
    Automatically adds the filepaths of the snippets to related files.

    This is for **existing**, or **just-written** files, not for things to be created in the future.

    Args:
        snippets: REQUIRED List of snippet information dictionaries containing:
                 - filepath: Path to the source file
                 - line_number: Line number where the snippet starts
                 - snippet: The source code snippet text
                 - description: Optional description of the significance

    Returns:
        List of stored snippet confirmation messages
    """
    # First collect unique filepaths to add as related files
    emit_related_files.invoke(
        {"files": [snippet_info["filepath"] for snippet_info in snippets]}
    )

    results = []
    for snippet_info in snippets:
        # Get and increment snippet ID
        snippet_id = _global_memory["key_snippet_id_counter"]
        _global_memory["key_snippet_id_counter"] += 1

        # Store snippet info
        _global_memory["key_snippets"][snippet_id] = snippet_info

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

        results.append(f"Stored snippet #{snippet_id}")

    log_work_event(f"Stored {len(snippets)} code snippets.")
    return "Snippets stored."


@tool("delete_key_snippets")
def delete_key_snippets(snippet_ids: List[int]) -> str:
    """Delete multiple key snippets from global memory by their IDs.
    Silently skips any IDs that don't exist.

    Args:
        snippet_ids: List of snippet IDs to delete

    Returns:
        List of success messages for deleted snippets
    """
    results = []
    for snippet_id in snippet_ids:
        if snippet_id in _global_memory["key_snippets"]:
            # Delete the snippet
            deleted_snippet = _global_memory["key_snippets"].pop(snippet_id)
            success_msg = f"Successfully deleted snippet #{snippet_id} from {deleted_snippet['filepath']}"
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

    Returns:
        Success or error message depending on outcome
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

    return "Tasks swapped."


@tool("one_shot_completed")
def one_shot_completed(message: str) -> str:
    """Signal that a one-shot task has been completed and execution should stop.

    Only call this if you have already **fully** completed the original request.

    Args:
        message: Completion message to display
    """
    if _global_memory.get("implementation_requested", False):
        return "Cannot complete in one shot - implementation was requested"

    _global_memory["task_completed"] = True
    _global_memory["completion_message"] = message
    console.print(Panel(Markdown(message), title="✅ Task Completed"))
    log_work_event(f"Task completed\n\n{message}")
    return "Completion noted."


@tool("task_completed")
def task_completed(message: str) -> str:
    """Mark the current task as completed with a completion message.

    Args:
        message: Message explaining how/why the task is complete

    Returns:
        The completion message
    """
    _global_memory["task_completed"] = True
    _global_memory["completion_message"] = message
    console.print(Panel(Markdown(message), title="✅ Task Completed"))
    return "Completion noted."


@tool("plan_implementation_completed")
def plan_implementation_completed(message: str) -> str:
    """Mark the entire implementation plan as completed.

    Args:
        message: Message explaining how the implementation plan was completed

    Returns:
        Confirmation message
    """
    _global_memory["plan_completed"] = True
    _global_memory["completion_message"] = message
    _global_memory["tasks"].clear()  # Clear task list when plan is completed
    _global_memory["task_id_counter"] = 1
    console.print(Panel(Markdown(message), title="✅ Plan Executed"))
    log_work_event(f"Plan execution completed:\n\n{message}")
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

    Returns:
        Formatted string containing file IDs and paths for all processed files
    """
    results = []
    added_files = []
    invalid_paths = []

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

        # Check if file path already exists in values
        existing_id = None
        for fid, fpath in _global_memory["related_files"].items():
            if fpath == file:
                existing_id = fid
                break

        if existing_id is not None:
            # File exists, use existing ID
            results.append(f"File ID #{existing_id}: {file}")
        else:
            # New file, assign new ID
            file_id = _global_memory["related_file_id_counter"]
            _global_memory["related_file_id_counter"] += 1

            # Store file with ID
            _global_memory["related_files"][file_id] = file
            added_files.append((file_id, file))
            results.append(f"File ID #{file_id}: {file}")

    # Rich output - single consolidated panel
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

    return "\n".join(results)


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

    Returns:
        Success message string
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

    return "File references removed."


def get_memory_value(key: str) -> str:
    """Get a value from global memory.

    Different memory types return different formats:
    - key_facts: Returns numbered list of facts in format '#ID: fact'
    - key_snippets: Returns formatted snippets with file path, line number and content
    - All other types: Returns newline-separated list of values

    Args:
        key: The key to get from memory

    Returns:
        String representation of the memory values:
        - For key_facts: '#ID: fact' format, one per line
        - For key_snippets: Formatted snippet blocks
        - For other types: One value per line
    """
    values = _global_memory.get(key, [])

    if key == "key_facts":
        # For empty dict, return empty string
        if not values:
            return ""
        # Sort by ID for consistent output and format as markdown sections
        facts = []
        for k, v in sorted(values.items()):
            facts.extend(
                [
                    f"## 🔑 Key Fact #{k}",
                    "",  # Empty line for better markdown spacing
                    v,
                    "",  # Empty line between facts
                ]
            )
        return "\n".join(facts).rstrip()  # Remove trailing newline

    if key == "key_snippets":
        if not values:
            return ""
        # Format each snippet with file info and content using markdown
        snippets = []
        for k, v in sorted(values.items()):
            snippet_text = [
                f"## 📝 Code Snippet #{k}",
                "",  # Empty line for better markdown spacing
                "**Source Location**:",
                f"- File: `{v['filepath']}`",
                f"- Line: `{v['line_number']}`",
                "",  # Empty line before code block
                "**Code**:",
                "```python",
                v["snippet"].rstrip(),  # Remove trailing whitespace
                "```",
            ]
            if v["description"]:
                # Add empty line and description
                snippet_text.extend(["", "**Description**:", v["description"]])
            snippets.append("\n".join(snippet_text))
        return "\n\n".join(snippets)

    if key == "work_log":
        if not values:
            return ""
        entries = [f"## {entry['timestamp']}\n{entry['event']}" for entry in values]
        return "\n\n".join(entries)

    # For other types (lists), join with newlines
    return "\n".join(str(v) for v in values)
