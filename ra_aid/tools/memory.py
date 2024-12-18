from typing import Dict, List, Any, Union, TypedDict, Optional, Sequence, Set
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from langchain_core.tools import tool

class SnippetInfo(TypedDict):
    """Type definition for source code snippet information"""
    filepath: str
    line_number: int
    snippet: str
    description: Optional[str]

console = Console()

# Global memory store
_global_memory: Dict[str, Union[List[Any], Dict[int, str], Dict[int, SnippetInfo], int, Set[str], bool, str]] = {
    'research_notes': [],
    'plans': [],
    'tasks': {},  # Dict[int, str] - ID to task mapping
    'task_completed': False,  # Flag indicating if task is complete
    'completion_message': '',  # Message explaining completion
    'task_id_counter': 0,  # Counter for generating unique task IDs
    'research_subtasks': [],
    'key_facts': {},  # Dict[int, str] - ID to fact mapping
    'key_fact_id_counter': 0,  # Counter for generating unique fact IDs
    'key_snippets': {},  # Dict[int, SnippetInfo] - ID to snippet mapping
    'key_snippet_id_counter': 0,  # Counter for generating unique snippet IDs
    'implementation_requested': [],
    'implementation_skipped': [],
    'related_files': set()
}

@tool("emit_research_notes")
def emit_research_notes(notes: str) -> str:
    """Store research notes in global memory.
    
    Args:
        notes: The research notes to store
        
    Returns:
        The stored notes
    """
    _global_memory['research_notes'].append(notes)
    console.print(Panel(Markdown(notes), title="🔍 Research Notes"))
    return notes

@tool("emit_plan")
def emit_plan(plan: str) -> str:
    """Store a plan step in global memory.
    
    Args:
        plan: The plan step to store
        
    Returns:
        The stored plan
    """
    _global_memory['plans'].append(plan)
    console.print(Panel(Markdown(plan), title="📋 Plan"))
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
    task_id = _global_memory['task_id_counter']
    _global_memory['task_id_counter'] += 1
    
    # Store task with ID
    _global_memory['tasks'][task_id] = task
    
    console.print(Panel(Markdown(task), title=f"✅ Task #{task_id}"))
    return f"Task #{task_id} stored."

@tool("request_research_subtask")
def request_research_subtask(subtask: str) -> str:
    """Spawn a research subtask for investigation of a specific topic.

    Use this anytime you can to offload your work to specific things that need to be looked into.
    
    Args:
        subtask: Detailed description of the research subtask
        
    Returns:
        Confirmation message
    """
    _global_memory['research_subtasks'].append(subtask)
    console.print(Panel(Markdown(subtask), title="🔬 Research Subtask"))
    return "Subtask added."


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
        fact_id = _global_memory['key_fact_id_counter']
        _global_memory['key_fact_id_counter'] += 1
        
        # Store fact with ID
        _global_memory['key_facts'][fact_id] = fact
        
        # Display panel with ID
        console.print(Panel(Markdown(fact), title=f"💡 Key Fact #{fact_id}", border_style="bright_cyan"))
        
        # Add result message
        results.append(f"Stored fact #{fact_id}: {fact}")
        
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
        if fact_id in _global_memory['key_facts']:
            # Delete the fact
            deleted_fact = _global_memory['key_facts'].pop(fact_id)
            success_msg = f"Successfully deleted fact #{fact_id}: {deleted_fact}"
            console.print(Panel(Markdown(success_msg), title="Fact Deleted", border_style="green"))
            results.append(success_msg)
            
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
        if task_id in _global_memory['tasks']:
            # Delete the task
            deleted_task = _global_memory['tasks'].pop(task_id)
            success_msg = f"Successfully deleted task #{task_id}: {deleted_task}"
            console.print(Panel(Markdown(success_msg), 
                              title="Task Deleted", 
                              border_style="green"))
            results.append(success_msg)
            
    return "Tasks deleted."

@tool("request_implementation")
def request_implementation(reason: str) -> str:
    """Request that implementation proceed after research/planning.
    Used to indicate the agent should move to implementation stage.

    Think carefully before requesting implementation.
      Do you need to request research subtasks first?
      Have you run relevant unit tests, if they exist, to get a baseline (this can be a subtask)?
      Do you need to crawl deeper to find all related files and symbols?
    
    Args:
        reason: Why implementation should proceed
        
    Returns:
        The stored reason
    """
    _global_memory['implementation_requested'].append(reason)
    console.print(Panel(Markdown(reason), title="🚀 Implementation Requested"))
    return reason


@tool("skip_implementation")
def skip_implementation(reason: str) -> str:
    """Indicate that implementation can be skipped.
    Used when research/planning determines no changes are needed.
    
    Args:
        reason: Why implementation can be skipped
        
    Returns:
        The stored reason
    """
    _global_memory['implementation_skipped'].append(reason)
    console.print(Panel(Markdown(reason), title="⏭️ Implementation Skipped"))
    return reason

@tool("emit_key_snippets")
def emit_key_snippets(snippets: List[SnippetInfo]) -> str:
    """Store multiple key source code snippets in global memory.
    Automatically adds the filepaths of the snippets to related files.
    
    Args:
        snippets: List of snippet information dictionaries containing:
                 - filepath: Path to the source file
                 - line_number: Line number where the snippet starts  
                 - snippet: The source code snippet text
                 - description: Optional description of the significance
                 
    Returns:
        List of stored snippet confirmation messages
    """
    # First collect unique filepaths to add as related files
    _global_memory['related_files'].update(snippet_info['filepath'] for snippet_info in snippets)

    results = []
    for snippet_info in snippets:
        # Get and increment snippet ID 
        snippet_id = _global_memory['key_snippet_id_counter']
        _global_memory['key_snippet_id_counter'] += 1
        
        # Store snippet info
        _global_memory['key_snippets'][snippet_id] = snippet_info
        
        # Format display text as markdown
        display_text = [
            f"**Source Location**:",
            f"- File: `{snippet_info['filepath']}`",
            f"- Line: `{snippet_info['line_number']}`",
            "",  # Empty line before code block
            "**Code**:",
            "```python",
            snippet_info['snippet'].rstrip(),  # Remove trailing whitespace 
            "```"
        ]
        if snippet_info['description']:
            display_text.extend(["", "**Description**:", snippet_info['description']])
            
        # Display panel
        console.print(Panel(Markdown("\n".join(display_text)), 
                          title=f"📝 Key Snippet #{snippet_id}", 
                          border_style="bright_cyan"))
        
        results.append(f"Stored snippet #{snippet_id}")
        
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
        if snippet_id in _global_memory['key_snippets']:
            # Delete the snippet
            deleted_snippet = _global_memory['key_snippets'].pop(snippet_id)
            success_msg = f"Successfully deleted snippet #{snippet_id} from {deleted_snippet['filepath']}"
            console.print(Panel(Markdown(success_msg), 
                              title="Snippet Deleted", 
                              border_style="green"))
            results.append(success_msg)
            
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
    if id1 not in _global_memory['tasks'] or id2 not in _global_memory['tasks']:
        return "Invalid task ID(s)"
        
    # Swap the tasks
    _global_memory['tasks'][id1], _global_memory['tasks'][id2] = \
        _global_memory['tasks'][id2], _global_memory['tasks'][id1]
    
    # Display what was swapped
    console.print(Panel(
        Markdown(f"Swapped:\n- Task #{id1} ↔️ Task #{id2}"),
        title="🔄 Tasks Reordered",
        border_style="green"
    ))
    
    return "Tasks swapped."

@tool("one_shot_completed")
def one_shot_completed(message: str) -> str:
    """Signal that a one-shot task has been completed and execution should stop.

    Only call this if you have already **fully** completed the original request.
    
    Args:
        message: Completion message to display
        
    Returns:
        Original message if task can be completed, or error message if there are
        pending subtasks or implementation requests
    """
    if len(_global_memory['research_subtasks']) > 0:
        return "Cannot complete in one shot - research subtasks pending"
    if len(_global_memory['implementation_requested']) > 0:
        return "Cannot complete in one shot - implementation was requested"
        
    _global_memory['task_completed'] = True
    _global_memory['completion_message'] = message
    return message

def get_related_files() -> Set[str]:
    """Get the current set of related files.
    
    Returns:
        Set of file paths that have been marked as related
    """
    return _global_memory['related_files']

@tool("emit_related_files")
def emit_related_files(files: List[str]) -> str:
    """Store multiple related files that tools should work with.
    
    Args:
        files: List of file paths to add
        
    Returns:
        Confirmation message
    """
    results = []
    added_files = []
    
    # Process unique files
    for file in set(files):  # Remove duplicates in input
        if file not in _global_memory['related_files']:
            _global_memory['related_files'].add(file)
            added_files.append(file)
            results.append(f"Added related file: {file}")
    
    # Rich output - single consolidated panel
    if added_files:
        files_added_md = '\n'.join(f"- `{file}`" for file in added_files)
        md_content = f"**Files Noted:**\n{files_added_md}"
        console.print(Panel(Markdown(md_content), 
                          title="📁 Related Files Noted", 
                          border_style="green"))
    
    return "Files noted."

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
    
    if key == 'key_facts':
        # For empty dict, return empty string
        if not values:
            return ""
        # Sort by ID for consistent output and format as markdown sections
        facts = []
        for k, v in sorted(values.items()):
            facts.extend([
                f"## 🔑 Key Fact #{k}",
                "",  # Empty line for better markdown spacing
                v,
                ""  # Empty line between facts
            ])
        return "\n".join(facts).rstrip()  # Remove trailing newline
    
    if key == 'key_snippets':
        if not values:
            return ""
        # Format each snippet with file info and content using markdown
        snippets = []
        for k, v in sorted(values.items()):
            snippet_text = [
                f"## 📝 Code Snippet #{k}",
                "",  # Empty line for better markdown spacing
                f"**Source Location**:",
                f"- File: `{v['filepath']}`",
                f"- Line: `{v['line_number']}`",
                "",  # Empty line before code block
                "**Code**:",
                "```python",
                v['snippet'].rstrip(),  # Remove trailing whitespace
                "```"
            ]
            if v['description']:
                # Add empty line and description
                snippet_text.extend(["", "**Description**:", v['description']])
            snippets.append("\n".join(snippet_text))
        return "\n\n".join(snippets)
    
    # For other types (lists), join with newlines
    return "\n".join(str(v) for v in values)
