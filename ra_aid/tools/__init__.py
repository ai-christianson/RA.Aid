"""Initialize tools for RA-Aid."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import inspect

# Import specific tool functions
# REMOVED: from .agent import run_agent_loop 
# REMOVED: from .code_completion import code_completion
# REMOVED: from .code_modification_editblock import code_modification_editblock
# REMOVED: from .create_plan import create_plan
from .expert import ask_expert, emit_expert_context
from .read_file import read_file_tool # Assuming this is the intended function name, not read_file_tool
from .file_str_replace import file_str_replace
from .write_file import put_complete_file_contents # Assuming this is the intended function name, not put_complete_file_contents
from .fuzzy_find import fuzzy_find_project_files
from .handle_user_defined_test_cmd_execution import execute_test_command
from .human import ask_human
from .list_directory import list_directory_tree
from .memory import (
    deregister_related_files,
    emit_key_facts,
    emit_key_snippet,
    emit_related_files,
    emit_research_notes,
    plan_implementation_completed,
    task_completed,
)
from .programmer import run_programming_task
# Note: read_file.py contains file_read, using that import
from .research import (
    existing_project_detected, 
    monorepo_detected, 
    ui_detected, 
    mark_research_complete_no_implementation_required
)
from .ripgrep import ripgrep_search
from .shell import run_shell_command
from .web_search_jina import web_search_jina
from .quick_web_search import quick_web_search # Added import
from .select_web_search import select_web_search # Added import
# Note: write_file.py contains file_write, using that import

# Define tool lists - This structure seems to have been replaced in the previous edit, restoring a basic available tools dict
AVAILABLE_TOOLS: Dict[str, Any] = {
    # REMOVED: "run_agent_loop": run_agent_loop, 
    # REMOVED: "code_completion": code_completion,
    # REMOVED: "code_modification_aider": code_modification_aider,
    # REMOVED: "code_modification_editblock": code_modification_editblock,
    # REMOVED: "create_plan": create_plan,
    "ask_expert": ask_expert,
    "emit_expert_context": emit_expert_context,
    "read_file_tool": read_file_tool,
    "file_str_replace": file_str_replace,
    "put_complete_file_contents": put_complete_file_contents,
    "fuzzy_find_project_files": fuzzy_find_project_files,
    # REMOVED: "git_tool": git_tool,
    "execute_test_command": execute_test_command,
    "ask_human": ask_human,
    "list_directory_tree": list_directory_tree,
    "deregister_related_files": deregister_related_files,
    "emit_key_facts": emit_key_facts,
    "emit_key_snippet": emit_key_snippet,
    "emit_related_files": emit_related_files,
    "emit_research_notes": emit_research_notes,
    "plan_implementation_completed": plan_implementation_completed,
    "task_completed": task_completed,
    "run_programming_task": run_programming_task,
    "existing_project_detected": existing_project_detected,
    "monorepo_detected": monorepo_detected,
    "ui_detected": ui_detected,
    "mark_research_complete_no_implementation_required": mark_research_complete_no_implementation_required,
    "ripgrep_search": ripgrep_search,
    "run_shell_command": run_shell_command,
    "web_search_jina": web_search_jina,
    "quick_web_search": quick_web_search, # Added tool
    "select_web_search": select_web_search, # Add the router tool
}

__all__ = [
    # REMOVED: "run_agent_loop", 
    # REMOVED: "code_completion",
    # REMOVED: "code_modification_aider",
    # REMOVED: "code_modification_editblock",
    # REMOVED: "create_plan": create_plan,
    "ask_expert",
    "emit_expert_context",
    "read_file_tool",
    "file_str_replace",
    "put_complete_file_contents",
    "fuzzy_find_project_files",
    # REMOVED: "git_tool",
    "execute_test_command",
    "ask_human",
    "list_directory_tree",
    "deregister_related_files",
    "emit_key_facts",
    "emit_key_snippet",
    "emit_related_files",
    "emit_research_notes",
    "plan_implementation_completed",
    "task_completed",
    "run_programming_task",
    "existing_project_detected",
    "monorepo_detected",
    "ui_detected",
    "mark_research_complete_no_implementation_required",
    "ripgrep_search",
    "run_shell_command",
    "web_search_jina",
    "quick_web_search", # Added tool
    "select_web_search", # Add the router tool
    # Include ToolMetadata, ToolArgument, ToolType if they are defined below
    "ToolMetadata",
    "ToolArgument",
    "ToolType",
]


# Tool metadata extraction (assuming this part was intended to be kept)
class ToolType(Enum):
    """Enum for tool types."""

    CODE_COMPLETION = "code_completion"
    CODE_MODIFICATION = "code_modification"
    PLANNING = "planning"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FINAL_ANSWER = "final_answer"
    GIT = "git"
    MEMORY = "memory"
    PROGRAMMING = "programming"
    RESEARCH = "research"
    SHELL = "shell"
    WEB_SEARCH = "web_search"
    EXPERT = "expert"
    HUMAN = "human"


@dataclass
class ToolArgument:
    """Dataclass for tool arguments."""

    name: str
    type: str
    description: str
    required: bool


@dataclass
class ToolMetadata:
    """Dataclass for tool metadata."""

    name: str
    description: str
    arguments: List[ToolArgument] = field(default_factory=list)
    type: Optional[ToolType] = None


def extract_tool_metadata(tool_func: Any) -> ToolMetadata:
    """Extract metadata from tool function docstring."""
    if not hasattr(tool_func, "__doc__") or not tool_func.__doc__:
        return ToolMetadata(name=tool_func.__name__, description="")

    docstring = inspect.getdoc(tool_func)
    lines = docstring.split("\n")
    description = lines[0] if lines else ""

    args_section = False
    arguments = []
    arg_name = ""
    arg_type = ""
    arg_desc_lines = []
    arg_required = True  # Assume required unless Optional is specified

    for line in lines[1:]:
        line = line.strip()
        if line == "Args:":
            args_section = True
            continue
        if line == "Returns:" or line == "Raises:":
            args_section = False
            if arg_name:  # Save the last argument
                arguments.append(
                    ToolArgument(
                        name=arg_name,
                        type=arg_type,
                        description=" ".join(arg_desc_lines),
                        required=arg_required,
                    )
                )
            break

        if args_section:
            if ":" in line and not line.startswith((" ", "\t")):
                if arg_name:  # Save the previous argument
                    arguments.append(
                        ToolArgument(
                            name=arg_name,
                            type=arg_type,
                            description=" ".join(arg_desc_lines),
                            required=arg_required,
                        )
                    )

                parts = line.split(":", 1)
                arg_name_type = parts[0].strip()
                arg_desc_lines = [parts[1].strip()]

                if "(" in arg_name_type and ")" in arg_name_type:
                    arg_name = arg_name_type.split("(")[0].strip()
                    type_str = arg_name_type.split("(")[1].split(")")[0].strip()
                    arg_type = type_str
                    arg_required = not type_str.startswith("Optional")
                else:
                    arg_name = arg_name_type
                    arg_type = "Any" # Default type if not specified
                    arg_required = True

            elif arg_name:
                arg_desc_lines.append(line)

    if arg_name and args_section: # Save the very last argument if loop finished
         arguments.append(
             ToolArgument(
                 name=arg_name,
                 type=arg_type,
                 description=" ".join(arg_desc_lines),
                 required=arg_required,
             )
         )

    # Infer tool type based on name or keywords
    tool_type = None
    name_lower = tool_func.__name__.lower()
    if "code" in name_lower or "modification" in name_lower:
        tool_type = ToolType.CODE_MODIFICATION
    if "complete" in name_lower:
        tool_type = ToolType.CODE_COMPLETION
    elif "plan" in name_lower:
        tool_type = ToolType.PLANNING
    elif "file_read" in name_lower or "read_file" in name_lower:
         tool_type = ToolType.FILE_READ
    elif "file_write" in name_lower or "write_file" in name_lower:
         tool_type = ToolType.FILE_WRITE
    elif "final_answer" in name_lower:
        tool_type = ToolType.FINAL_ANSWER
    elif "git" in name_lower:
        tool_type = ToolType.GIT
    elif "memory" in name_lower or "emit" in name_lower or "register" in name_lower:
        tool_type = ToolType.MEMORY
    elif "program" in name_lower:
        tool_type = ToolType.PROGRAMMING
    elif "research" in name_lower or "detect" in name_lower:
        tool_type = ToolType.RESEARCH
    elif "shell" in name_lower or "ripgrep" in name_lower: # ripgrep uses shell
        tool_type = ToolType.SHELL
    elif "web_search" in name_lower:
        tool_type = ToolType.WEB_SEARCH
    elif "expert" in name_lower:
        tool_type = ToolType.EXPERT
    elif "human" in name_lower:
        tool_type = ToolType.HUMAN


    return ToolMetadata(
        name=tool_func.__name__,
        description=description,
        arguments=arguments,
        type=tool_type,
    )


# Example usage (can be removed or adapted)
# for name, tool in AVAILABLE_TOOLS.items():
#     metadata = extract_tool_metadata(tool)
#     print(f"Tool: {metadata.name} ({metadata.type})")
#     print(f"  Description: {metadata.description}")
#     if metadata.arguments:
#         print("  Arguments:")
#         for arg in metadata.arguments:
#             req_status = "Required" if arg.required else "Optional"
#             print(f"    - {arg.name} ({arg.type}, {req_status}): {arg.description}")
#     print("---")