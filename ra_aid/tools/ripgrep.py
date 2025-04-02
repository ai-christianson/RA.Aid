from typing import Dict, List, Union, Optional

from langchain_core.tools import tool
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ra_aid.console.formatting import console_panel, cpm
from ra_aid.database.repositories.human_input_repository import get_human_input_repository
from ra_aid.database.repositories.trajectory_repository import get_trajectory_repository
from ra_aid.proc.interactive import run_interactive_command
from ra_aid.text.processing import truncate_output

console = Console()

DEFAULT_EXCLUDE_DIRS = [
    ".git",
    "node_modules",
    "vendor",
    ".venv",
    "__pycache__",
    ".cache",
    "dist",
    "build",
    "env",
    ".env",
    "venv",
    ".idea",
    ".vscode",
    ".ra-aid",
]


FILE_TYPE_MAP = {
    # General programming languages
    "py": "python",
    "rs": "rust",
    "js": "javascript",
    "ts": "typescript",
    "java": "java",
    "c": "c",
    "cpp": "cpp",
    "h": "c-header",
    "hpp": "cpp-header",
    "cs": "csharp",
    "go": "go",
    "rb": "ruby",
    "php": "php",
    "swift": "swift",
    "kt": "kotlin",
    "sh": "sh",
    "bash": "sh",
    "r": "r",
    "pl": "perl",
    "scala": "scala",
    "dart": "dart",
    # Markup, data, and web
    "html": "html",
    "htm": "html",
    "xml": "xml",
    "css": "css",
    "scss": "scss",
    "json": "json",
    "yaml": "yaml",
    "yml": "yaml",
    "toml": "toml",
    "md": "markdown",
    "markdown": "markdown",
    "sql": "sql",
    "psql": "postgres",
}


@tool
def ripgrep_search(
    pattern: str,
    *,
    before_context_lines: Optional[int] = None,
    after_context_lines: Optional[int] = None,
    file_type: Optional[str] = None,
    case_sensitive: bool = True,
    include_hidden: bool = False,
    follow_links: bool = False,
    exclude_dirs: Optional[List[str]] = None,
    include_paths: Optional[List[str]] = None,
    fixed_string: bool = False,
) -> Dict[str, Union[str, int, bool]]:
    """Execute a ripgrep (rg) search with formatting and common options.

    Prefer to use this with after_context_lines and/or before_context_lines over reading entire file contents, to conserve tokens and resources.

    Args:
        pattern: Search pattern to find
        before_context_lines: Number of lines to show before each match (default: None)
        after_context_lines: Number of lines to show after each match (default: None)
        file_type: Optional file type to filter results (e.g. 'py' for Python files)
        case_sensitive: Whether to do case-sensitive search (default: True)
        include_hidden: Whether to search hidden files and directories (default: False)
        follow_links: Whether to follow symbolic links (default: False)
        exclude_dirs: Additional directories to exclude (combines with defaults)
        include_paths: Optional list of specific file or directory paths to search within.
                       If provided, rg will only search these paths.
        fixed_string: Whether to treat pattern as a literal string instead of regex (default: False)
    """
    # Build rg command with options
    cmd = ["rg", "--color", "always"]

    if before_context_lines is not None:
        cmd.extend(["-B", str(before_context_lines)])

    if after_context_lines is not None:
        cmd.extend(["-A", str(after_context_lines)])

    if not case_sensitive:
        cmd.append("-i")

    if include_hidden:
        cmd.append("--hidden")

    if follow_links:
        cmd.append("--follow")

    if file_type:
        mapped_type = FILE_TYPE_MAP.get(file_type)
        if mapped_type:
            cmd.extend(["-t", mapped_type])
        else:
             cmd.extend(["-t", file_type]) # Pass original if not in map

    # Add exclusions
    exclusions = DEFAULT_EXCLUDE_DIRS + (exclude_dirs or [])
    for dir in exclusions:
        cmd.extend(["--glob", f"!{dir}"])

    # Add fixed string flag if specified
    if fixed_string:
        cmd.append("-F")

    # Add the search pattern
    cmd.append(pattern)

    # Add include paths if specified
    if include_paths:
        cmd.extend(include_paths)

    # Build info string for display
    info_lines = []

    # Search parameters section
    params = [
        "## Search Parameters",
        f"**Pattern**: `{pattern}`",
        f"**Case Sensitive**: {case_sensitive}",
        f"**File Type**: {file_type or 'all'}",
        f"**Fixed String**: {fixed_string}",
    ]
    if before_context_lines is not None:
        params.append(f"**Before Context Lines**: {before_context_lines}")
    if after_context_lines is not None:
        params.append(f"**After Context Lines**: {after_context_lines}")

    if include_hidden:
        params.append("**Including Hidden Files**: yes")
    if follow_links:
        params.append("**Following Symlinks**: yes")
    if exclude_dirs:
        params.append("\n**Additional Exclusions**:")
        for dir in exclude_dirs:
            params.append(f"- `{dir}`")
    # Use \n for Markdown line breaks
    if include_paths:
        params.append("\n**Included Paths**:") # Corrected newline
        for path in include_paths:
            params.append(f"- `{path}`")
    info_lines.append("\n".join(params)) # Corrected newline join

    # Execute command
    # Record ripgrep search in trajectory
    trajectory_repo = get_trajectory_repository()
    human_input_id = get_human_input_repository().get_most_recent_id()
    trajectory_repo.create(
        tool_name="ripgrep_search",
        tool_parameters={
            "pattern": pattern,
            "before_context_lines": before_context_lines,
            "after_context_lines": after_context_lines,
            "file_type": file_type,
            "case_sensitive": case_sensitive,
            "include_hidden": include_hidden,
            "follow_links": follow_links,
            "exclude_dirs": exclude_dirs,
            "include_paths": include_paths,
            "fixed_string": fixed_string
        },
        step_data={
            "search_pattern": pattern,
            "display_title": "Ripgrep Search",
        },
        record_type="tool_execution",
        human_input_id=human_input_id
    )

    cpm(
        f"Searching for: **{pattern}**{' in ' + ', '.join(include_paths) if include_paths else ''}",
        title="🔎 Ripgrep Search",
        border_style="bright_blue"
    )
    try:
        print()
        output, return_code = run_interactive_command(cmd)
        print()
        decoded_output = output.decode() if output else ""

        # Update trajectory with results
        trajectory_repo.create(
            tool_name="ripgrep_search",
            tool_parameters={"pattern": pattern, "after_context_lines": after_context_lines, "before_context_lines": before_context_lines},
            tool_result={"output": truncate_output(decoded_output), "return_code": return_code, "success": return_code == 0}
        )

        if return_code != 0:
            # Only show the panel if there's actual output
            if decoded_output and decoded_output.strip():
                console_panel(truncate_output(decoded_output), title="🚨 Error", border_style="red")
            # Always return failure on non-zero exit code
            return {"output": truncate_output(decoded_output), "return_code": return_code, "success": False}

        return {
            "output": truncate_output(decoded_output),
            "return_code": return_code,
            "success": return_code == 0,
        }

    except Exception as e:
        error_msg = str(e)

        # Record error in trajectory
        trajectory_repo.create(
            tool_name="ripgrep_search",
            tool_parameters={"pattern": pattern, "after_context_lines": after_context_lines, "before_context_lines": before_context_lines},
            tool_result={"output": error_msg, "return_code": 1, "success": False},
            is_error=True,
            error_message=error_msg,
            error_type=type(e).__name__
        )

        console_panel(error_msg, title="❌ Error", border_style="red")
        return {"output": error_msg, "return_code": 1, "success": False}
