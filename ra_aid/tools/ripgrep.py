from typing import Dict, List, Union

from langchain_core.tools import tool
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

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
    before_context_lines: int = None,
    after_context_lines: int = None,
    file_type: str = None,
    case_sensitive: bool = True,
    include_hidden: bool = False,
    follow_links: bool = False,
    exclude_dirs: List[str] = None,
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
        if FILE_TYPE_MAP.get(file_type):
            file_type = FILE_TYPE_MAP.get(file_type)
        cmd.extend(["-t", file_type])

    # Add exclusions
    exclusions = DEFAULT_EXCLUDE_DIRS + (exclude_dirs or [])
    for dir in exclusions:
        cmd.extend(["--glob", f"!{dir}"])

    # Add the search pattern
    cmd.append(pattern)

    # Build info sections for display
    info_sections = []

    # Search parameters section
    params = [
        "## Search Parameters",
        f"**Pattern**: `{pattern}`",
        f"**Case Sensitive**: {case_sensitive}",
        f"**File Type**: {file_type or 'all'}",
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
    info_sections.append("\n".join(params))

    # Execute command
    console.print(
        Panel(
            Markdown(f"Searching for: **{pattern}**"),
            title="🔎 Ripgrep Search",
            border_style="bright_blue",
        )
    )
    try:
        print()
        output, return_code = run_interactive_command(cmd)
        print()
        decoded_output = output.decode() if output else ""

        return {
            "output": truncate_output(decoded_output),
            "return_code": return_code,
            "success": return_code == 0,
        }

    except Exception as e:
        error_msg = str(e)
        console.print(Panel(error_msg, title="❌ Error", border_style="red"))
        return {"output": error_msg, "return_code": 1, "success": False}
