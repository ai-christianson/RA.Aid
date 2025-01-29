import datetime
import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pathspec
from langchain_core.tools import tool
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.tree import Tree

console = Console()


@dataclass
class DirScanConfig:
    """Configuration for directory scanning"""

    max_depth: int
    follow_links: bool
    show_size: bool
    show_modified: bool
    exclude_patterns: List[str]


def format_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def format_time(timestamp: float) -> str:
    """Format timestamp as readable date"""
    dt = datetime.datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M")


# Default patterns to exclude
DEFAULT_EXCLUDE_PATTERNS = [
    ".*",  # Hidden files
    "__pycache__",  # Python cache
    "*.pyc",  # Python bytecode
    "node_modules",  # Node.js modules
    "*.swp",  # Vim swap files
    "*.swo",  # Vim swap files
    "*.swn",  # Vim swap files
    "*.class",  # Java bytecode
    "*.o",  # Object files
    "*.so",  # Shared libraries
    "*.dll",  # Dynamic libraries
    "*.exe",  # Executables
    "*.log",  # Log files
    "*.bak",  # Backup files
    "*.tmp",  # Temporary files
    "*.cache",  # Cache files
]


def load_gitignore_patterns(path: Path) -> pathspec.PathSpec:
    """Load gitignore patterns from .gitignore file or use defaults.

    Args:
        path: Directory path to search for .gitignore

    Returns:
        PathSpec object configured with the loaded patterns
    """
    gitignore_path = path / ".gitignore"
    patterns = []

    def modify_path(p: str) -> str:
        # Python pathspec doesn't treat `blah/` as a ignore folder, but `blah`. So we strip them
        p = p.strip()
        if p.endswith("/"):
            return p[:-1]
        return p

    # Load patterns from .gitignore if it exists
    if gitignore_path.exists():
        with open(gitignore_path) as f:
            patterns.extend(
                modify_path(line)
                for line in f
                if line.strip() and not line.startswith("#")
            )

    # add patterns from .aiderignore if it exists
    aiderignore_path = path / ".aiderignore"

    # Load patterns from .gitignore if it exists
    if aiderignore_path.exists():
        with open(aiderignore_path) as f:
            patterns.extend(
                modify_path(line)
                for line in f
                if line.strip() and not line.startswith("#")
            )

    # Add default patterns
    patterns.extend(DEFAULT_EXCLUDE_PATTERNS)

    return pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, patterns)


def should_ignore(path: str, spec: pathspec.PathSpec) -> bool:
    """Check if a path should be ignored based on gitignore patterns"""
    return spec.match_file(path)


def should_exclude(name: str, patterns: List[str]) -> bool:
    """Check if a file/directory name matches any exclude patterns"""
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def build_tree(
    path: Path,
    tree: Tree,
    config: DirScanConfig,
    current_depth: int = 0,
    spec: Optional[pathspec.PathSpec] = None,
) -> None:
    """Recursively build a Rich tree representation of the directory"""
    if current_depth >= config.max_depth:
        return

    try:
        # Get sorted list of directory contents
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))

        for entry in entries:
            # Get relative path from root for pattern matching
            rel_path = entry.relative_to(path)

            # Skip if path matches exclude patterns
            if spec and should_ignore(str(rel_path), spec):
                continue
            if should_exclude(entry.name, config.exclude_patterns):
                continue

            # Skip if symlink and not following links
            if entry.is_symlink() and not config.follow_links:
                continue

            try:
                if entry.is_dir():
                    # Add directory node
                    branch = tree.add(f"📁 {entry.name}/")

                    # Recursively process subdirectory
                    build_tree(entry, branch, config, current_depth + 1, spec)
                else:
                    # Add file node with optional metadata
                    meta = []
                    if config.show_size:
                        meta.append(format_size(entry.stat().st_size))
                    if config.show_modified:
                        meta.append(format_time(entry.stat().st_mtime))

                    label = entry.name
                    if meta:
                        label = f"{label} ({', '.join(meta)})"

                    tree.add(label)

            except PermissionError:
                tree.add(f"🔒 {entry.name} (Permission denied)")

    except PermissionError:
        tree.add("🔒 (Permission denied)")


@tool
def list_directory_tree(
    path: str = ".",
    *,
    max_depth: int = 1,  # Default to no recursion
    follow_links: bool = False,
    show_size: bool = False,  # Default to not showing size
    show_modified: bool = False,  # Default to not showing modified time
    exclude_patterns: List[str] = None,
) -> str:
    """List directory contents in a tree format with optional metadata.

    Args:
        path: Directory path to list
        max_depth: Maximum depth to traverse (default: 1 for no recursion)
        follow_links: Whether to follow symbolic links
        show_size: Show file sizes (default: False)
        show_modified: Show last modified times (default: False)
        exclude_patterns: List of patterns to exclude (uses gitignore syntax)

    Returns:
        Rendered tree string
    """
    root_path = Path(path).resolve()
    if not root_path.exists():
        raise ValueError(f"Path does not exist: {path}")
    if not root_path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    # Load .gitignore patterns if present
    spec = load_gitignore_patterns(root_path)

    # Create tree
    tree = Tree(f"📁 {root_path}/")
    config = DirScanConfig(
        max_depth=max_depth,
        follow_links=follow_links,
        show_size=show_size,
        show_modified=show_modified,
        exclude_patterns=DEFAULT_EXCLUDE_PATTERNS + (exclude_patterns or []),
    )

    # Build tree
    build_tree(root_path, tree, config, 0, spec)

    # Capture tree output
    with console.capture() as capture:
        console.print(tree)
    tree_str = capture.get()

    # Display panel
    console.print(
        Panel(
            Markdown(f"```\n{tree_str}\n```"),
            title="📂 Directory Tree",
            border_style="bright_blue",
        )
    )

    return tree_str
