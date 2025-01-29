"""Module providing unified interface for project information."""

from dataclasses import dataclass
from typing import List, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

__all__ = [
    "ProjectInfo",
    "ProjectInfoError",
    "get_project_info",
    "format_project_info",
    "display_project_status",
]

from ra_aid.file_listing import FileListerError, get_file_listing
from ra_aid.project_state import ProjectStateError, is_new_project


@dataclass
class ProjectInfo:
    """Data class containing project information.

    Attributes:
        is_new: Whether the project is new/empty
        files: List of tracked files in the project
        total_files: Total number of tracked files (before any limit)
    """

    is_new: bool
    files: List[str]
    total_files: int


class ProjectInfoError(Exception):
    """Base exception for project info related errors."""

    pass


def get_project_info(directory: str, file_limit: Optional[int] = None) -> ProjectInfo:
    """
    Get unified project information including new status and file listing.

    Args:
        directory: Path to the project directory
        file_limit: Optional maximum number of files to return in listing

    Returns:
        ProjectInfo: Object containing project information

    Raises:
        ProjectInfoError: If there are any errors accessing project information
        ProjectStateError: If there are errors checking project state
        FileListerError: If there are errors listing files
    """
    try:
        # Check if project is new
        new_status = is_new_project(directory)

        # Get file listing
        files, total = get_file_listing(directory, limit=file_limit)

        return ProjectInfo(is_new=new_status, files=files, total_files=total)

    except (ProjectStateError, FileListerError):
        # Re-raise known errors
        raise
    except Exception as e:
        # Wrap unexpected errors
        raise ProjectInfoError(f"Error getting project info: {e}")


def format_project_info(info: ProjectInfo) -> str:
    """Format project information into a displayable string.

    Args:
        info: ProjectInfo object to format

    Returns:
        Formatted string containing project status and file listing
    """
    # Create project status line
    status = "New/Empty Project" if info.is_new else "Existing Project"

    # Handle empty project case
    if info.total_files == 0:
        return f"Project Status: {status}\nTotal Files: 0\nFiles: None"

    # Format file count with truncation notice if needed
    file_count = (
        f"{len(info.files)} of {info.total_files}"
        if len(info.files) < info.total_files
        else str(info.total_files)
    )
    file_count_line = f"Total Files: {file_count}"

    # Format file listing
    files_section = "Files:\n" + "\n".join(f"- {f}" for f in info.files)

    # Add truncation notice if list was truncated
    if len(info.files) < info.total_files:
        files_section += (
            f"\n[Note: Showing {len(info.files)} of {info.total_files} total files]"
        )

    return f"Project Status: {status}\n{file_count_line}\n{files_section}"


def display_project_status(info: ProjectInfo) -> None:
    """Display project status in a visual panel.

    Args:
        info: ProjectInfo object containing project state
    """
    # Create project status text
    status = "**New/empty project**" if info.is_new else "**Existing project**"

    # Format file count (with truncation notice if needed)
    file_count = (
        f"{len(info.files)} of {info.total_files}"
        if len(info.files) < info.total_files
        else str(info.total_files)
    )

    # Build status text with markdown
    status_text = f"""
{status} with **{file_count} file(s)**
"""
    # Add truncation notice if list was truncated
    if len(info.files) < info.total_files:
        status_text += f"\n[*Note: File listing truncated ({len(info.files)} of {info.total_files} shown)*]"

    # Create and display panel
    console = Console()
    console.print(Panel(Markdown(status_text.strip()), title="📊 Project Status"))
