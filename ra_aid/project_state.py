"""Module for determining project state and initialization status."""

from pathlib import Path
from typing import Set


class ProjectStateError(Exception):
    """Base exception for project state related errors."""

    pass


class DirectoryNotFoundError(ProjectStateError):
    """Raised when the specified directory does not exist."""

    pass


class DirectoryAccessError(ProjectStateError):
    """Raised when the directory cannot be accessed due to permissions."""

    pass


def is_new_project(directory: str) -> bool:
    """
    Determine if a directory represents a new/empty project.

    A project is considered new if it either:
    - Is an empty directory
    - Contains only .git directory and/or .gitignore file

    Args:
        directory: String path to the directory to check

    Returns:
        bool: True if the directory is empty or contains only git files,
              False otherwise

    Raises:
        DirectoryNotFoundError: If the specified directory does not exist
        DirectoryAccessError: If the directory cannot be accessed
        ProjectStateError: For other unexpected errors
    """
    try:
        path = Path(directory).resolve()
        if not path.exists():
            raise DirectoryNotFoundError(f"Directory does not exist: {directory}")
        if not path.is_dir():
            raise DirectoryNotFoundError(f"Path is not a directory: {directory}")

        # Get all files/dirs in the directory, excluding contents of .git
        _allowed_items: Set[str] = {".git", ".gitignore"}
        try:
            contents = set()
            for item in path.iterdir():
                # Only consider top-level items
                if item.name != ".git":
                    contents.add(item.name)
        except PermissionError as e:
            raise DirectoryAccessError(f"Cannot access directory {directory}: {e}")

        # Directory is new if empty or only contains .gitignore
        return len(contents) == 0 or contents.issubset({".gitignore"})

    except Exception as e:
        if isinstance(e, ProjectStateError):
            raise
        raise ProjectStateError(f"Error checking project state: {e}")
