"""Module for efficient file listing using git."""

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple


class FileListerError(Exception):
    """Base exception for file listing related errors."""

    pass


class GitCommandError(FileListerError):
    """Raised when a git command fails."""

    pass


class DirectoryNotFoundError(FileListerError):
    """Raised when the specified directory does not exist."""

    pass


class DirectoryAccessError(FileListerError):
    """Raised when the directory cannot be accessed due to permissions."""

    pass


def is_git_repo(directory: str) -> bool:
    """
    Check if the given directory is a git repository.

    Args:
        directory: Path to the directory to check

    Returns:
        bool: True if directory is a git repository, False otherwise

    Raises:
        DirectoryNotFoundError: If directory does not exist
        DirectoryAccessError: If directory cannot be accessed
        GitCommandError: If git command fails unexpectedly
    """
    try:
        path = Path(directory).resolve()
        if not path.exists():
            raise DirectoryNotFoundError(f"Directory does not exist: {directory}")
        if not path.is_dir():
            raise DirectoryNotFoundError(f"Path is not a directory: {directory}")

        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=str(path),
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    except subprocess.CalledProcessError as e:
        raise GitCommandError(f"Git command failed: {e}")
    except PermissionError as e:
        raise DirectoryAccessError(f"Cannot access directory {directory}: {e}")
    except Exception as e:
        if isinstance(e, FileListerError):
            raise
        raise FileListerError(f"Error checking git repository: {e}")


def get_file_listing(
    directory: str, limit: Optional[int] = None
) -> Tuple[List[str], int]:
    """
    Get a list of tracked files in a git repository.

    Uses `git ls-files` for efficient file listing that respects .gitignore rules.
    Returns a tuple containing the list of files (truncated if limit is specified)
    and the total count of files.

    Args:
        directory: Path to the git repository
        limit: Optional maximum number of files to return

    Returns:
        Tuple[List[str], int]: Tuple containing:
            - List of file paths (truncated to limit if specified)
            - Total number of files (before truncation)

    Raises:
        DirectoryNotFoundError: If directory does not exist
        DirectoryAccessError: If directory cannot be accessed
        GitCommandError: If git command fails
        FileListerError: For other unexpected errors
    """
    try:
        # Check if directory is a git repo first
        if not is_git_repo(directory):
            return [], 0

        # Run git ls-files
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=directory,
            capture_output=True,
            text=True,
            check=True,
        )

        # Process the output
        files = [line.strip() for line in result.stdout.splitlines() if line.strip()]

        # Deduplicate and sort for consistency
        files = list(dict.fromkeys(files))  # Remove duplicates while preserving order

        # Sort for consistency
        files.sort()

        # Get total count before truncation
        total_count = len(files)

        # Truncate if limit specified
        if limit is not None:
            files = files[:limit]

        return files, total_count

    except subprocess.CalledProcessError as e:
        raise GitCommandError(f"Git command failed: {e}")
    except PermissionError as e:
        raise DirectoryAccessError(f"Cannot access directory {directory}: {e}")
    except Exception as e:
        if isinstance(e, FileListerError):
            raise
        raise FileListerError(f"Error listing files: {e}")
