"""Module for efficient file listing using git."""

import subprocess
import os
from pathlib import Path
from typing import List, Optional, Tuple
import tempfile
import shutil


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
    directory: str, limit: Optional[int] = None, include_hidden: bool = False
) -> Tuple[List[str], int]:
    """
    Get a list of tracked files in a git repository.

    Uses `git ls-files` for efficient file listing that respects .gitignore rules.
    Returns a tuple containing the list of files (truncated if limit is specified)
    and the total count of files.

    Args:
        directory: Path to the git repository
        limit: Optional maximum number of files to return
        include_hidden: Whether to include hidden files (starting with .) in the results

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
        # Check if directory exists and is accessible
        if not os.path.exists(directory):
            raise DirectoryNotFoundError(f"Directory not found: {directory}")
        if not os.path.isdir(directory):
            raise DirectoryNotFoundError(f"Not a directory: {directory}")

        # Check if it's a git repository
        if not is_git_repo(directory):
            return [], 0

        # Get list of files from git ls-files
        try:
            # Get both tracked and untracked files
            tracked_files_process = subprocess.run(
                ["git", "ls-files"],
                cwd=directory,
                capture_output=True,
                text=True,
                check=True,
            )
            untracked_files_process = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                cwd=directory,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise GitCommandError(f"Git command failed: {e}")
        except PermissionError as e:
            raise DirectoryAccessError(f"Permission denied: {e}")

        # Combine and process the files
        all_files = []
        for file in tracked_files_process.stdout.splitlines() + untracked_files_process.stdout.splitlines():
            file = file.strip()
            if not file:
                continue
            # Skip hidden files unless explicitly included
            if not include_hidden and (file.startswith(".") or any(part.startswith(".") for part in file.split("/"))):
                continue
            # Skip .aider files
            if ".aider" in file:
                continue
            all_files.append(file)

        # Remove duplicates and sort
        all_files = sorted(set(all_files))
        total_count = len(all_files)

        # Apply limit if specified
        if limit is not None:
            all_files = all_files[:limit]

        return all_files, total_count

    except (DirectoryNotFoundError, DirectoryAccessError, GitCommandError) as e:
        # Re-raise known exceptions
        raise
    except PermissionError as e:
        raise DirectoryAccessError(f"Permission denied: {e}")
    except Exception as e:
        raise FileListerError(f"Unexpected error: {e}")
