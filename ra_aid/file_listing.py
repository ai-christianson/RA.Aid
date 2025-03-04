"""Module for efficient file listing using git."""

import os
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
import fnmatch


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


def get_all_project_files(
    directory: str, include_hidden: bool = False, exclude_patterns: Optional[List[str]] = None
) -> List[str]:
    """
    Get a list of all files in a project directory, handling both git and non-git repositories.
    
    Args:
        directory: Path to the directory
        include_hidden: Whether to include hidden files (starting with .) in the results
        exclude_patterns: Optional list of patterns to exclude from the results
        
    Returns:
        List[str]: List of file paths relative to the directory
        
    Raises:
        DirectoryNotFoundError: If directory does not exist
        DirectoryAccessError: If directory cannot be accessed
        GitCommandError: If git command fails
        FileListerError: For other unexpected errors
    """
    # Check if directory exists and is accessible
    if not os.path.exists(directory):
        raise DirectoryNotFoundError(f"Directory not found: {directory}")
    if not os.path.isdir(directory):
        raise DirectoryNotFoundError(f"Not a directory: {directory}")
    
    # Default excluded directories
    excluded_dirs = {'.ra-aid', '.venv', '.git', '.aider', '__pycache__'}
    
    # Check if it's a git repository
    try:
        is_git = is_git_repo(directory)
    except FileListerError:
        # If checking fails, default to non-git approach
        is_git = False
    
    all_files = []
    
    if is_git:
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
        for file in (
            tracked_files_process.stdout.splitlines()
            + untracked_files_process.stdout.splitlines()
        ):
            file = file.strip()
            if not file:
                continue
            # Skip hidden files unless explicitly included
            if not include_hidden and (
                file.startswith(".")
                or any(part.startswith(".") for part in file.split("/"))
            ):
                continue
            # Skip .aider files
            if ".aider" in file:
                continue
            all_files.append(file)
    else:
        # Not a git repository, use manual file listing
        base_path = Path(directory)
        
        # First check if we can access the directory (check exists and isdir already done above)
        try:
            # We already verified existence, just check for permission errors
            # Handle potential FileNotFoundError for mock tests
            try:
                os.listdir(directory)
            except FileNotFoundError:
                # This should normally not happen as we checked existence above
                # But it can happen in mock tests
                pass
        except PermissionError as e:
            raise DirectoryAccessError(f"Cannot access directory {directory}: {e}")
        
        try:
            for root, dirs, files in os.walk(directory):
                # Filter out excluded directories
                dirs[:] = [d for d in dirs if d not in excluded_dirs and (include_hidden or not d.startswith('.'))]
                
                # Calculate relative path
                rel_root = os.path.relpath(root, directory)
                if rel_root == '.':
                    rel_root = ''
                
                # Process files
                for file in files:
                    # Skip hidden files unless explicitly included
                    if not include_hidden and file.startswith('.'):
                        continue
                    
                    # Create relative path
                    rel_path = os.path.join(rel_root, file) if rel_root else file
                    all_files.append(rel_path)
        except PermissionError as e:
            raise DirectoryAccessError(f"Permission denied while walking directory {directory}: {e}")
    
    # Apply additional exclude patterns if specified
    if exclude_patterns:
        for pattern in exclude_patterns:
            all_files = [f for f in all_files if not fnmatch.fnmatch(f, pattern)]
            
    # Remove duplicates and sort
    return sorted(set(all_files))


def get_file_listing(
    directory: str, limit: Optional[int] = None, include_hidden: bool = False
) -> Tuple[List[str], int]:
    """
    Get a list of files in a directory.

    For git repositories, uses `git ls-files` for efficient file listing that respects .gitignore rules.
    For non-git directories, falls back to manual file listing using Python's standard library.
    Returns a tuple containing the list of files (truncated if limit is specified)
    and the total count of files.

    Args:
        directory: Path to the directory
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
        # Use the common function to get all files
        all_files = get_all_project_files(directory, include_hidden)
        
        # Get total count before truncation
        total_count = len(all_files)

        # Apply limit if specified
        if limit is not None:
            all_files = all_files[:limit]

        return all_files, total_count

    except (DirectoryNotFoundError, DirectoryAccessError, GitCommandError):
        # Re-raise known exceptions
        raise
    except PermissionError as e:
        raise DirectoryAccessError(f"Permission denied: {e}")
    except Exception as e:
        raise FileListerError(f"Unexpected error: {e}")
