import contextvars
import os
from typing import Dict, List, Optional

# Import is_binary_file from memory.py
from ra_aid.utils.file_utils import is_binary_file

# Create contextvar to hold the RelatedFilesRepository instance
related_files_repo_var = contextvars.ContextVar("related_files_repo", default=None)


class RelatedFilesRepository:
    """
    Repository for managing related files in memory.
    
    This class provides methods to add, remove, and retrieve related files.
    It does not require database models and operates entirely in memory.
    """
    
    def __init__(self):
        """
        Initialize the RelatedFilesRepository.
        """
        self._related_files: Dict[int, str] = {}
        self._id_counter: int = 1
    
    def get_all(self) -> Dict[int, str]:
        """
        Get all related files.
        
        Returns:
            Dict[int, str]: Dictionary mapping file IDs to file paths
        """
        return self._related_files.copy()
    
    def add_file(self, filepath: str) -> Optional[int]:
        """
        Add a file to the repository.
        
        Args:
            filepath: Path to the file to add
            
        Returns:
            Optional[int]: The ID assigned to the file, or None if the file could not be added
        """
        # First check if path exists
        if not os.path.exists(filepath):
            return None
            
        # Then check if it's a directory
        if os.path.isdir(filepath):
            return None
            
        # Validate it's a regular file
        if not os.path.isfile(filepath):
            return None
            
        # Check if it's a binary file
        if is_binary_file(filepath):
            return None
            
        # Normalize the path
        normalized_path = os.path.abspath(filepath)
        
        # Check if normalized path already exists in values
        for file_id, path in self._related_files.items():
            if path == normalized_path:
                return file_id
                
        # Add new file
        file_id = self._id_counter
        self._id_counter += 1
        self._related_files[file_id] = normalized_path
        
        return file_id
    
    def remove_file(self, file_id: int) -> Optional[str]:
        """
        Remove a file from the repository.
        
        Args:
            file_id: ID of the file to remove
            
        Returns:
            Optional[str]: The path of the removed file, or None if the file ID was not found
        """
        if file_id in self._related_files:
            return self._related_files.pop(file_id)
        return None
    
    def format_related_files(self) -> List[str]:
        """
        Format related files as 'ID#X path/to/file'.
        
        Returns:
            List[str]: Formatted strings for each related file
        """
        return [f"ID#{file_id} {filepath}" for file_id, filepath in sorted(self._related_files.items())]
        
    def get_next_id(self) -> int:
        """
        Get the next ID that would be assigned to a new file.
        
        Returns:
            int: The next ID value
        """
        return self._id_counter


class RelatedFilesRepositoryManager:
    """
    Context manager for RelatedFilesRepository.
    
    This class provides a context manager interface for RelatedFilesRepository,
    using the contextvars approach for thread safety.
    
    Example:
        with RelatedFilesRepositoryManager() as repo:
            # Use the repository
            file_id = repo.add_file("path/to/file.py")
            all_files = repo.get_all()
    """
    
    def __init__(self):
        """
        Initialize the RelatedFilesRepositoryManager.
        """
        pass
        
    def __enter__(self) -> 'RelatedFilesRepository':
        """
        Initialize the RelatedFilesRepository and return it.
        
        Returns:
            RelatedFilesRepository: The initialized repository
        """
        repo = RelatedFilesRepository()
        related_files_repo_var.set(repo)
        return repo
        
    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[object],
    ) -> None:
        """
        Reset the repository when exiting the context.
        
        Args:
            exc_type: The exception type if an exception was raised
            exc_val: The exception value if an exception was raised
            exc_tb: The traceback if an exception was raised
        """
        # Reset the contextvar to None
        related_files_repo_var.set(None)
        
        # Don't suppress exceptions
        return False


def get_related_files_repository() -> RelatedFilesRepository:
    """
    Get the current RelatedFilesRepository instance.
    
    Returns:
        RelatedFilesRepository: The current repository instance
        
    Raises:
        RuntimeError: If no repository is set in the current context
    """
    repo = related_files_repo_var.get()
    if repo is None:
        raise RuntimeError(
            "RelatedFilesRepository not initialized in current context. "
            "Make sure to use RelatedFilesRepositoryManager."
        )
    return repo