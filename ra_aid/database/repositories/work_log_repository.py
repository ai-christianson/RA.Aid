import contextvars
from datetime import datetime
from typing import Dict, List, Optional, TypedDict

# Define WorkLogEntry TypedDict
class WorkLogEntry(TypedDict):
    timestamp: str
    event: str

# Create contextvar to hold the WorkLogRepository instance
work_log_repo_var = contextvars.ContextVar("work_log_repo", default=None)


class WorkLogRepository:
    """
    Repository for managing work log entries in memory.
    
    This class provides methods to add, retrieve, and clear work log entries.
    It does not require database models and operates entirely in memory.
    """
    
    def __init__(self):
        """
        Initialize an empty work log.
        """
        self._entries: List[WorkLogEntry] = []
        
    def add_entry(self, event: str) -> None:
        """
        Add a new work log entry with timestamp.
        
        Args:
            event: Description of the event to log
        """
        entry = WorkLogEntry(timestamp=datetime.now().isoformat(), event=event)
        self._entries.append(entry)
        
    def get_all(self) -> List[WorkLogEntry]:
        """
        Get all work log entries.
        
        Returns:
            List of WorkLogEntry objects
        """
        return self._entries.copy()
        
    def clear(self) -> None:
        """
        Clear all work log entries.
        """
        self._entries.clear()
        
    def format_work_log(self) -> str:
        """
        Format work log entries as markdown.
        
        Returns:
            Markdown formatted text with timestamps as headings and events as content,
            or 'No work log entries' if the log is empty.
            
        Example:
            ## 2024-12-23T11:39:10
            
            Task #1 added: Create login form
        """
        if not self._entries:
            return "No work log entries"
            
        entries = []
        for entry in self._entries:
            entries.extend([
                f"## {entry['timestamp']}",
                "",
                entry['event'],
                "",  # Blank line between entries
            ])
            
        return "\n".join(entries).rstrip()  # Remove trailing newline


class WorkLogRepositoryManager:
    """
    Context manager for WorkLogRepository.
    
    This class provides a context manager interface for WorkLogRepository,
    using the contextvars approach for thread safety.
    
    Example:
        with WorkLogRepositoryManager() as repo:
            # Use the repository
            repo.add_entry("Task #1 added: Create login form")
            log_text = repo.format_work_log()
    """
    
    def __init__(self):
        """
        Initialize the WorkLogRepositoryManager.
        """
        pass
        
    def __enter__(self) -> 'WorkLogRepository':
        """
        Initialize the WorkLogRepository and return it.
        
        Returns:
            WorkLogRepository: The initialized repository
        """
        repo = WorkLogRepository()
        work_log_repo_var.set(repo)
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
        work_log_repo_var.set(None)
        
        # Don't suppress exceptions
        return False


def get_work_log_repository() -> WorkLogRepository:
    """
    Get the current WorkLogRepository instance.
    
    Returns:
        WorkLogRepository: The current repository instance
        
    Raises:
        RuntimeError: If no repository is set in the current context
    """
    repo = work_log_repo_var.get()
    if repo is None:
        raise RuntimeError(
            "WorkLogRepository not initialized in current context. "
            "Make sure to use WorkLogRepositoryManager."
        )
    return repo