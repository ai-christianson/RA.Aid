"""
Key snippet repository implementation for database access.

This module provides a repository implementation for the KeySnippet model,
following the repository pattern for data access abstraction.
"""

from typing import Dict, List, Optional, Any
import contextvars

import peewee

from ra_aid.database.models import KeySnippet
from ra_aid.database.pydantic_models import KeySnippetModel
from ra_aid.logging_config import get_logger

logger = get_logger(__name__)

# Create contextvar to hold the KeySnippetRepository instance
key_snippet_repo_var = contextvars.ContextVar("key_snippet_repo", default=None)


class KeySnippetRepositoryManager:
    """
    Context manager for KeySnippetRepository.

    This class provides a context manager interface for KeySnippetRepository,
    using the contextvars approach for thread safety.

    Example:
        with DatabaseManager() as db:
            with KeySnippetRepositoryManager(db) as repo:
                # Use the repository
                snippet = repo.create(
                    filepath="main.py", 
                    line_number=42, 
                    snippet="def hello_world():", 
                    description="Main function definition"
                )
                all_snippets = repo.get_all()
    """

    def __init__(self, db):
        """
        Initialize the KeySnippetRepositoryManager.

        Args:
            db: Database connection to use (required)
        """
        self.db = db

    def __enter__(self) -> 'KeySnippetRepository':
        """
        Initialize the KeySnippetRepository and return it.

        Returns:
            KeySnippetRepository: The initialized repository
        """
        repo = KeySnippetRepository(self.db)
        key_snippet_repo_var.set(repo)
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
        key_snippet_repo_var.set(None)

        # Don't suppress exceptions
        return False


def get_key_snippet_repository() -> 'KeySnippetRepository':
    """
    Get the current KeySnippetRepository instance.

    Returns:
        KeySnippetRepository: The current repository instance
        
    Raises:
        RuntimeError: If no repository has been initialized with KeySnippetRepositoryManager
    """
    repo = key_snippet_repo_var.get()
    if repo is None:
        raise RuntimeError(
            "No KeySnippetRepository available. "
            "Make sure to initialize one with KeySnippetRepositoryManager first."
        )
    return repo


class KeySnippetRepository:
    """
    Repository for managing KeySnippet database operations.
    
    This class provides methods for performing CRUD operations on the KeySnippet model,
    abstracting the database access details from the business logic.
    
    Example:
        with DatabaseManager() as db:
            with KeySnippetRepositoryManager(db) as repo:
                snippet = repo.create(
                    filepath="main.py", 
                    line_number=42, 
                    snippet="def hello_world():", 
                    description="Main function definition"
                )
                all_snippets = repo.get_all()
    """
    
    def __init__(self, db):
        """
        Initialize the repository with a database connection.
        
        Args:
            db: Database connection to use (required)
        """
        if db is None:
            raise ValueError("Database connection is required for KeySnippetRepository")
        self.db = db
    
    def _to_model(self, snippet: Optional[KeySnippet]) -> Optional[KeySnippetModel]:
        """
        Convert a Peewee KeySnippet object to a Pydantic KeySnippetModel.
        
        Args:
            snippet: Peewee KeySnippet instance or None
            
        Returns:
            Optional[KeySnippetModel]: Pydantic model representation or None if snippet is None
        """
        if snippet is None:
            return None
        
        return KeySnippetModel.model_validate(snippet, from_attributes=True)
    
    def create(
        self, filepath: str, line_number: int, snippet: str, description: Optional[str] = None,
        human_input_id: Optional[int] = None
    ) -> KeySnippetModel:
        """
        Create a new key snippet in the database.
        
        Args:
            filepath: Path to the source file
            line_number: Line number where the snippet starts
            snippet: The source code snippet text
            description: Optional description of the significance
            human_input_id: Optional ID of the associated human input
            
        Returns:
            KeySnippetModel: The newly created key snippet instance
            
        Raises:
            peewee.DatabaseError: If there's an error creating the snippet
        """
        try:
            key_snippet = KeySnippet.create(
                filepath=filepath,
                line_number=line_number,
                snippet=snippet,
                description=description,
                human_input_id=human_input_id
            )
            logger.debug(f"Created key snippet ID {key_snippet.id}: {filepath}:{line_number}")
            return self._to_model(key_snippet)
        except peewee.DatabaseError as e:
            logger.error(f"Failed to create key snippet: {str(e)}")
            raise
    
    def get(self, snippet_id: int) -> Optional[KeySnippetModel]:
        """
        Retrieve a key snippet by its ID.
        
        Args:
            snippet_id: The ID of the key snippet to retrieve
            
        Returns:
            Optional[KeySnippetModel]: The key snippet instance if found, None otherwise
            
        Raises:
            peewee.DatabaseError: If there's an error accessing the database
        """
        try:
            snippet = KeySnippet.get_or_none(KeySnippet.id == snippet_id)
            return self._to_model(snippet)
        except peewee.DatabaseError as e:
            logger.error(f"Failed to fetch key snippet {snippet_id}: {str(e)}")
            raise
    
    def update(
        self, 
        snippet_id: int, 
        filepath: str, 
        line_number: int, 
        snippet: str, 
        description: Optional[str] = None
    ) -> Optional[KeySnippetModel]:
        """
        Update an existing key snippet.
        
        Args:
            snippet_id: The ID of the key snippet to update
            filepath: Path to the source file
            line_number: Line number where the snippet starts
            snippet: The source code snippet text
            description: Optional description of the significance
            
        Returns:
            Optional[KeySnippetModel]: The updated key snippet if found, None otherwise
            
        Raises:
            peewee.DatabaseError: If there's an error updating the snippet
        """
        try:
            # First check if the snippet exists
            key_snippet = KeySnippet.get_or_none(KeySnippet.id == snippet_id)
            if not key_snippet:
                logger.warning(f"Attempted to update non-existent key snippet {snippet_id}")
                return None
            
            # Update the snippet
            key_snippet.filepath = filepath
            key_snippet.line_number = line_number
            key_snippet.snippet = snippet
            key_snippet.description = description
            key_snippet.save()
            logger.debug(f"Updated key snippet ID {snippet_id}: {filepath}:{line_number}")
            return self._to_model(key_snippet)
        except peewee.DatabaseError as e:
            logger.error(f"Failed to update key snippet {snippet_id}: {str(e)}")
            raise
    
    def delete(self, snippet_id: int) -> bool:
        """
        Delete a key snippet by its ID.
        
        Args:
            snippet_id: The ID of the key snippet to delete
            
        Returns:
            bool: True if the snippet was deleted, False if it wasn't found
            
        Raises:
            peewee.DatabaseError: If there's an error deleting the snippet
        """
        try:
            # First check if the snippet exists
            key_snippet = KeySnippet.get_or_none(KeySnippet.id == snippet_id)
            if not key_snippet:
                logger.warning(f"Attempted to delete non-existent key snippet {snippet_id}")
                return False
            
            # Delete the snippet
            key_snippet.delete_instance()
            logger.debug(f"Deleted key snippet ID {snippet_id}")
            return True
        except peewee.DatabaseError as e:
            logger.error(f"Failed to delete key snippet {snippet_id}: {str(e)}")
            raise
    
    def get_all(self) -> List[KeySnippetModel]:
        """
        Retrieve all key snippets from the database.
        
        Returns:
            List[KeySnippetModel]: List of all key snippet instances
            
        Raises:
            peewee.DatabaseError: If there's an error accessing the database
        """
        try:
            snippets = list(KeySnippet.select().order_by(KeySnippet.id))
            return [self._to_model(snippet) for snippet in snippets]
        except peewee.DatabaseError as e:
            logger.error(f"Failed to fetch all key snippets: {str(e)}")
            raise
    
    def get_snippets_dict(self) -> Dict[int, Dict[str, Any]]:
        """
        Retrieve all key snippets as a dictionary mapping IDs to snippet information.
        
        This method is useful for compatibility with the existing memory format.
        
        Returns:
            Dict[int, Dict[str, Any]]: Dictionary with snippet IDs as keys and 
                                       snippet information as values
            
        Raises:
            peewee.DatabaseError: If there's an error accessing the database
        """
        try:
            snippets = self.get_all()
            return {
                snippet.id: {
                    "filepath": snippet.filepath,
                    "line_number": snippet.line_number,
                    "snippet": snippet.snippet,
                    "description": snippet.description
                } 
                for snippet in snippets
            }
        except peewee.DatabaseError as e:
            logger.error(f"Failed to fetch key snippets as dictionary: {str(e)}")
            raise