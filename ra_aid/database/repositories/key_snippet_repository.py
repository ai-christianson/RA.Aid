"""
Key snippet repository implementation for database access.

This module provides a repository implementation for the KeySnippet model,
following the repository pattern for data access abstraction.
"""

from typing import Dict, List, Optional, Any

import peewee

from ra_aid.database.connection import get_db
from ra_aid.database.models import KeySnippet, initialize_database
from ra_aid.logging_config import get_logger

logger = get_logger(__name__)


class KeySnippetRepository:
    """
    Repository for managing KeySnippet database operations.
    
    This class provides methods for performing CRUD operations on the KeySnippet model,
    abstracting the database access details from the business logic.
    
    Example:
        repo = KeySnippetRepository()
        snippet = repo.create(
            filepath="main.py", 
            line_number=42, 
            snippet="def hello_world():", 
            description="Main function definition"
        )
        all_snippets = repo.get_all()
    """
    
    def __init__(self, db=None):
        """
        Initialize the repository with an optional database connection.
        
        Args:
            db: Optional database connection to use. If None, will use initialize_database()
        """
        self.db = db
    
    def create(
        self, filepath: str, line_number: int, snippet: str, description: Optional[str] = None,
        human_input_id: Optional[int] = None
    ) -> KeySnippet:
        """
        Create a new key snippet in the database.
        
        Args:
            filepath: Path to the source file
            line_number: Line number where the snippet starts
            snippet: The source code snippet text
            description: Optional description of the significance
            human_input_id: Optional ID of the associated human input
            
        Returns:
            KeySnippet: The newly created key snippet instance
            
        Raises:
            peewee.DatabaseError: If there's an error creating the snippet
        """
        try:
            db = self.db if self.db is not None else initialize_database()
            key_snippet = KeySnippet.create(
                filepath=filepath,
                line_number=line_number,
                snippet=snippet,
                description=description,
                human_input_id=human_input_id
            )
            logger.debug(f"Created key snippet ID {key_snippet.id}: {filepath}:{line_number}")
            return key_snippet
        except peewee.DatabaseError as e:
            logger.error(f"Failed to create key snippet: {str(e)}")
            raise
    
    def get(self, snippet_id: int) -> Optional[KeySnippet]:
        """
        Retrieve a key snippet by its ID.
        
        Args:
            snippet_id: The ID of the key snippet to retrieve
            
        Returns:
            Optional[KeySnippet]: The key snippet instance if found, None otherwise
            
        Raises:
            peewee.DatabaseError: If there's an error accessing the database
        """
        try:
            db = self.db if self.db is not None else initialize_database()
            return KeySnippet.get_or_none(KeySnippet.id == snippet_id)
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
    ) -> Optional[KeySnippet]:
        """
        Update an existing key snippet.
        
        Args:
            snippet_id: The ID of the key snippet to update
            filepath: Path to the source file
            line_number: Line number where the snippet starts
            snippet: The source code snippet text
            description: Optional description of the significance
            
        Returns:
            Optional[KeySnippet]: The updated key snippet if found, None otherwise
            
        Raises:
            peewee.DatabaseError: If there's an error updating the snippet
        """
        try:
            db = self.db if self.db is not None else initialize_database()
            # First check if the snippet exists
            key_snippet = self.get(snippet_id)
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
            return key_snippet
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
            db = self.db if self.db is not None else initialize_database()
            # First check if the snippet exists
            key_snippet = self.get(snippet_id)
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
    
    def get_all(self) -> List[KeySnippet]:
        """
        Retrieve all key snippets from the database.
        
        Returns:
            List[KeySnippet]: List of all key snippet instances
            
        Raises:
            peewee.DatabaseError: If there's an error accessing the database
        """
        try:
            db = self.db if self.db is not None else initialize_database()
            return list(KeySnippet.select().order_by(KeySnippet.id))
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


# Global singleton instance
_key_snippet_repository = None


def get_key_snippet_repository() -> KeySnippetRepository:
    """
    Get or create a singleton instance of KeySnippetRepository.
    
    Returns:
        KeySnippetRepository: Singleton instance of the repository
    """
    global _key_snippet_repository
    if _key_snippet_repository is None:
        _key_snippet_repository = KeySnippetRepository()
    return _key_snippet_repository