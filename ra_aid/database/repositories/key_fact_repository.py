"""
Key fact repository implementation for database access.

This module provides a repository implementation for the KeyFact model,
following the repository pattern for data access abstraction.
"""

from typing import Dict, List, Optional
import contextvars
from contextlib import contextmanager

import peewee

from ra_aid.database.models import KeyFact
from ra_aid.logging_config import get_logger

logger = get_logger(__name__)

# Create contextvar to hold the KeyFactRepository instance
key_fact_repo_var = contextvars.ContextVar("key_fact_repo", default=None)


class KeyFactRepositoryManager:
    """
    Context manager for KeyFactRepository.

    This class provides a context manager interface for KeyFactRepository,
    using the contextvars approach for thread safety.

    Example:
        with DatabaseManager() as db:
            with KeyFactRepositoryManager(db) as repo:
                # Use the repository
                fact = repo.create("Important fact about the project")
                all_facts = repo.get_all()
    """

    def __init__(self, db):
        """
        Initialize the KeyFactRepositoryManager.

        Args:
            db: Database connection to use (required)
        """
        self.db = db

    def __enter__(self) -> 'KeyFactRepository':
        """
        Initialize the KeyFactRepository and return it.

        Returns:
            KeyFactRepository: The initialized repository
        """
        repo = KeyFactRepository(self.db)
        key_fact_repo_var.set(repo)
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
        key_fact_repo_var.set(None)

        # Don't suppress exceptions
        return False


def get_key_fact_repository() -> 'KeyFactRepository':
    """
    Get the current KeyFactRepository instance.

    Returns:
        KeyFactRepository: The current repository instance
        
    Raises:
        RuntimeError: If no repository has been initialized with KeyFactRepositoryManager
    """
    repo = key_fact_repo_var.get()
    if repo is None:
        raise RuntimeError(
            "No KeyFactRepository available. "
            "Make sure to initialize one with KeyFactRepositoryManager first."
        )
    return repo


class KeyFactRepository:
    """
    Repository for managing KeyFact database operations.
    
    This class provides methods for performing CRUD operations on the KeyFact model,
    abstracting the database access details from the business logic.
    
    Example:
        with DatabaseManager() as db:
            with KeyFactRepositoryManager(db) as repo:
                fact = repo.create("Important fact about the project")
                all_facts = repo.get_all()
    """
    
    def __init__(self, db):
        """
        Initialize the repository with a database connection.
        
        Args:
            db: Database connection to use (required)
        """
        if db is None:
            raise ValueError("Database connection is required for KeyFactRepository")
        self.db = db
    
    def create(self, content: str, human_input_id: Optional[int] = None) -> KeyFact:
        """
        Create a new key fact in the database.
        
        Args:
            content: The text content of the key fact
            human_input_id: Optional ID of the associated human input
            
        Returns:
            KeyFact: The newly created key fact instance
            
        Raises:
            peewee.DatabaseError: If there's an error creating the fact
        """
        try:
            fact = KeyFact.create(content=content, human_input_id=human_input_id)
            logger.debug(f"Created key fact ID {fact.id}: {content}")
            return fact
        except peewee.DatabaseError as e:
            logger.error(f"Failed to create key fact: {str(e)}")
            raise
    
    def get(self, fact_id: int) -> Optional[KeyFact]:
        """
        Retrieve a key fact by its ID.
        
        Args:
            fact_id: The ID of the key fact to retrieve
            
        Returns:
            Optional[KeyFact]: The key fact instance if found, None otherwise
            
        Raises:
            peewee.DatabaseError: If there's an error accessing the database
        """
        try:
            return KeyFact.get_or_none(KeyFact.id == fact_id)
        except peewee.DatabaseError as e:
            logger.error(f"Failed to fetch key fact {fact_id}: {str(e)}")
            raise
    
    def update(self, fact_id: int, content: str) -> Optional[KeyFact]:
        """
        Update an existing key fact.
        
        Args:
            fact_id: The ID of the key fact to update
            content: The new content for the key fact
            
        Returns:
            Optional[KeyFact]: The updated key fact if found, None otherwise
            
        Raises:
            peewee.DatabaseError: If there's an error updating the fact
        """
        try:
            # First check if the fact exists
            fact = self.get(fact_id)
            if not fact:
                logger.warning(f"Attempted to update non-existent key fact {fact_id}")
                return None
            
            # Update the fact
            fact.content = content
            fact.save()
            logger.debug(f"Updated key fact ID {fact_id}: {content}")
            return fact
        except peewee.DatabaseError as e:
            logger.error(f"Failed to update key fact {fact_id}: {str(e)}")
            raise
    
    def delete(self, fact_id: int) -> bool:
        """
        Delete a key fact by its ID.
        
        Args:
            fact_id: The ID of the key fact to delete
            
        Returns:
            bool: True if the fact was deleted, False if it wasn't found
            
        Raises:
            peewee.DatabaseError: If there's an error deleting the fact
        """
        try:
            # First check if the fact exists
            fact = self.get(fact_id)
            if not fact:
                logger.warning(f"Attempted to delete non-existent key fact {fact_id}")
                return False
            
            # Delete the fact
            fact.delete_instance()
            logger.debug(f"Deleted key fact ID {fact_id}")
            return True
        except peewee.DatabaseError as e:
            logger.error(f"Failed to delete key fact {fact_id}: {str(e)}")
            raise
    
    def get_all(self) -> List[KeyFact]:
        """
        Retrieve all key facts from the database.
        
        Returns:
            List[KeyFact]: List of all key fact instances
            
        Raises:
            peewee.DatabaseError: If there's an error accessing the database
        """
        try:
            return list(KeyFact.select().order_by(KeyFact.id))
        except peewee.DatabaseError as e:
            logger.error(f"Failed to fetch all key facts: {str(e)}")
            raise
    
    def get_facts_dict(self) -> Dict[int, str]:
        """
        Retrieve all key facts as a dictionary mapping IDs to content.
        
        This method is useful for compatibility with the existing memory format.
        
        Returns:
            Dict[int, str]: Dictionary with fact IDs as keys and content as values
            
        Raises:
            peewee.DatabaseError: If there's an error accessing the database
        """
        try:
            facts = self.get_all()
            return {fact.id: fact.content for fact in facts}
        except peewee.DatabaseError as e:
            logger.error(f"Failed to fetch key facts as dictionary: {str(e)}")
            raise