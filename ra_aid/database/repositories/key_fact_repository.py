"""
Key fact repository implementation for database access.

This module provides a repository implementation for the KeyFact model,
following the repository pattern for data access abstraction.
"""

from typing import Dict, List, Optional

import peewee

from ra_aid.database.connection import get_db
from ra_aid.database.models import KeyFact
from ra_aid.logging_config import get_logger

logger = get_logger(__name__)


class KeyFactRepository:
    """
    Repository for managing KeyFact database operations.
    
    This class provides methods for performing CRUD operations on the KeyFact model,
    abstracting the database access details from the business logic.
    
    Example:
        repo = KeyFactRepository()
        fact = repo.create("Important fact about the project")
        all_facts = repo.get_all()
    """
    
    def create(self, content: str) -> KeyFact:
        """
        Create a new key fact in the database.
        
        Args:
            content: The text content of the key fact
            
        Returns:
            KeyFact: The newly created key fact instance
            
        Raises:
            peewee.DatabaseError: If there's an error creating the fact
        """
        try:
            db = get_db()
            fact = KeyFact.create(content=content)
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
            db = get_db()
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
            db = get_db()
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
            db = get_db()
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
            db = get_db()
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