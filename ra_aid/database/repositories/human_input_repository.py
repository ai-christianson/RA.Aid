"""
Human input repository implementation for database access.

This module provides a repository implementation for the HumanInput model,
following the repository pattern for data access abstraction.
"""

from typing import Dict, List, Optional

import peewee

from ra_aid.database.connection import get_db
from ra_aid.database.models import HumanInput, initialize_database
from ra_aid.logging_config import get_logger

logger = get_logger(__name__)


class HumanInputRepository:
    """
    Repository for managing HumanInput database operations.
    
    This class provides methods for performing CRUD operations on the HumanInput model,
    abstracting the database access details from the business logic.
    
    Example:
        repo = HumanInputRepository()
        input = repo.create("User's message", "chat")
        recent_inputs = repo.get_recent(5)
    """
    
    def __init__(self, db=None):
        """
        Initialize the repository with an optional database connection.
        
        Args:
            db: Optional database connection to use. If None, will use initialize_database()
        """
        self.db = db
    
    def create(self, content: str, source: str) -> HumanInput:
        """
        Create a new human input record in the database.
        
        Args:
            content: The text content of the human input
            source: The source of the input (e.g., "cli", "chat", "hil")
            
        Returns:
            HumanInput: The newly created human input instance
            
        Raises:
            peewee.DatabaseError: If there's an error creating the record
        """
        try:
            db = self.db if self.db is not None else initialize_database()
            input_record = HumanInput.create(content=content, source=source)
            logger.debug(f"Created human input ID {input_record.id} from {source}")
            return input_record
        except peewee.DatabaseError as e:
            logger.error(f"Failed to create human input record: {str(e)}")
            raise
    
    def get(self, input_id: int) -> Optional[HumanInput]:
        """
        Retrieve a human input record by its ID.
        
        Args:
            input_id: The ID of the human input to retrieve
            
        Returns:
            Optional[HumanInput]: The human input instance if found, None otherwise
            
        Raises:
            peewee.DatabaseError: If there's an error accessing the database
        """
        try:
            db = self.db if self.db is not None else initialize_database()
            return HumanInput.get_or_none(HumanInput.id == input_id)
        except peewee.DatabaseError as e:
            logger.error(f"Failed to fetch human input {input_id}: {str(e)}")
            raise
    
    def update(self, input_id: int, content: str = None, source: str = None) -> Optional[HumanInput]:
        """
        Update an existing human input record.
        
        Args:
            input_id: The ID of the human input to update
            content: The new content for the human input
            source: The new source for the human input
            
        Returns:
            Optional[HumanInput]: The updated human input if found, None otherwise
            
        Raises:
            peewee.DatabaseError: If there's an error updating the record
        """
        try:
            db = self.db if self.db is not None else initialize_database()
            # First check if the record exists
            input_record = self.get(input_id)
            if not input_record:
                logger.warning(f"Attempted to update non-existent human input {input_id}")
                return None
            
            # Update the fields that were provided
            if content is not None:
                input_record.content = content
            if source is not None:
                input_record.source = source
                
            input_record.save()
            logger.debug(f"Updated human input ID {input_id}")
            return input_record
        except peewee.DatabaseError as e:
            logger.error(f"Failed to update human input {input_id}: {str(e)}")
            raise
    
    def delete(self, input_id: int) -> bool:
        """
        Delete a human input record by its ID.
        
        Args:
            input_id: The ID of the human input to delete
            
        Returns:
            bool: True if the record was deleted, False if it wasn't found
            
        Raises:
            peewee.DatabaseError: If there's an error deleting the record
        """
        try:
            db = self.db if self.db is not None else initialize_database()
            # First check if the record exists
            input_record = self.get(input_id)
            if not input_record:
                logger.warning(f"Attempted to delete non-existent human input {input_id}")
                return False
            
            # Delete the record
            input_record.delete_instance()
            logger.debug(f"Deleted human input ID {input_id}")
            return True
        except peewee.DatabaseError as e:
            logger.error(f"Failed to delete human input {input_id}: {str(e)}")
            raise
    
    def get_all(self) -> List[HumanInput]:
        """
        Retrieve all human input records from the database.
        
        Returns:
            List[HumanInput]: List of all human input instances
            
        Raises:
            peewee.DatabaseError: If there's an error accessing the database
        """
        try:
            db = self.db if self.db is not None else initialize_database()
            return list(HumanInput.select().order_by(HumanInput.created_at.desc()))
        except peewee.DatabaseError as e:
            logger.error(f"Failed to fetch all human inputs: {str(e)}")
            raise
    
    def get_recent(self, limit: int = 10) -> List[HumanInput]:
        """
        Retrieve the most recent human input records.
        
        Args:
            limit: Maximum number of records to retrieve (default: 10)
            
        Returns:
            List[HumanInput]: List of the most recent human input records
            
        Raises:
            peewee.DatabaseError: If there's an error accessing the database
        """
        try:
            db = self.db if self.db is not None else initialize_database()
            return list(HumanInput.select().order_by(HumanInput.created_at.desc()).limit(limit))
        except peewee.DatabaseError as e:
            logger.error(f"Failed to fetch recent human inputs: {str(e)}")
            raise
    
    def get_by_source(self, source: str) -> List[HumanInput]:
        """
        Retrieve human input records by source.
        
        Args:
            source: The source to filter by (e.g., "cli", "chat", "hil")
            
        Returns:
            List[HumanInput]: List of human input records from the specified source
            
        Raises:
            peewee.DatabaseError: If there's an error accessing the database
        """
        try:
            db = self.db if self.db is not None else initialize_database()
            return list(HumanInput.select().where(HumanInput.source == source).order_by(HumanInput.created_at.desc()))
        except peewee.DatabaseError as e:
            logger.error(f"Failed to fetch human inputs by source {source}: {str(e)}")
            raise
    
    def garbage_collect(self) -> int:
        """
        Remove old human input records when the count exceeds 100.
        
        This method keeps the 100 most recent records and deletes any older ones.
        
        Returns:
            int: Number of records deleted
            
        Raises:
            peewee.DatabaseError: If there's an error accessing the database
        """
        try:
            db = self.db if self.db is not None else initialize_database()
            # Get the count of records
            record_count = HumanInput.select().count()
            
            # If we have more than 100 records, delete the oldest ones
            if record_count > 100:
                # Get IDs of records to keep (100 most recent)
                keep_ids = [input_record.id for input_record in HumanInput.select(HumanInput.id)
                           .order_by(HumanInput.created_at.desc())
                           .limit(100)]
                
                # Delete records not in the keep_ids list
                delete_query = HumanInput.delete().where(HumanInput.id.not_in(keep_ids))
                deleted_count = delete_query.execute()
                
                logger.info(f"Garbage collected {deleted_count} old human input records")
                return deleted_count
            
            return 0
        except peewee.DatabaseError as e:
            logger.error(f"Failed to garbage collect human input records: {str(e)}")
            raise