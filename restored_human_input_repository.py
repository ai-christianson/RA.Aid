"""
Human input repository implementation for database access.

This module provides a repository implementation for the HumanInput model,
following the repository pattern for data access abstraction.
"""

from typing import Dict, List, Optional
import contextvars

import peewee

from ra_aid.database.models import HumanInput, Session
from ra_aid.database.pydantic_models import HumanInputModel
from ra_aid.logging_config import get_logger

logger = get_logger(__name__)

# Create contextvar to hold the HumanInputRepository instance
human_input_repo_var = contextvars.ContextVar("human_input_repo", default=None)


class HumanInputRepositoryManager:
    """
    Context manager for HumanInputRepository.

    This class provides a context manager interface for HumanInputRepository,
    using the contextvars approach for thread safety.

    Example:
        with DatabaseManager() as db:
            with HumanInputRepositoryManager(db) as repo:
                # Use the repository
                input_record = repo.create(content="User input", source="chat")
                recent_inputs = repo.get_recent(5)
    """

    def __init__(self, db):
        """
        Initialize the HumanInputRepositoryManager.

        Args:
            db: Database connection to use (required)
        """
        self.db = db

    def __enter__(self) -> 'HumanInputRepository':
        """
        Initialize the HumanInputRepository and return it.

        Returns:
            HumanInputRepository: The initialized repository
        """
        repo = HumanInputRepository(self.db)
        human_input_repo_var.set(repo)
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
        human_input_repo_var.set(None)

        # Don't suppress exceptions
        return False


def get_human_input_repository() -> 'HumanInputRepository':
    """
    Get the current HumanInputRepository instance.

    Returns:
        HumanInputRepository: The current repository instance
        
    Raises:
        RuntimeError: If no repository has been initialized with HumanInputRepositoryManager
    """
    repo = human_input_repo_var.get()
    if repo is None:
        raise RuntimeError(
            "No HumanInputRepository available. "
            "Make sure to initialize one with HumanInputRepositoryManager first."
        )
    return repo


class HumanInputRepository:
    """
    Repository for managing HumanInput database operations.
    
    This class provides methods for performing CRUD operations on the HumanInput model,
    abstracting the database access details from the business logic.
    
    Example:
        with DatabaseManager() as db:
            with HumanInputRepositoryManager(db) as repo:
                input_record = repo.create("User's message", "chat")
                recent_inputs = repo.get_recent(5)
    """
    
    def __init__(self, db):
        """
        Initialize the repository with a database connection.
        
        Args:
            db: Database connection to use (required)
        """
        if db is None:
            raise ValueError("Database connection is required for HumanInputRepository")
        self.db = db
        
    def _to_model(self, human_input: Optional[HumanInput]) -> Optional[HumanInputModel]:
        """
        Convert a Peewee HumanInput object to a Pydantic HumanInputModel.
        
        Args:
            human_input: Peewee HumanInput instance or None
            
        Returns:
            Optional[HumanInputModel]: Pydantic model representation or None if human_input is None
        """
        if human_input is None:
            return None
        
        return HumanInputModel.model_validate(human_input, from_attributes=True)
    
    def create(self, content: str, source: str, session_id: Optional[int] = None) -> HumanInputModel:
        """
        Create a new human input record in the database.
        
        Args:
            content: The text content of the human input
            source: The source of the input (e.g., "cli", "chat", "hil")
            session_id: Optional ID of the session to associate with this input
            
        Returns:
            HumanInputModel: The newly created human input instance
            
        Raises:
            peewee.DatabaseError: If there's an error creating the record
        """
        try:
            # Get session object if session_id provided
            session = None
            if session_id is not None:
                try:
                    session = Session.get_by_id(session_id)
                except peewee.DoesNotExist:
                    logger.warning(f"Session with ID {session_id} not found, creating human input without session")
            
            input_record = HumanInput.create(content=content, source=source, session=session)
            logger.debug(f"Created human input ID {input_record.id} from {source}" + 
                        (f" for session {session_id}" if session_id else ""))
            return self._to_model(input_record)
        except peewee.DatabaseError as e:
            logger.error(f"Failed to create human input record: {str(e)}")
            raise
    
    def get(self, input_id: int) -> Optional[HumanInputModel]:
        """
        Retrieve a human input record by its ID.
        
        Args:
            input_id: The ID of the human input to retrieve
            
        Returns:
            Optional[HumanInputModel]: The human input instance if found, None otherwise
            
        Raises:
            peewee.DatabaseError: If there's an error accessing the database
        """
        try:
            human_input = HumanInput.get_or_none(HumanInput.id == input_id)
            return self._to_model(human_input)
        except peewee.DatabaseError as e:
            logger.error(f"Failed to fetch human input {input_id}: {str(e)}")
            raise
    
    def update(self, input_id: int, content: str = None, source: str = None) -> Optional[HumanInputModel]:
        """
        Update an existing human input record.
        
        Args:
            input_id: The ID of the human input to update
            content: The new content for the human input
            source: The new source for the human input
            
        Returns:
            Optional[HumanInputModel]: The updated human input if found, None otherwise
            
        Raises:
            peewee.DatabaseError: If there's an error updating the record
        """
        try:
            # We need to get the raw Peewee object for updating
            input_record = HumanInput.get_or_none(HumanInput.id == input_id)
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
            return self._to_model(input_record)
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
    
    def get_all(self) -> List[HumanInputModel]:
        """
        Retrieve all human input records from the database.
        
        Returns:
            List[HumanInputModel]: List of all human input instances
            
        Raises:
            peewee.DatabaseError: If there's an error accessing the database
        """
        try:
            human_inputs = list(HumanInput.select().order_by(HumanInput.created_at.desc()))
            return [self._to_model(input) for input in human_inputs]
        except peewee.DatabaseError as e:
            logger.error(f"Failed to fetch all human inputs: {str(e)}")
            raise
    
    def get_recent(self, limit: int = 10) -> List[HumanInputModel]:
        """
        Retrieve the most recent human input records.
        
        Args:
            limit: Maximum number of records to retrieve (default: 10)
            
        Returns:
            List[HumanInputModel]: List of the most recent human input records
            
        Raises:
            peewee.DatabaseError: If there's an error accessing the database
        """
        try:
            human_inputs = list(HumanInput.select().order_by(HumanInput.created_at.desc()).limit(limit))
            return [self._to_model(input) for input in human_inputs]
        except peewee.DatabaseError as e:
            logger.error(f"Failed to fetch recent human inputs: {str(e)}")
            raise

    def get_most_recent_id(self) -> Optional[int]:
        """
        Get the ID of the most recent human input record.
        
        Returns:
            Optional[int]: The ID of the most recent human input, or None if no records exist
            
        Raises:
            peewee.DatabaseError: If there's an error accessing the database
        """
        try:
            recent_inputs = self.get_recent(1)
            if recent_inputs and len(recent_inputs) > 0:
                return recent_inputs[0].id
            return None
        except peewee.DatabaseError as e:
            logger.error(f"Failed to fetch most recent human input ID: {str(e)}")
            raise
    
    def get_by_source(self, source: str) -> List[HumanInputModel]:
        """
        Retrieve human input records by source.
        
        Args:
            source: The source to filter by (e.g., "cli", "chat", "hil")
            
        Returns:
            List[HumanInputModel]: List of human input records from the specified source
            
        Raises:
            peewee.DatabaseError: If there's an error accessing the database
        """
        try:
            human_inputs = list(HumanInput.select().where(HumanInput.source == source).order_by(HumanInput.created_at.desc()))
            return [self._to_model(input) for input in human_inputs]
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