"""
Research note repository implementation for database access.

This module provides a repository implementation for the ResearchNote model,
following the repository pattern for data access abstraction.
"""

from typing import Dict, List, Optional
import contextvars
from contextlib import contextmanager

import peewee

from ra_aid.database.models import ResearchNote
from ra_aid.logging_config import get_logger

logger = get_logger(__name__)

# Create contextvar to hold the ResearchNoteRepository instance
research_note_repo_var = contextvars.ContextVar("research_note_repo", default=None)


class ResearchNoteRepositoryManager:
    """
    Context manager for ResearchNoteRepository.

    This class provides a context manager interface for ResearchNoteRepository,
    using the contextvars approach for thread safety.

    Example:
        with DatabaseManager() as db:
            with ResearchNoteRepositoryManager(db) as repo:
                # Use the repository
                note = repo.create("Research findings about the topic")
                all_notes = repo.get_all()
    """

    def __init__(self, db):
        """
        Initialize the ResearchNoteRepositoryManager.

        Args:
            db: Database connection to use (required)
        """
        self.db = db

    def __enter__(self) -> 'ResearchNoteRepository':
        """
        Initialize the ResearchNoteRepository and return it.

        Returns:
            ResearchNoteRepository: The initialized repository
        """
        repo = ResearchNoteRepository(self.db)
        research_note_repo_var.set(repo)
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
        research_note_repo_var.set(None)

        # Don't suppress exceptions
        return False


def get_research_note_repository() -> 'ResearchNoteRepository':
    """
    Get the current ResearchNoteRepository instance.

    Returns:
        ResearchNoteRepository: The current repository instance
        
    Raises:
        RuntimeError: If no repository has been initialized with ResearchNoteRepositoryManager
    """
    repo = research_note_repo_var.get()
    if repo is None:
        raise RuntimeError(
            "No ResearchNoteRepository available. "
            "Make sure to initialize one with ResearchNoteRepositoryManager first."
        )
    return repo


class ResearchNoteRepository:
    """
    Repository for managing ResearchNote database operations.
    
    This class provides methods for performing CRUD operations on the ResearchNote model,
    abstracting the database access details from the business logic.
    
    Example:
        with DatabaseManager() as db:
            with ResearchNoteRepositoryManager(db) as repo:
                note = repo.create("Research findings about the topic")
                all_notes = repo.get_all()
    """
    
    def __init__(self, db):
        """
        Initialize the repository with a database connection.
        
        Args:
            db: Database connection to use (required)
        """
        if db is None:
            raise ValueError("Database connection is required for ResearchNoteRepository")
        self.db = db
    
    def create(self, content: str, human_input_id: Optional[int] = None) -> ResearchNote:
        """
        Create a new research note in the database.
        
        Args:
            content: The text content of the research note
            human_input_id: Optional ID of the associated human input
            
        Returns:
            ResearchNote: The newly created research note instance
            
        Raises:
            peewee.DatabaseError: If there's an error creating the note
        """
        try:
            note = ResearchNote.create(content=content, human_input_id=human_input_id)
            logger.debug(f"Created research note ID {note.id}: {content[:50]}...")
            return note
        except peewee.DatabaseError as e:
            logger.error(f"Failed to create research note: {str(e)}")
            raise
    
    def get(self, note_id: int) -> Optional[ResearchNote]:
        """
        Retrieve a research note by its ID.
        
        Args:
            note_id: The ID of the research note to retrieve
            
        Returns:
            Optional[ResearchNote]: The research note instance if found, None otherwise
            
        Raises:
            peewee.DatabaseError: If there's an error accessing the database
        """
        try:
            return ResearchNote.get_or_none(ResearchNote.id == note_id)
        except peewee.DatabaseError as e:
            logger.error(f"Failed to fetch research note {note_id}: {str(e)}")
            raise
    
    def update(self, note_id: int, content: str) -> Optional[ResearchNote]:
        """
        Update an existing research note.
        
        Args:
            note_id: The ID of the research note to update
            content: The new content for the research note
            
        Returns:
            Optional[ResearchNote]: The updated research note if found, None otherwise
            
        Raises:
            peewee.DatabaseError: If there's an error updating the note
        """
        try:
            # First check if the note exists
            note = self.get(note_id)
            if not note:
                logger.warning(f"Attempted to update non-existent research note {note_id}")
                return None
            
            # Update the note
            note.content = content
            note.save()
            logger.debug(f"Updated research note ID {note_id}: {content[:50]}...")
            return note
        except peewee.DatabaseError as e:
            logger.error(f"Failed to update research note {note_id}: {str(e)}")
            raise
    
    def delete(self, note_id: int) -> bool:
        """
        Delete a research note by its ID.
        
        Args:
            note_id: The ID of the research note to delete
            
        Returns:
            bool: True if the note was deleted, False if it wasn't found
            
        Raises:
            peewee.DatabaseError: If there's an error deleting the note
        """
        try:
            # First check if the note exists
            note = self.get(note_id)
            if not note:
                logger.warning(f"Attempted to delete non-existent research note {note_id}")
                return False
            
            # Delete the note
            note.delete_instance()
            logger.debug(f"Deleted research note ID {note_id}")
            return True
        except peewee.DatabaseError as e:
            logger.error(f"Failed to delete research note {note_id}: {str(e)}")
            raise
    
    def get_all(self) -> List[ResearchNote]:
        """
        Retrieve all research notes from the database.
        
        Returns:
            List[ResearchNote]: List of all research note instances
            
        Raises:
            peewee.DatabaseError: If there's an error accessing the database
        """
        try:
            return list(ResearchNote.select().order_by(ResearchNote.id))
        except peewee.DatabaseError as e:
            logger.error(f"Failed to fetch all research notes: {str(e)}")
            raise
    
    def get_notes_dict(self) -> Dict[int, str]:
        """
        Retrieve all research notes as a dictionary mapping IDs to content.
        
        This method is useful for compatibility with the existing memory format.
        
        Returns:
            Dict[int, str]: Dictionary with note IDs as keys and content as values
            
        Raises:
            peewee.DatabaseError: If there's an error accessing the database
        """
        try:
            notes = self.get_all()
            return {note.id: note.content for note in notes}
        except peewee.DatabaseError as e:
            logger.error(f"Failed to fetch research notes as dictionary: {str(e)}")
            raise