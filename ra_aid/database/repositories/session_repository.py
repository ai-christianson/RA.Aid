"""
Session repository implementation for database access.

This module provides a repository implementation for the Session model,
following the repository pattern for data access abstraction. It handles
operations for storing and retrieving application session information.
"""

from typing import Dict, List, Optional, Any
import contextvars
import datetime
import json
import logging
import sys

import peewee

from ra_aid.database.models import Session
from ra_aid.__version__ import __version__
from ra_aid.logging_config import get_logger

logger = get_logger(__name__)

# Create contextvar to hold the SessionRepository instance
session_repo_var = contextvars.ContextVar("session_repo", default=None)


class SessionRepositoryManager:
    """
    Context manager for SessionRepository.

    This class provides a context manager interface for SessionRepository,
    using the contextvars approach for thread safety.

    Example:
        with DatabaseManager() as db:
            with SessionRepositoryManager(db) as repo:
                # Use the repository
                session = repo.create_session()
                current_session = repo.get_current_session()
    """

    def __init__(self, db):
        """
        Initialize the SessionRepositoryManager.

        Args:
            db: Database connection to use (required)
        """
        self.db = db

    def __enter__(self) -> 'SessionRepository':
        """
        Initialize the SessionRepository and return it.

        Returns:
            SessionRepository: The initialized repository
        """
        repo = SessionRepository(self.db)
        session_repo_var.set(repo)
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
        session_repo_var.set(None)

        # Don't suppress exceptions
        return False


def get_session_repository() -> 'SessionRepository':
    """
    Get the current SessionRepository instance.

    Returns:
        SessionRepository: The current repository instance
        
    Raises:
        RuntimeError: If no repository has been initialized with SessionRepositoryManager
    """
    repo = session_repo_var.get()
    if repo is None:
        raise RuntimeError(
            "No SessionRepository available. "
            "Make sure to initialize one with SessionRepositoryManager first."
        )
    return repo


class SessionRepository:
    """
    Repository for handling Session records in the database.

    This class provides methods for creating, retrieving, and managing Session records.
    It abstracts away the database operations and provides a clean interface for working
    with Session entities.
    """

    def __init__(self, db):
        """
        Initialize the SessionRepository.

        Args:
            db: Database connection to use (required)
        """
        if db is None:
            raise ValueError("Database connection is required for SessionRepository")
        self.db = db
        self.current_session = None

    def create_session(self, metadata: Optional[Dict[str, Any]] = None) -> Session:
        """
        Create a new session record in the database.
        
        Args:
            metadata: Optional dictionary of additional metadata to store with the session
            
        Returns:
            Session: The newly created session instance
            
        Raises:
            peewee.DatabaseError: If there's an error creating the record
        """
        try:
            # Get command line arguments
            command_line = " ".join(sys.argv)
            
            # Get program version
            program_version = __version__
            
            # JSON encode metadata if provided
            machine_info = json.dumps(metadata) if metadata is not None else None
            
            session = Session.create(
                start_time=datetime.datetime.now(),
                command_line=command_line,
                program_version=program_version,
                machine_info=machine_info
            )
            
            # Store the current session
            self.current_session = session
            
            logger.debug(f"Created new session with ID {session.id}")
            return session
        except peewee.DatabaseError as e:
            logger.error(f"Failed to create session record: {str(e)}")
            raise

    def get_current_session(self) -> Optional[Session]:
        """
        Get the current active session.
        
        If no session has been created in this repository instance,
        retrieves the most recent session from the database.
        
        Returns:
            Optional[Session]: The current session or None if no sessions exist
        """
        if self.current_session is not None:
            return self.current_session
        
        try:
            # Find the most recent session
            session = Session.select().order_by(Session.created_at.desc()).first()
            if session:
                self.current_session = session
            return session
        except peewee.DatabaseError as e:
            logger.error(f"Failed to get current session: {str(e)}")
            return None

    def get_current_session_id(self) -> Optional[int]:
        """
        Get the ID of the current active session.
        
        Returns:
            Optional[int]: The ID of the current session or None if no session exists
        """
        session = self.get_current_session()
        return session.id if session else None

    def get(self, session_id: int) -> Optional[Session]:
        """
        Get a session by its ID.
        
        Args:
            session_id: The ID of the session to retrieve
            
        Returns:
            Optional[Session]: The session with the given ID or None if not found
        """
        try:
            return Session.get_or_none(Session.id == session_id)
        except peewee.DatabaseError as e:
            logger.error(f"Database error getting session {session_id}: {str(e)}")
            return None

    def get_all(self) -> List[Session]:
        """
        Get all sessions from the database.
        
        Returns:
            List[Session]: List of all sessions
        """
        try:
            return list(Session.select().order_by(Session.created_at.desc()))
        except peewee.DatabaseError as e:
            logger.error(f"Failed to get all sessions: {str(e)}")
            return []

    def get_recent(self, limit: int = 10) -> List[Session]:
        """
        Get the most recent sessions from the database.
        
        Args:
            limit: Maximum number of sessions to return (default: 10)
            
        Returns:
            List[Session]: List of the most recent sessions
        """
        try:
            return list(
                Session.select()
                .order_by(Session.created_at.desc())
                .limit(limit)
            )
        except peewee.DatabaseError as e:
            logger.error(f"Failed to get recent sessions: {str(e)}")
            return []