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

from ra_aid.config import DEFAULT_MODEL
from ra_aid.database.models import Session
from ra_aid.database.pydantic_models import SessionModel
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

    def __enter__(self) -> "SessionRepository":
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


def get_session_repository() -> "SessionRepository":
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

    def _to_model(self, session: Optional[Session]) -> Optional[SessionModel]:
        """
        Convert a Peewee Session object to a Pydantic SessionModel.

        Args:
            session: Peewee Session instance or None

        Returns:
            Optional[SessionModel]: Pydantic model representation or None if session is None
        """
        if session is None:
            return None

        return SessionModel.model_validate(session, from_attributes=True)

    def create_session(self, metadata: Optional[Dict[str, Any]] = None) -> SessionModel:
        """
        Create a new session record in the database.

        Args:
            metadata: Optional dictionary of additional metadata to store with the session

        Returns:
            SessionModel: The newly created session instance

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
                machine_info=machine_info,
                total_input_tokens=0,
                total_output_tokens=0,
                total_tokens=0,
                total_cost=0.0,
            )

            # Store the current session
            self.current_session = session

            logger.debug(f"Created new session with ID {session.id}")
            return self._to_model(session)
        except peewee.DatabaseError as e:
            logger.error(f"Failed to create session record: {str(e)}")
            raise

    def get_current_session(self) -> Optional[SessionModel]:
        """
        Get the current active session as a Pydantic model.

        If no session has been created in this repository instance,
        retrieves the most recent session from the database.

        Returns:
            Optional[SessionModel]: The current session or None if no sessions exist
        """
        session_record = self.get_current_session_record()
        return self._to_model(session_record)

    def get_current_session_record(self) -> Optional[Session]:
        """
        Get the current active session as a Peewee model.

        If no session has been created in this repository instance,
        retrieves the most recent session from the database.

        Returns:
            Optional[Session]: The current session Peewee record or None if no sessions exist
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
            logger.error(f"Failed to get current session record: {str(e)}")
            return None

    def get_current_session_id(self) -> Optional[int]:
        """
        Get the ID of the current active session.

        Returns:
            Optional[int]: The ID of the current session or None if no session exists
        """
        session = self.get_current_session()
        return session.id if session else None

    def get(self, session_id: int) -> Optional[SessionModel]:
        """
        Get a session by its ID.

        Args:
            session_id: The ID of the session to retrieve

        Returns:
            Optional[SessionModel]: The session with the given ID or None if not found
        """
        try:
            session = Session.get_or_none(Session.id == session_id)
            return self._to_model(session)
        except peewee.DatabaseError as e:
            logger.error(f"Database error getting session {session_id}: {str(e)}")
            return None

    def get_all(
        self, offset: int = 0, limit: int = 10
    ) -> tuple[List[SessionModel], int]:
        """
        Get all sessions from the database with pagination support.

        Args:
            offset: Number of sessions to skip (default: 0)
            limit: Maximum number of sessions to return (default: 10)

        Returns:
            tuple: (List[SessionModel], int) containing the list of sessions and the total count
        """
        try:
            # Get total count for pagination info
            total_count = Session.select().count()

            # Get paginated sessions ordered by created_at in descending order (newest first)
            sessions = list(
                Session.select()
                .order_by(Session.created_at.desc())
                .offset(offset)
                .limit(limit)
            )

            return [self._to_model(session) for session in sessions], total_count
        except peewee.DatabaseError as e:
            logger.error(f"Failed to get all sessions with pagination: {str(e)}")
            return [], 0

    def get_recent(self, limit: int = 10) -> List[SessionModel]:
        """
        Get the most recent sessions from the database.

        Args:
            limit: Maximum number of sessions to return (default: 10)

        Returns:
            List[SessionModel]: List of the most recent sessions
        """
        try:
            sessions = list(
                Session.select().order_by(Session.created_at.desc()).limit(limit)
            )
            return [self._to_model(session) for session in sessions]
        except peewee.DatabaseError as e:
            logger.error(f"Failed to get recent sessions: {str(e)}")
            return []

    def update_token_usage(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        last_cost: float = 0.0,
    ) -> Optional[Session]:
        """
        Update token usage for a session.

        Args:
            input_tokens: Number of input tokens to add
            output_tokens: Number of output tokens to add
            last_cost: Pre-calculated cost for this update

        Returns:
            Optional[Session]: Updated session or None if not found
        """
        try:
            session = self.get_current_session_record()
            if not session:
                logger.warning("Attempted to update non-existent session.")
                return None

            session.total_input_tokens = session.total_input_tokens + input_tokens
            session.total_output_tokens = session.total_output_tokens + output_tokens
            session.total_tokens = (
                session.total_input_tokens + session.total_output_tokens
            )
            session.total_cost = session.total_cost + last_cost
            session.save()

            session_id = session.get_id()
            print(f"session_id={session_id}")
            logger.debug(
                f"Updated session {session_id} with {input_tokens} input tokens, "
                f"{output_tokens} output tokens, ${session.total_cost:.6f} cost"
            )

            return self.get(session_id)
        except peewee.DatabaseError as e:
            logger.error(
                f"Failed to update token usage for session {session_id}: {str(e)}"
            )
            return None
