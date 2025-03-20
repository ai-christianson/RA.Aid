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
from ra_aid.database.models import Session, HumanInput
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

    def _get_display_name_for_session(self, session_id: int) -> Optional[str]:
        """
        Get the display name for a session using Peewee ORM.

        Args:
            session_id: The ID of the session

        Returns:
            Optional[str]: The display name or None if an error occurs
        """
        try:
            # Get the session
            session = Session.get_by_id(session_id)
            if not session:
                return None
            
            # Try to get the oldest human input for this session
            oldest_input = (
                HumanInput.select()
                .where(HumanInput.session == session_id)
                .order_by(HumanInput.id)
                .first()
            )
            
            # Use human input content if available, otherwise use command line
            if oldest_input:
                content = oldest_input.content
                if len(content) > 80:
                    return content[:80] + "..."
                return content
            
            # Fallback to command line
            if len(session.command_line) > 80:
                return session.command_line[:80] + "..."
            return session.command_line
        except Exception as e:
            logger.error(f"Error getting display name for session {session_id}: {str(e)}")
            return None

    def _to_model(self, session: Optional[Session]) -> Optional[SessionModel]:
        """
        Convert a Session model to a SessionModel Pydantic model.

        Args:
            session: Session peewee model or None

        Returns:
            SessionModel or None if session is None
        """
        if session is None:
            return None

        # Handle both Model objects and dictionaries from query aliases
        if isinstance(session, dict) or hasattr(session, 'keys'):
            # This is a dictionary-like object from a complex query
            session_dict = dict(session)
            return SessionModel(
                id=session_dict.get("id"),
                start_time=session_dict.get("start_time"),
                command_line=session_dict.get("command_line"),
                program_version=session_dict.get("program_version"),
                machine_info=session_dict.get("machine_info"),
                created_at=session_dict.get("created_at"),
                updated_at=session_dict.get("updated_at"),
                display_name=session_dict.get("display_name"),
            )
        else:
            # Handle regular model object
            # Extract display_name if available as attribute
            display_name = getattr(session, "display_name", None)
            
            # Create the Pydantic model
            return SessionModel(
                id=session.id,
                start_time=session.start_time,
                command_line=session.command_line,
                program_version=session.program_version,
                machine_info=session.machine_info,
                created_at=session.created_at,
                updated_at=session.updated_at,
                display_name=display_name,
            )

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
            )

            # Store the current session
            self.current_session = session

            logger.debug(f"Created new session with ID {session.id}")
            
            # Get the session with display_name computed
            result = self.get(session.id)
            if result is None:
                # Fallback to direct conversion if get() fails for some reason
                result = self._to_model(session)
                # Set display_name to command_line (truncated if needed)
                if result.command_line and len(result.command_line) > 80:
                    result.display_name = result.command_line[:80] + "..."
                else:
                    result.display_name = result.command_line
            
            return result
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
        current_session = self.get_current_session_record()
        if current_session is None:
            return None
            
        try:
            return self.get(current_session.id)
        except peewee.DatabaseError as e:
            logger.error(f"Database error getting current session: {str(e)}")
            # Fallback to direct conversion
            result = self._to_model(current_session)
            if result and result.command_line:
                # Set display_name to command_line (truncated if needed)
                if len(result.command_line) > 80:
                    result.display_name = result.command_line[:80] + "..."
                else:
                    result.display_name = result.command_line
            return result

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
        session = self.get_current_session_record()
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
            # Get the session using normal ORM
            session = Session.get_or_none(Session.id == session_id)
            if session is None:
                return None
            
            # Convert to model
            model = self._to_model(session)
            
            # Get display name directly
            display_name = self._get_display_name_for_session(session_id)
            if display_name:
                model.display_name = display_name
            elif model.command_line:
                # Fallback to command line
                if len(model.command_line) > 80:
                    model.display_name = model.command_line[:80] + "..."
                else:
                    model.display_name = model.command_line
            
            return model
            
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
            
            # Process sessions and add display_name
            result = []
            for session in sessions:
                model = self._to_model(session)
                
                # Get display name directly
                display_name = self._get_display_name_for_session(session.id)
                if display_name:
                    model.display_name = display_name
                elif model.command_line:
                    # Fallback to command line
                    if len(model.command_line) > 80:
                        model.display_name = model.command_line[:80] + "..."
                    else:
                        model.display_name = model.command_line
                
                result.append(model)
                
            return result, total_count
            
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
            # Get recent sessions
            sessions = list(
                Session.select().order_by(Session.created_at.desc()).limit(limit)
            )
            
            # Process sessions and add display_name
            result = []
            for session in sessions:
                model = self._to_model(session)
                
                # Get display name directly
                display_name = self._get_display_name_for_session(session.id)
                if display_name:
                    model.display_name = display_name
                elif model.command_line:
                    # Fallback to command line
                    if len(model.command_line) > 80:
                        model.display_name = model.command_line[:80] + "..."
                    else:
                        model.display_name = model.command_line
                
                result.append(model)
                
            return result
            
        except peewee.DatabaseError as e:
            logger.error(f"Failed to get recent sessions: {str(e)}")
            return []

    def get_latest_session(self) -> Optional[SessionModel]:
        """
        Get the most recent session from the database.

        This method retrieves the single most recent session based on creation time.
        Unlike get_current_session(), this always queries the database and doesn't
        use the cached current_session.

        Returns:
            Optional[SessionModel]: The most recent session or None if no sessions exist
        """
        try:
            # Get the most recent session
            session = Session.select().order_by(Session.created_at.desc()).first()
            if session is None:
                return None
                
            # Convert to model
            model = self._to_model(session)
            
            # Get display name directly
            display_name = self._get_display_name_for_session(session.id)
            if display_name:
                model.display_name = display_name
            elif model.command_line:
                # Fallback to command line
                if len(model.command_line) > 80:
                    model.display_name = model.command_line[:80] + "..."
                else:
                    model.display_name = model.command_line
            
            return model
            
        except peewee.DatabaseError as e:
            logger.error(f"Failed to get latest session: {str(e)}")
            return None
            
    def get_all_session_ids(self) -> List[int]:
        """
        Get all session IDs from the database.
        
        Returns:
            List[int]: List of all session IDs ordered by creation time (newest first)
        """
        try:
            # Query for all session IDs
            query = Session.select(Session.id).order_by(Session.created_at.desc())
            return [session.id for session in query]
        except peewee.DatabaseError as e:
            logger.error(f"Failed to get all session IDs: {str(e)}")
            return []

    def _get_display_name_subquery(self):
        """
        Create a subquery for computing the display_name field.

        This creates a SQL expression that computes the display_name based on:
        1. First 80 chars of the oldest human_input content, or
        2. First 80 chars of command_line if no human_input exists

        Returns:
            peewee.SQL: The SQL expression for the display_name computation
        """
        # Use peewee.SQL to create a raw SQL expression for display_name
        # We need to use the actual table name "session" instead of any aliases
        display_name_sql = """
            COALESCE(
                (SELECT 
                    CASE 
                        WHEN LENGTH(hi.content) > 80 THEN SUBSTR(hi.content, 1, 80) || '...'
                        ELSE hi.content
                    END
                FROM human_input hi
                WHERE hi.session_id = t1.id
                ORDER BY hi.id ASC
                LIMIT 1),
                CASE 
                    WHEN LENGTH(t1.command_line) > 80 THEN SUBSTR(t1.command_line, 1, 80) || '...'
                    ELSE t1.command_line
                END
            )
        """
        return peewee.SQL(display_name_sql)
