"""
Database connection management for ra_aid.

This module provides functions to initialize, get, and close database connections.
It also provides a context manager for database connections.
"""

import contextvars
import os
from pathlib import Path
from typing import Any, Optional

import peewee

from ra_aid.logging_config import get_logger

# Create contextvar to hold the database connection
db_var = contextvars.ContextVar("db", default=None)
logger = get_logger(__name__)


class DatabaseManager:
    """
    Context manager for database connections.

    This class provides a context manager interface for database connections,
    using the existing contextvars approach internally.

    Example:
        with DatabaseManager() as db:
            # Use the database connection
            db.execute_sql("SELECT * FROM table")

        # Or with in-memory database:
        with DatabaseManager(in_memory=True) as db:
            # Use in-memory database
    """

    def __init__(self, in_memory: bool = False):
        """
        Initialize the DatabaseManager.

        Args:
            in_memory: Whether to use an in-memory database (default: False)
        """
        self.in_memory = in_memory

    def __enter__(self) -> peewee.SqliteDatabase:
        """
        Initialize the database connection and return it.

        Returns:
            peewee.SqliteDatabase: The initialized database connection
        """
        return init_db(in_memory=self.in_memory)

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[Any],
    ) -> None:
        """
        Close the database connection when exiting the context.

        Args:
            exc_type: The exception type if an exception was raised
            exc_val: The exception value if an exception was raised
            exc_tb: The traceback if an exception was raised
        """
        close_db()

        # Don't suppress exceptions
        return False


def init_db(in_memory: bool = False) -> peewee.SqliteDatabase:
    """
    Initialize the database connection.

    Creates the .ra-aid directory if it doesn't exist and initializes
    the SQLite database connection. If a database connection already exists,
    returns the existing connection instead of creating a new one.

    Args:
        in_memory: Whether to use an in-memory database (default: False)

    Returns:
        peewee.SqliteDatabase: The initialized database connection
    """
    # Check if a database connection already exists
    existing_db = db_var.get()
    if existing_db is not None:
        # If the connection exists but is closed, reopen it
        if existing_db.is_closed():
            try:
                existing_db.connect()
            except peewee.OperationalError as e:
                logger.error(f"Failed to reopen existing database connection: {str(e)}")
                # Continue to create a new connection if reopening fails
            else:
                return existing_db
        else:
            # Connection exists and is open, return it
            return existing_db

    # Set up database path
    if in_memory:
        # Use in-memory database
        db_path = ":memory:"
        logger.debug("Using in-memory SQLite database")
    else:
        # Get current working directory and create .ra-aid directory if it doesn't exist
        cwd = os.getcwd()
        logger.debug(f"Current working directory: {cwd}")

        # Define the .ra-aid directory path
        ra_aid_dir_str = os.path.join(cwd, ".ra-aid")
        ra_aid_dir = Path(ra_aid_dir_str)
        ra_aid_dir = ra_aid_dir.absolute()  # Ensure we have the absolute path
        ra_aid_dir_str = str(
            ra_aid_dir
        )  # Update string representation with absolute path

        logger.debug(f"Creating database directory at: {ra_aid_dir_str}")

        # Multiple approaches to ensure directory creation
        directory_created = False
        error_messages = []

        # Approach 1: Try os.mkdir directly
        if not os.path.exists(ra_aid_dir_str):
            try:
                logger.debug("Attempting directory creation with os.mkdir")
                os.mkdir(ra_aid_dir_str, mode=0o755)
                directory_created = os.path.exists(ra_aid_dir_str) and os.path.isdir(
                    ra_aid_dir_str
                )
                if directory_created:
                    logger.debug("Directory created successfully with os.mkdir")
            except Exception as e:
                error_msg = f"os.mkdir failed: {str(e)}"
                logger.debug(error_msg)
                error_messages.append(error_msg)
        else:
            logger.debug("Directory already exists, skipping creation")
            directory_created = True

        # Approach 2: Try os.makedirs if os.mkdir failed
        if not directory_created:
            try:
                logger.debug("Attempting directory creation with os.makedirs")
                os.makedirs(ra_aid_dir_str, exist_ok=True, mode=0o755)
                directory_created = os.path.exists(ra_aid_dir_str) and os.path.isdir(
                    ra_aid_dir_str
                )
                if directory_created:
                    logger.debug("Directory created successfully with os.makedirs")
            except Exception as e:
                error_msg = f"os.makedirs failed: {str(e)}"
                logger.debug(error_msg)
                error_messages.append(error_msg)

        # Approach 3: Try Path.mkdir if previous methods failed
        if not directory_created:
            try:
                logger.debug("Attempting directory creation with Path.mkdir")
                ra_aid_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
                directory_created = os.path.exists(ra_aid_dir_str) and os.path.isdir(
                    ra_aid_dir_str
                )
                if directory_created:
                    logger.debug("Directory created successfully with Path.mkdir")
            except Exception as e:
                error_msg = f"Path.mkdir failed: {str(e)}"
                logger.debug(error_msg)
                error_messages.append(error_msg)

        # Verify the directory was actually created
        path_exists = ra_aid_dir.exists()
        os_exists = os.path.exists(ra_aid_dir_str)
        is_dir = os.path.isdir(ra_aid_dir_str) if os_exists else False

        logger.debug(
            f"Directory verification: Path.exists={path_exists}, os.path.exists={os_exists}, os.path.isdir={is_dir}"
        )

        # Check parent directory permissions and contents for debugging
        try:
            parent_dir = os.path.dirname(ra_aid_dir_str)
            parent_perms = oct(os.stat(parent_dir).st_mode)[-3:]
            parent_contents = os.listdir(parent_dir)
            logger.debug(f"Parent directory {parent_dir} permissions: {parent_perms}")
            logger.debug(f"Parent directory contents: {parent_contents}")
        except Exception as e:
            logger.debug(f"Could not check parent directory: {str(e)}")

        if not os_exists or not is_dir:
            error_msg = f"Directory does not exist or is not a directory after creation attempts: {ra_aid_dir_str}"
            logger.error(error_msg)
            if error_messages:
                logger.error(f"Previous errors: {', '.join(error_messages)}")
            raise FileNotFoundError(f"Failed to create directory: {ra_aid_dir_str}")

        # Check directory permissions
        try:
            permissions = oct(os.stat(ra_aid_dir_str).st_mode)[-3:]
            logger.debug(
                f"Directory created/verified: {ra_aid_dir_str} with permissions {permissions}"
            )

            # List directory contents for debugging
            dir_contents = os.listdir(ra_aid_dir_str)
            logger.debug(f"Directory contents: {dir_contents}")
        except Exception as e:
            logger.debug(f"Could not check directory details: {str(e)}")

        # Database path for file-based database - use os.path.join for maximum compatibility
        db_path = os.path.join(ra_aid_dir_str, "pk.db")
        logger.debug(f"Database path: {db_path}")

    try:
        # For file-based databases, ensure the file exists or can be created
        if db_path != ":memory:":
            # Check if the database file exists
            db_file_exists = os.path.exists(db_path)
            logger.debug(f"Database file exists check: {db_file_exists}")

            # If the file doesn't exist, try to create an empty file to ensure we have write permissions
            if not db_file_exists:
                try:
                    logger.debug(f"Creating empty database file at: {db_path}")
                    with open(db_path, "w") as f:
                        pass  # Create empty file

                    # Verify the file was created
                    if os.path.exists(db_path):
                        logger.debug("Empty database file created successfully")
                    else:
                        logger.error(f"Failed to create database file at: {db_path}")
                except Exception as e:
                    logger.error(f"Error creating database file: {str(e)}")
                    # Continue anyway, as SQLite might be able to create the file itself

        # Initialize the database connection
        logger.debug(f"Initializing SQLite database at: {db_path}")
        db = peewee.SqliteDatabase(
            db_path,
            pragmas={
                "journal_mode": "wal",  # Write-Ahead Logging for better concurrency
                "foreign_keys": 1,  # Enforce foreign key constraints
                "cache_size": -1024 * 32,  # 32MB cache
            },
        )

        # Always explicitly connect to ensure the connection is established
        if db.is_closed():
            logger.debug("Explicitly connecting to database")
            db.connect()

        # Store the database connection in the contextvar
        db_var.set(db)

        # Store whether this is an in-memory database (for backward compatibility)
        db._is_in_memory = in_memory

        # Verify the database is usable by executing a simple query
        if not in_memory:
            try:
                db.execute_sql("SELECT 1")
                logger.debug("Database connection verified with test query")

                # Check if the database file exists after initialization
                db_file_exists = os.path.exists(db_path)
                db_file_size = os.path.getsize(db_path) if db_file_exists else 0
                logger.debug(
                    f"Database file check after init: exists={db_file_exists}, size={db_file_size} bytes"
                )
            except Exception as e:
                logger.error(f"Database verification failed: {str(e)}")
                # Continue anyway, as this is just a verification step

        # Only show initialization message if it hasn't been shown before
        if not hasattr(db, "_message_shown") or not db._message_shown:
            if in_memory:
                logger.debug("In-memory database connection initialized successfully")
            else:
                logger.debug("Database connection initialized successfully")
            db._message_shown = True

        return db
    except peewee.OperationalError as e:
        logger.error(f"Database Operational Error: {str(e)}")
        raise
    except peewee.DatabaseError as e:
        logger.error(f"Database Error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise


def get_db() -> peewee.SqliteDatabase:
    """
    Get the current database connection.

    If no connection exists, initializes a new one.
    If connection exists but is closed, reopens it.

    Returns:
        peewee.SqliteDatabase: The current database connection
    """
    db = db_var.get()

    if db is None:
        # No database connection exists, initialize one
        # Use the default in-memory mode (False)
        return init_db(in_memory=False)

    # Check if connection is closed and reopen if needed
    if db.is_closed():
        try:
            logger.debug("Attempting to reopen closed database connection")
            db.connect()
            logger.info("Reopened existing database connection")
        except peewee.OperationalError as e:
            logger.error(f"Failed to reopen database connection: {str(e)}")
            # Create a completely new connection
            # First, remove the old connection from the context var
            db_var.set(None)
            # Then initialize a new connection with the same in-memory setting
            in_memory = hasattr(db, "_is_in_memory") and db._is_in_memory
            logger.debug(f"Creating new database connection (in_memory={in_memory})")
            # Create a completely new database object, don't reuse the old one
            return init_db(in_memory=in_memory)

    return db


def close_db() -> None:
    """
    Close the current database connection if it exists.

    Handles various error conditions gracefully.
    """
    db = db_var.get()
    if db is None:
        logger.warning("No database connection to close")
        return

    try:
        if not db.is_closed():
            db.close()
            logger.info("Database connection closed successfully")
        else:
            logger.debug(
                "Database connection was already closed (normal during shutdown)"
            )
    except peewee.DatabaseError as e:
        logger.error(f"Database Error: Failed to close connection: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to close database connection: {str(e)}")
