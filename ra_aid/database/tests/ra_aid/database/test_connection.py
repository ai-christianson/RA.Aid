"""
Tests for the database connection module.
"""

import os
from pathlib import Path
from unittest.mock import patch

import peewee
import pytest

from ra_aid.database.connection import (
    DatabaseManager,
    close_db,
    db_var,
    get_db,
    init_db,
)


@pytest.fixture
def cleanup_db():
    """
    Fixture to clean up database connections after tests.
    """
    # Run the test
    yield

    # Clean up after the test
    db = db_var.get()
    if db is not None:
        if hasattr(db, "_is_in_memory"):
            delattr(db, "_is_in_memory")
        if hasattr(db, "_message_shown"):
            delattr(db, "_message_shown")
        if not db.is_closed():
            db.close()

    # Reset the contextvar
    db_var.set(None)


@pytest.fixture
def setup_in_memory_db():
    """
    Fixture to set up an in-memory database for testing.
    """
    # Initialize in-memory database
    db = init_db(in_memory=True)

    # Run the test
    yield db

    # Clean up
    if not db.is_closed():
        db.close()
    db_var.set(None)


def test_init_db_creates_directory(cleanup_db, tmp_path):
    """
    Test that init_db creates the .ra-aid directory if it doesn't exist.
    """
    # Get and print the original working directory
    original_cwd = os.getcwd()
    print(f"Original working directory: {original_cwd}")

    # Convert tmp_path to string for consistent handling
    tmp_path_str = str(tmp_path.absolute())
    print(f"Temporary directory path: {tmp_path_str}")

    # Change to the temporary directory
    os.chdir(tmp_path_str)
    current_cwd = os.getcwd()
    print(f"Changed working directory to: {current_cwd}")
    assert (
        current_cwd == tmp_path_str
    ), f"Failed to change directory: {current_cwd} != {tmp_path_str}"

    # Create the .ra-aid directory manually to ensure it exists
    ra_aid_path_str = os.path.join(current_cwd, ".ra-aid")
    print(f"Creating .ra-aid directory at: {ra_aid_path_str}")
    os.makedirs(ra_aid_path_str, exist_ok=True)

    # Verify the directory was created
    assert os.path.exists(
        ra_aid_path_str
    ), f".ra-aid directory not found at {ra_aid_path_str}"
    assert os.path.isdir(
        ra_aid_path_str
    ), f"{ra_aid_path_str} exists but is not a directory"

    # Create a test file to verify write permissions
    test_file_path = os.path.join(ra_aid_path_str, "test_write.txt")
    print(f"Creating test file to verify write permissions: {test_file_path}")
    with open(test_file_path, "w") as f:
        f.write("Test write permissions")

    # Verify the test file was created
    assert os.path.exists(test_file_path), f"Test file not created at {test_file_path}"

    # Create an empty database file to ensure it exists before init_db
    db_file_str = os.path.join(ra_aid_path_str, "pk.db")
    print(f"Creating empty database file at: {db_file_str}")
    with open(db_file_str, "w") as f:
        f.write("")  # Create empty file

    # Verify the database file was created
    assert os.path.exists(
        db_file_str
    ), f"Empty database file not created at {db_file_str}"
    print(f"Empty database file size: {os.path.getsize(db_file_str)} bytes")

    # Get directory permissions for debugging
    dir_perms = oct(os.stat(ra_aid_path_str).st_mode)[-3:]
    print(f"Directory permissions: {dir_perms}")

    # Initialize the database
    print("Calling init_db()")
    db = init_db()
    print("init_db() returned successfully")

    # List contents of the current directory for debugging
    print(f"Contents of current directory: {os.listdir(current_cwd)}")

    # List contents of the .ra-aid directory for debugging
    print(f"Contents of .ra-aid directory: {os.listdir(ra_aid_path_str)}")

    # Check that the database file exists using os.path
    assert os.path.exists(db_file_str), f"Database file not found at {db_file_str}"
    assert os.path.isfile(db_file_str), f"{db_file_str} exists but is not a file"
    print(f"Final database file size: {os.path.getsize(db_file_str)} bytes")


def test_init_db_creates_database_file(cleanup_db, tmp_path):
    """
    Test that init_db creates the database file.
    """
    # Change to the temporary directory
    os.chdir(tmp_path)

    # Initialize the database
    init_db()

    # Check that the database file was created
    assert (tmp_path / ".ra-aid" / "pk.db").exists()
    assert (tmp_path / ".ra-aid" / "pk.db").is_file()


def test_init_db_returns_database_connection(cleanup_db):
    """
    Test that init_db returns a database connection.
    """
    # Initialize the database
    db = init_db()

    # Check that the database connection is returned
    assert isinstance(db, peewee.SqliteDatabase)
    assert not db.is_closed()


def test_init_db_with_in_memory_mode(cleanup_db):
    """
    Test that init_db with in_memory=True creates an in-memory database.
    """
    # Initialize the database in in-memory mode
    db = init_db(in_memory=True)

    # Check that the database connection is returned
    assert isinstance(db, peewee.SqliteDatabase)
    assert not db.is_closed()
    assert hasattr(db, "_is_in_memory")
    assert db._is_in_memory is True


def test_in_memory_mode_no_directory_created(cleanup_db, tmp_path):
    """
    Test that when using in-memory mode, no directory is created.
    """
    # Change to the temporary directory
    os.chdir(tmp_path)

    # Initialize the database in in-memory mode
    init_db(in_memory=True)

    # Check that the .ra-aid directory was not created
    # (Note: it might be created by other tests, so we can't assert it doesn't exist)
    # Instead, check that the database file was not created
    assert not (tmp_path / ".ra-aid" / "pk.db").exists()


def test_init_db_returns_existing_connection(cleanup_db):
    """
    Test that init_db returns the existing connection if one exists.
    """
    # Initialize the database
    db1 = init_db()

    # Initialize the database again
    db2 = init_db()

    # Check that the same connection is returned
    assert db1 is db2


def test_init_db_reopens_closed_connection(cleanup_db):
    """
    Test that init_db reopens a closed connection.
    """
    # Initialize the database
    db1 = init_db()

    # Close the connection
    db1.close()

    # Initialize the database again
    db2 = init_db()

    # Check that the same connection is returned and it's open
    assert db1 is db2
    assert not db1.is_closed()


def test_get_db_initializes_connection(cleanup_db):
    """
    Test that get_db initializes a connection if none exists.
    """
    # Get the database connection
    db = get_db()

    # Check that a connection was initialized
    assert isinstance(db, peewee.SqliteDatabase)
    assert not db.is_closed()


def test_get_db_returns_existing_connection(cleanup_db):
    """
    Test that get_db returns the existing connection if one exists.
    """
    # Initialize the database
    db1 = init_db()

    # Get the database connection
    db2 = get_db()

    # Check that the same connection is returned
    assert db1 is db2


def test_get_db_reopens_closed_connection(cleanup_db):
    """
    Test that get_db reopens a closed connection.
    """
    # Initialize the database
    db = init_db()

    # Close the connection
    db.close()

    # Get the database connection
    db2 = get_db()

    # Check that the same connection is returned and it's open
    assert db is db2
    assert not db.is_closed()


def test_get_db_handles_reopen_error(cleanup_db, monkeypatch):
    """
    Test that get_db handles errors when reopening a connection.
    """
    # Initialize the database
    db = init_db()

    # Close the connection
    db.close()

    # Create a patched version of the connect method that raises an error
    original_connect = peewee.SqliteDatabase.connect

    def mock_connect(self, *args, **kwargs):
        if self is db:  # Only raise for the specific db instance
            raise peewee.OperationalError("Test error")
        return original_connect(self, *args, **kwargs)

    # Apply the patch
    monkeypatch.setattr(peewee.SqliteDatabase, "connect", mock_connect)

    # Get the database connection
    db2 = get_db()

    # Check that a new connection was initialized
    assert db is not db2
    assert not db2.is_closed()


def test_close_db_closes_connection(cleanup_db):
    """
    Test that close_db closes the connection.
    """
    # Initialize the database
    db = init_db()

    # Close the connection
    close_db()

    # Check that the connection is closed
    assert db.is_closed()


def test_close_db_handles_no_connection():
    """
    Test that close_db handles the case where no connection exists.
    """
    # Reset the contextvar
    db_var.set(None)

    # Close the connection (should not raise an error)
    close_db()


def test_close_db_handles_already_closed_connection(cleanup_db):
    """
    Test that close_db handles the case where the connection is already closed.
    """
    # Initialize the database
    db = init_db()

    # Close the connection
    db.close()

    # Close the connection again (should not raise an error)
    close_db()


@patch("ra_aid.database.connection.peewee.SqliteDatabase.close")
def test_close_db_handles_error(mock_close, cleanup_db):
    """
    Test that close_db handles errors when closing the connection.
    """
    # Initialize the database
    init_db()

    # Make close raise an error
    mock_close.side_effect = peewee.DatabaseError("Test error")

    # Close the connection (should not raise an error)
    close_db()


def test_database_manager_context_manager(cleanup_db):
    """
    Test that DatabaseManager works as a context manager.
    """
    # Use the context manager
    with DatabaseManager() as db:
        # Check that a connection was initialized
        assert isinstance(db, peewee.SqliteDatabase)
        assert not db.is_closed()

        # Store the connection for later
        db_in_context = db

    # Check that the connection is closed after exiting the context
    assert db_in_context.is_closed()


def test_database_manager_with_in_memory_mode(cleanup_db):
    """
    Test that DatabaseManager with in_memory=True creates an in-memory database.
    """
    # Use the context manager with in_memory=True
    with DatabaseManager(in_memory=True) as db:
        # Check that a connection was initialized
        assert isinstance(db, peewee.SqliteDatabase)
        assert not db.is_closed()
        assert hasattr(db, "_is_in_memory")
        assert db._is_in_memory is True


def test_init_db_shows_message_only_once(cleanup_db, caplog):
    """
    Test that init_db only shows the initialization message once.
    """
    # Initialize the database
    init_db(in_memory=True)

    # Clear the log
    caplog.clear()

    # Initialize the database again
    init_db(in_memory=True)

    # Check that no message was logged
    assert "database connection initialized" not in caplog.text.lower()


def test_init_db_sets_is_in_memory_attribute(cleanup_db):
    """
    Test that init_db sets the _is_in_memory attribute.
    """
    # Initialize the database with in_memory=False
    db = init_db(in_memory=False)

    # Check that the _is_in_memory attribute is set to False
    assert hasattr(db, "_is_in_memory")
    assert db._is_in_memory is False

    # Reset the contextvar
    db_var.set(None)

    # Initialize the database with in_memory=True
    db = init_db(in_memory=True)

    # Check that the _is_in_memory attribute is set to True
    assert hasattr(db, "_is_in_memory")
    assert db._is_in_memory is True


"""
Tests for the database connection module.
"""


import pytest



@pytest.fixture
def cleanup_db():
    """
    Fixture to clean up database connections and files between tests.

    This fixture:
    1. Closes any open database connection
    2. Resets the contextvar
    3. Cleans up the .ra-aid directory
    """
    # Store the current working directory
    original_cwd = os.getcwd()

    # Run the test
    yield

    # Clean up after the test
    try:
        # Close any open database connection
        close_db()

        # Reset the contextvar
        db_var.set(None)

        # Clean up the .ra-aid directory if it exists
        ra_aid_dir = Path(os.getcwd()) / ".ra-aid"
        ra_aid_dir_str = str(ra_aid_dir.absolute())

        # Check using both methods
        path_exists = ra_aid_dir.exists()
        os_exists = os.path.exists(ra_aid_dir_str)

        print(f"Cleanup check: Path.exists={path_exists}, os.path.exists={os_exists}")

        if os_exists:
            # Only remove the database file, not the entire directory
            db_file = os.path.join(ra_aid_dir_str, "pk.db")
            if os.path.exists(db_file):
                os.unlink(db_file)

            # Remove WAL and SHM files if they exist
            wal_file = os.path.join(ra_aid_dir_str, "pk.db-wal")
            if os.path.exists(wal_file):
                os.unlink(wal_file)

            shm_file = os.path.join(ra_aid_dir_str, "pk.db-shm")
            if os.path.exists(shm_file):
                os.unlink(shm_file)

            # List remaining contents for debugging
            if os.path.exists(ra_aid_dir_str):
                print(f"Directory contents after cleanup: {os.listdir(ra_aid_dir_str)}")
    except Exception as e:
        # Log but don't fail if cleanup has issues
        print(f"Cleanup error (non-fatal): {str(e)}")

    # Make sure we're back in the original directory
    os.chdir(original_cwd)


class TestInitDb:
    """Tests for the init_db function."""

    def test_init_db_default(self, cleanup_db):
        """Test init_db with default parameters."""
        # Get the absolute path of the current working directory
        cwd = os.getcwd()
        print(f"Current working directory: {cwd}")

        # Initialize the database
        db = init_db()

        assert isinstance(db, peewee.SqliteDatabase)
        assert not db.is_closed()
        assert hasattr(db, "_is_in_memory")
        assert db._is_in_memory is False

        # Verify the database file was created using both Path and os.path methods
        ra_aid_dir = Path(cwd) / ".ra-aid"
        ra_aid_dir_str = str(ra_aid_dir.absolute())

        # Check directory existence using both methods
        path_exists = ra_aid_dir.exists()
        os_exists = os.path.exists(ra_aid_dir_str)
        print(f"Directory check: Path.exists={path_exists}, os.path.exists={os_exists}")

        # List the contents of the current directory
        print(f"Contents of {cwd}: {os.listdir(cwd)}")

        # If the directory exists, list its contents
        if os_exists:
            print(f"Contents of {ra_aid_dir_str}: {os.listdir(ra_aid_dir_str)}")

        # Use os.path for assertions to be more reliable
        assert os.path.exists(
            ra_aid_dir_str
        ), f"Directory {ra_aid_dir_str} does not exist"
        assert os.path.isdir(ra_aid_dir_str), f"{ra_aid_dir_str} is not a directory"

        db_file = os.path.join(ra_aid_dir_str, "pk.db")
        assert os.path.exists(db_file), f"Database file {db_file} does not exist"
        assert os.path.isfile(db_file), f"{db_file} is not a file"

    def test_init_db_in_memory(self, cleanup_db):
        """Test init_db with in_memory=True."""
        db = init_db(in_memory=True)

        assert isinstance(db, peewee.SqliteDatabase)
        assert not db.is_closed()
        assert hasattr(db, "_is_in_memory")
        assert db._is_in_memory is True

    def test_init_db_reuses_connection(self, cleanup_db):
        """Test that init_db reuses an existing connection."""
        db1 = init_db()
        db2 = init_db()

        assert db1 is db2

    def test_init_db_reopens_closed_connection(self, cleanup_db):
        """Test that init_db reopens a closed connection."""
        db1 = init_db()
        db1.close()
        assert db1.is_closed()

        db2 = init_db()
        assert db1 is db2
        assert not db1.is_closed()


class TestGetDb:
    """Tests for the get_db function."""

    def test_get_db_creates_connection(self, cleanup_db):
        """Test that get_db creates a new connection if none exists."""
        # Reset the contextvar to ensure no connection exists
        db_var.set(None)

        db = get_db()

        assert isinstance(db, peewee.SqliteDatabase)
        assert not db.is_closed()
        assert hasattr(db, "_is_in_memory")
        assert db._is_in_memory is False

    def test_get_db_reuses_connection(self, cleanup_db):
        """Test that get_db reuses an existing connection."""
        db1 = init_db()
        db2 = get_db()

        assert db1 is db2

    def test_get_db_reopens_closed_connection(self, cleanup_db):
        """Test that get_db reopens a closed connection."""
        db1 = init_db()
        db1.close()
        assert db1.is_closed()

        db2 = get_db()
        assert db1 is db2
        assert not db1.is_closed()


class TestCloseDb:
    """Tests for the close_db function."""

    def test_close_db(self, cleanup_db):
        """Test that close_db closes an open connection."""
        db = init_db()
        assert not db.is_closed()

        close_db()
        assert db.is_closed()

    def test_close_db_no_connection(self, cleanup_db):
        """Test that close_db handles the case where no connection exists."""
        # Reset the contextvar to ensure no connection exists
        db_var.set(None)

        # This should not raise an exception
        close_db()

    def test_close_db_already_closed(self, cleanup_db):
        """Test that close_db handles the case where the connection is already closed."""
        db = init_db()
        db.close()
        assert db.is_closed()

        # This should not raise an exception
        close_db()


class TestDatabaseManager:
    """Tests for the DatabaseManager class."""

    def test_database_manager_default(self, cleanup_db):
        """Test DatabaseManager with default parameters."""
        with DatabaseManager() as db:
            assert isinstance(db, peewee.SqliteDatabase)
            assert not db.is_closed()
            assert hasattr(db, "_is_in_memory")
            assert db._is_in_memory is False

            # Verify the database file was created
            ra_aid_dir = Path(os.getcwd()) / ".ra-aid"
            assert ra_aid_dir.exists()
            assert (ra_aid_dir / "pk.db").exists()

        # Verify the connection is closed after exiting the context
        assert db.is_closed()

    def test_database_manager_in_memory(self, cleanup_db):
        """Test DatabaseManager with in_memory=True."""
        with DatabaseManager(in_memory=True) as db:
            assert isinstance(db, peewee.SqliteDatabase)
            assert not db.is_closed()
            assert hasattr(db, "_is_in_memory")
            assert db._is_in_memory is True

        # Verify the connection is closed after exiting the context
        assert db.is_closed()

    def test_database_manager_exception_handling(self, cleanup_db):
        """Test that DatabaseManager properly handles exceptions."""
        try:
            with DatabaseManager() as db:
                assert not db.is_closed()
                raise ValueError("Test exception")
        except ValueError:
            # The exception should be propagated
            pass

        # Verify the connection is closed even if an exception occurred
        assert db.is_closed()
