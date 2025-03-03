"""
Tests for the database connection module.

This file tests the database connection functionality using pytest's fixtures
for proper test isolation.
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
    Fixture to clean up database connections between tests.
    
    This ensures that we don't leak database connections between tests
    and that the db_var contextvar is reset.
    """
    # Run the test
    yield
    
    # Clean up after the test
    db = db_var.get()
    if db is not None:
        # Clean up attributes we may have added
        if hasattr(db, "_is_in_memory"):
            delattr(db, "_is_in_memory")
        if hasattr(db, "_message_shown"):
            delattr(db, "_message_shown")
        
        # Close the connection if it's open
        if not db.is_closed():
            db.close()
    
    # Reset the contextvar
    db_var.set(None)


@pytest.fixture
def db_path_mock(tmp_path, monkeypatch):
    """
    Fixture to mock os.getcwd() to return a temporary directory path.
    
    This ensures that all database operations use the temporary directory
    and never touch the actual current working directory.
    """
    original_cwd = os.getcwd()
    tmp_path_str = str(tmp_path.absolute())
    
    # Create the .ra-aid directory in the temporary path
    ra_aid_dir = tmp_path / ".ra-aid"
    ra_aid_dir.mkdir(exist_ok=True)
    
    # Mock os.getcwd() to return the temporary directory
    monkeypatch.setattr(os, "getcwd", lambda: tmp_path_str)
    
    yield tmp_path
    
    # Ensure we're back to the original directory after the test
    os.chdir(original_cwd)


class TestInitDb:
    """Tests for the init_db function."""
    
    def test_init_db_in_memory(self, cleanup_db):
        """Test init_db with in_memory=True."""
        # Reset the contextvar to ensure a fresh start
        db_var.set(None)
        db = init_db(in_memory=True)
        
        assert isinstance(db, peewee.SqliteDatabase)
        assert not db.is_closed()
        assert hasattr(db, "_is_in_memory")
        assert db._is_in_memory is True
    
    def test_init_db_creates_directory(self, cleanup_db, db_path_mock):
        """Test that init_db creates the .ra-aid directory if it doesn't exist."""
        # Remove the .ra-aid directory to test creation
        ra_aid_dir = db_path_mock / ".ra-aid"
        if ra_aid_dir.exists():
            for item in ra_aid_dir.iterdir():
                if item.is_file():
                    item.unlink()
            ra_aid_dir.rmdir()
        
        # Initialize the database
        db = init_db()
        
        # Check that the directory was created
        assert ra_aid_dir.exists()
        assert ra_aid_dir.is_dir()
        assert isinstance(db, peewee.SqliteDatabase)
        assert not db.is_closed()
        assert hasattr(db, "_is_in_memory")
        assert db._is_in_memory is False
    
    def test_init_db_creates_database_file(self, cleanup_db, db_path_mock):
        """Test that init_db creates the database file."""
        # Initialize the database
        init_db()
        
        # Check that the database file was created
        assert (db_path_mock / ".ra-aid" / "pk.db").exists()
        assert (db_path_mock / ".ra-aid" / "pk.db").is_file()
    
    def test_init_db_reuses_connection(self, cleanup_db):
        """Test that init_db reuses an existing connection."""
        # Reset the contextvar to ensure a fresh start
        db_var.set(None)
        
        # Use in_memory=True for this test to avoid touching the filesystem
        db1 = init_db(in_memory=True)
        db2 = init_db(in_memory=True)
        
        assert db1 is db2
    
    def test_init_db_reopens_closed_connection(self, cleanup_db):
        """Test that init_db reopens a closed connection."""
        # Reset the contextvar to ensure a fresh start
        db_var.set(None)
        
        # Use in_memory=True for this test to avoid touching the filesystem
        db1 = init_db(in_memory=True)
        db1.close()
        assert db1.is_closed()
        
        db2 = init_db(in_memory=True)
        assert db1 is db2
        assert not db1.is_closed()
    
    def test_in_memory_mode_no_directory_created(self, cleanup_db, db_path_mock):
        """Test that when using in_memory mode, no database file is created."""
        # Initialize the database in in-memory mode
        init_db(in_memory=True)
        
        # Check that the database file was not created
        assert not (db_path_mock / ".ra-aid" / "pk.db").exists()
    
    def test_init_db_sets_is_in_memory_attribute(self, cleanup_db):
        """Test that init_db sets the _is_in_memory attribute."""
        # Test with in_memory=True
        db = init_db(in_memory=True)
        assert hasattr(db, "_is_in_memory")
        assert db._is_in_memory is True
        
        # Reset the contextvar
        db_var.set(None)
        
        # Test with in_memory=False, but use a mocked directory
        with patch("os.getcwd") as mock_getcwd:
            temp_dir = Path("/tmp/testdb")
            mock_getcwd.return_value = str(temp_dir)
            
            # Mock os.path.exists and os.makedirs to avoid filesystem operations
            with patch("os.path.exists", return_value=True):
                with patch("os.makedirs"):
                    with patch("os.path.isdir", return_value=True):
                        with patch.object(peewee.SqliteDatabase, "connect"):
                            with patch.object(peewee.SqliteDatabase, "execute_sql"):
                                db = init_db(in_memory=False)
                                assert hasattr(db, "_is_in_memory")
                                assert db._is_in_memory is False


class TestGetDb:
    """Tests for the get_db function."""
    
    def test_get_db_initializes_connection(self, cleanup_db):
        """Test that get_db initializes a connection if none exists."""
        # Reset the contextvar to ensure no connection exists
        db_var.set(None)
        
        # Use a patch to avoid touching the filesystem
        with patch("ra_aid.database.connection.init_db") as mock_init_db:
            mock_db = peewee.SqliteDatabase(":memory:")
            mock_db._is_in_memory = False
            mock_init_db.return_value = mock_db
            
            db = get_db()
            
            mock_init_db.assert_called_once_with(in_memory=False, base_dir=None)
            assert db is mock_db
    
    def test_get_db_returns_existing_connection(self, cleanup_db):
        """Test that get_db returns the existing connection if one exists."""
        # Reset the contextvar to ensure a fresh start
        db_var.set(None)
        
        # Use in_memory=True for this test to avoid touching the filesystem
        db1 = init_db(in_memory=True)
        db2 = get_db()
        
        assert db1 is db2
    
    def test_get_db_reopens_closed_connection(self, cleanup_db):
        """Test that get_db reopens a closed connection."""
        # Reset the contextvar to ensure a fresh start
        db_var.set(None)
        
        # Use in_memory=True for this test to avoid touching the filesystem
        db1 = init_db(in_memory=True)
        db1.close()
        assert db1.is_closed()
        
        db2 = get_db()
        assert db1 is db2
        assert not db1.is_closed()
    
    def test_get_db_handles_reopen_error(self, cleanup_db, monkeypatch):
        """Test that get_db handles errors when reopening a connection."""
        # Reset the contextvar to ensure a fresh start
        db_var.set(None)
        
        # Use in_memory=True for this test to avoid touching the filesystem
        db = init_db(in_memory=True)
        
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
        
        # Get the database connection - this should create a new one
        db2 = get_db()
        
        # Check that a new connection was initialized
        assert db is not db2
        assert not db2.is_closed()
        assert hasattr(db2, "_is_in_memory")
        assert db2._is_in_memory is True  # Should preserve the in_memory setting


class TestCloseDb:
    """Tests for the close_db function."""
    
    def test_close_db_closes_connection(self, cleanup_db):
        """Test that close_db closes the connection."""
        # Use in_memory=True for this test to avoid touching the filesystem
        db = init_db(in_memory=True)
        
        # Close the connection
        close_db()
        
        # Check that the connection is closed
        assert db.is_closed()
    
    def test_close_db_handles_no_connection(self):
        """Test that close_db handles the case where no connection exists."""
        # Reset the contextvar
        db_var.set(None)
        
        # Close the connection (should not raise an error)
        close_db()
    
    def test_close_db_handles_already_closed_connection(self, cleanup_db):
        """Test that close_db handles the case where the connection is already closed."""
        # Use in_memory=True for this test to avoid touching the filesystem
        db = init_db(in_memory=True)
        
        # Close the connection
        db.close()
        
        # Close the connection again (should not raise an error)
        close_db()
    
    @patch("ra_aid.database.connection.peewee.SqliteDatabase.close")
    def test_close_db_handles_error(self, mock_close, cleanup_db):
        """Test that close_db handles errors when closing the connection."""
        # Use in_memory=True for this test to avoid touching the filesystem
        init_db(in_memory=True)
        
        # Make close raise an error
        mock_close.side_effect = peewee.DatabaseError("Test error")
        
        # Close the connection (should not raise an error)
        close_db()


class TestDatabaseManager:
    """Tests for the DatabaseManager class."""
    
    def test_database_manager_context_manager_in_memory(self, cleanup_db):
        """Test that DatabaseManager works as a context manager with in_memory=True."""
        # Use in_memory=True for this test to avoid touching the filesystem
        with DatabaseManager(in_memory=True) as db:
            # Check that a connection was initialized
            assert isinstance(db, peewee.SqliteDatabase)
            assert not db.is_closed()
            assert hasattr(db, "_is_in_memory")
            assert db._is_in_memory is True
            
            # Store the connection for later
            db_in_context = db
        
        # Check that the connection is closed after exiting the context
        assert db_in_context.is_closed()
    
    def test_database_manager_context_manager_physical_file(self, cleanup_db, db_path_mock):
        """Test that DatabaseManager works as a context manager with a physical file."""
        with DatabaseManager(in_memory=False) as db:
            # Check that a connection was initialized
            assert isinstance(db, peewee.SqliteDatabase)
            assert not db.is_closed()
            assert hasattr(db, "_is_in_memory")
            assert db._is_in_memory is False
            
            # Check that the database file was created
            assert (db_path_mock / ".ra-aid" / "pk.db").exists()
            assert (db_path_mock / ".ra-aid" / "pk.db").is_file()
            
            # Store the connection for later
            db_in_context = db
        
        # Check that the connection is closed after exiting the context
        assert db_in_context.is_closed()
    
    def test_database_manager_exception_handling(self, cleanup_db):
        """Test that DatabaseManager properly handles exceptions."""
        # Use in_memory=True for this test to avoid touching the filesystem
        try:
            with DatabaseManager(in_memory=True) as db:
                assert not db.is_closed()
                raise ValueError("Test exception")
        except ValueError:
            # The exception should be propagated
            pass
        
        # Verify the connection is closed even if an exception occurred
        assert db.is_closed()


def test_init_db_shows_message_only_once(cleanup_db, caplog):
    """Test that init_db only shows the initialization message once."""
    # Reset the contextvar to ensure a fresh start
    db_var.set(None)
    
    # Use in_memory=True for this test to avoid touching the filesystem
    init_db(in_memory=True)
    
    # Clear the log
    caplog.clear()
    
    # Initialize the database again
    init_db(in_memory=True)
    
    # Check that no message was logged
    assert "database connection initialized" not in caplog.text.lower()