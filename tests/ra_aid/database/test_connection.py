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
    Fixture to clean up database connections and files between tests.
    This fixture:
    1. Closes any open database connection
    2. Resets the contextvar
    3. Cleans up the .ra-aid directory
    """
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
        if ra_aid_dir.exists():
            # Only remove the database file, not the entire directory
            db_file = ra_aid_dir / "pk.db"
            if db_file.exists():
                db_file.unlink()
            # Remove WAL and SHM files if they exist
            wal_file = ra_aid_dir / "pk.db-wal"
            if wal_file.exists():
                wal_file.unlink()
            shm_file = ra_aid_dir / "pk.db-shm"
            if shm_file.exists():
                shm_file.unlink()
    except Exception as e:
        # Log but don't fail if cleanup has issues
        print(f"Cleanup error (non-fatal): {str(e)}")


@pytest.fixture
def mock_logger():
    """Mock the logger to test for output messages."""
    with patch("ra_aid.database.connection.logger") as mock:
        yield mock


class TestInitDb:
    """Tests for the init_db function."""

    def test_init_db_default(self, cleanup_db):
        """Test init_db with default parameters."""
        db = init_db()
        assert isinstance(db, peewee.SqliteDatabase)
        assert not db.is_closed()
        assert hasattr(db, "_is_in_memory")
        assert db._is_in_memory is False
        # Verify the database file was created
        ra_aid_dir = Path(os.getcwd()) / ".ra-aid"
        assert ra_aid_dir.exists()
        assert (ra_aid_dir / "pk.db").exists()

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
