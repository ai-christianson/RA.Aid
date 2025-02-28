"""
Tests for the database utils module.
"""

from unittest.mock import MagicMock, patch

import peewee
import pytest

from ra_aid.database.connection import db_var, init_db
from ra_aid.database.models import BaseModel
from ra_aid.database.utils import ensure_tables_created, get_model_count, truncate_table


@pytest.fixture
def cleanup_db():
    """Reset the database contextvar and connection state after each test."""
    # Reset before the test
    db = db_var.get()
    if db is not None:
        try:
            if not db.is_closed():
                db.close()
        except Exception:
            # Ignore errors when closing the database
            pass
    db_var.set(None)

    # Run the test
    yield

    # Reset after the test
    db = db_var.get()
    if db is not None:
        try:
            if not db.is_closed():
                db.close()
        except Exception:
            # Ignore errors when closing the database
            pass
    db_var.set(None)


@pytest.fixture
def mock_logger():
    """Mock the logger to test for output messages."""
    with patch("ra_aid.database.utils.logger") as mock:
        yield mock


@pytest.fixture
def setup_test_model(cleanup_db):
    """Set up a test model for database tests."""
    # Initialize the database in memory
    db = init_db(in_memory=True)

    # Define a test model class
    class TestModel(BaseModel):
        name = peewee.CharField(max_length=100)
        value = peewee.IntegerField(default=0)

        class Meta:
            database = db

    # Create the test table in a transaction
    with db.atomic():
        db.create_tables([TestModel], safe=True)

    # Yield control to the test
    yield TestModel

    # Clean up: drop the test table
    with db.atomic():
        db.drop_tables([TestModel], safe=True)


def test_ensure_tables_created_with_models(cleanup_db, mock_logger):
    """Test ensure_tables_created with explicit models."""
    # Initialize the database in memory
    db = init_db(in_memory=True)

    # Define a test model that uses this database
    class TestModel(BaseModel):
        name = peewee.CharField(max_length=100)
        value = peewee.IntegerField(default=0)

        class Meta:
            database = db

    # Call ensure_tables_created with explicit models
    ensure_tables_created([TestModel])

    # Verify success message was logged
    mock_logger.info.assert_called_with("Successfully created tables for 1 models")

    # Verify the table exists by trying to use it
    TestModel.create(name="test", value=42)
    count = TestModel.select().count()
    assert count == 1


@patch("ra_aid.database.utils.get_db")
def test_ensure_tables_created_database_error(
    mock_get_db, setup_test_model, cleanup_db, mock_logger
):
    """Test ensure_tables_created handles database errors."""
    # Get the TestModel class from the fixture
    TestModel = setup_test_model

    # Create a mock database with a create_tables method that raises an error
    mock_db = MagicMock()
    mock_db.atomic.return_value.__enter__.return_value = None
    mock_db.atomic.return_value.__exit__.return_value = None
    mock_db.create_tables.side_effect = peewee.DatabaseError("Test database error")

    # Configure get_db to return our mock
    mock_get_db.return_value = mock_db

    # Call ensure_tables_created and expect an exception
    with pytest.raises(peewee.DatabaseError):
        ensure_tables_created([TestModel])

    # Verify error message was logged
    mock_logger.error.assert_called_with(
        "Database Error: Failed to create tables: Test database error"
    )


def test_get_model_count(setup_test_model, mock_logger):
    """Test get_model_count returns the correct count."""
    # Get the TestModel class from the fixture
    TestModel = setup_test_model

    # First ensure the table is empty
    TestModel.delete().execute()

    # Create some test records
    TestModel.create(name="test1", value=1)
    TestModel.create(name="test2", value=2)

    # Call get_model_count
    count = get_model_count(TestModel)

    # Verify the count is correct
    assert count == 2


@patch("peewee.ModelSelect.count")
def test_get_model_count_database_error(mock_count, setup_test_model, mock_logger):
    """Test get_model_count handles database errors."""
    # Get the TestModel class from the fixture
    TestModel = setup_test_model

    # Configure the mock to raise a DatabaseError
    mock_count.side_effect = peewee.DatabaseError("Test count error")

    # Call get_model_count
    count = get_model_count(TestModel)

    # Verify error message was logged
    mock_logger.error.assert_called_with(
        "Database Error: Failed to count records: Test count error"
    )

    # Verify the function returns 0 on error
    assert count == 0


def test_truncate_table(setup_test_model, mock_logger):
    """Test truncate_table deletes all records."""
    # Get the TestModel class from the fixture
    TestModel = setup_test_model

    # Create some test records
    TestModel.create(name="test1", value=1)
    TestModel.create(name="test2", value=2)

    # Verify records exist
    assert TestModel.select().count() == 2

    # Call truncate_table
    truncate_table(TestModel)

    # Verify success message was logged
    mock_logger.info.assert_called_with(
        f"Successfully truncated table for {TestModel.__name__}"
    )

    # Verify all records were deleted
    assert TestModel.select().count() == 0


@patch("ra_aid.database.models.BaseModel.delete")
def test_truncate_table_database_error(mock_delete, setup_test_model, mock_logger):
    """Test truncate_table handles database errors."""
    # Get the TestModel class from the fixture
    TestModel = setup_test_model

    # Create a test record
    TestModel.create(name="test", value=42)

    # Configure the mock to return a mock query with execute that raises an error
    mock_query = MagicMock()
    mock_query.execute.side_effect = peewee.DatabaseError("Test delete error")
    mock_delete.return_value = mock_query

    # Call truncate_table and expect an exception
    with pytest.raises(peewee.DatabaseError):
        truncate_table(TestModel)

    # Verify error message was logged
    mock_logger.error.assert_called_with(
        "Database Error: Failed to truncate table: Test delete error"
    )
