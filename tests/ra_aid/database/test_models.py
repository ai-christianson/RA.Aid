"""
Tests for the database models module.
"""

from unittest.mock import patch

import peewee
import pytest

from ra_aid.database.connection import db_var, init_db
from ra_aid.database.models import BaseModel


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
def setup_test_model(cleanup_db):
    """Set up a test model class for testing."""
    # Initialize an in-memory database connection
    db = init_db(in_memory=True)

    # Define a test model
    class TestModel(BaseModel):
        name = peewee.CharField()
        value = peewee.IntegerField(null=True)

        class Meta:
            database = db

    # Create the table
    with db.atomic():
        db.create_tables([TestModel], safe=True)

    yield TestModel

    # Clean up
    with db.atomic():
        TestModel.drop_table(safe=True)


def test_base_model_save_updates_timestamps(setup_test_model):
    """Test that saving a model updates the timestamps."""
    TestModel = setup_test_model

    # Create a new instance
    instance = TestModel(name="test", value=42)
    instance.save()

    # Check that created_at and updated_at are set
    assert instance.created_at is not None
    assert instance.updated_at is not None

    # Store the original timestamps
    original_created_at = instance.created_at
    original_updated_at = instance.updated_at

    # Wait a moment to ensure timestamps would be different
    import time

    time.sleep(0.001)

    # Update the instance
    instance.value = 43
    instance.save()

    # Check that updated_at changed but created_at didn't
    assert instance.created_at == original_created_at
    assert instance.updated_at != original_updated_at


def test_base_model_get_or_create(setup_test_model):
    """Test the get_or_create method."""
    TestModel = setup_test_model

    # First call should create a new instance
    instance, created = TestModel.get_or_create(name="test", value=42)
    assert created is True
    assert instance.name == "test"
    assert instance.value == 42

    # Second call with same parameters should return existing instance
    instance2, created2 = TestModel.get_or_create(name="test", value=42)
    assert created2 is False
    assert instance2.id == instance.id

    # Call with different parameters should create a new instance
    instance3, created3 = TestModel.get_or_create(name="test2", value=43)
    assert created3 is True
    assert instance3.id != instance.id


@patch("ra_aid.database.models.logger")
def test_base_model_get_or_create_handles_errors(mock_logger, setup_test_model):
    """Test that get_or_create handles database errors properly."""
    TestModel = setup_test_model

    # Mock the parent get_or_create to raise a DatabaseError
    with patch(
        "peewee.Model.get_or_create", side_effect=peewee.DatabaseError("Test error")
    ):
        # Call should raise the error
        with pytest.raises(peewee.DatabaseError):
            TestModel.get_or_create(name="test")

        # Verify error was logged
        mock_logger.error.assert_called_with("Failed in get_or_create: Test error")
