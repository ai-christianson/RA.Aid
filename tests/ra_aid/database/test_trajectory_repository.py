"""
Tests for the TrajectoryRepository class.
"""

import pytest
import datetime
import json
from unittest.mock import patch

import peewee

from ra_aid.database.connection import DatabaseManager, db_var
from ra_aid.database.models import Trajectory, HumanInput, Session, BaseModel
from ra_aid.database.repositories.trajectory_repository import (
    TrajectoryRepository,
    TrajectoryRepositoryManager,
    get_trajectory_repository,
    trajectory_repo_var
)
from ra_aid.database.pydantic_models import TrajectoryModel


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
def cleanup_repo():
    """Reset the repository contextvar after each test."""
    # Reset before the test
    trajectory_repo_var.set(None)
    
    # Run the test
    yield
    
    # Reset after the test
    trajectory_repo_var.set(None)


@pytest.fixture
def setup_db(cleanup_db):
    """Set up an in-memory database with the necessary tables and patch the BaseModel.Meta.database."""
    # Initialize an in-memory database connection
    with DatabaseManager(in_memory=True) as db:
        # Patch the BaseModel.Meta.database to use our in-memory database
        with patch.object(BaseModel._meta, 'database', db):
            # Create the required tables
            with db.atomic():
                db.create_tables([Trajectory, HumanInput, Session], safe=True)
            
            yield db
            
            # Clean up
            with db.atomic():
                Trajectory.drop_table(safe=True)
                HumanInput.drop_table(safe=True)


def test_create_trajectory_with_token_fields(setup_db):
    """Test creating a trajectory with input_tokens and output_tokens fields."""
    # Set up repository
    repo = TrajectoryRepository(db=setup_db)
    
    # Create a trajectory with token fields
    tool_name = "test_tool"
    tool_parameters = {"param1": "value1", "param2": "value2"}
    input_tokens_value = 100
    output_tokens_value = 50
    cost_value = 0.003
    
    trajectory = repo.create(
        tool_name=tool_name,
        tool_parameters=tool_parameters,
        input_tokens=input_tokens_value,
        output_tokens=output_tokens_value,
        cost=cost_value
    )
    
    # Verify the trajectory was created correctly
    assert trajectory.id is not None
    assert trajectory.tool_name == tool_name
    assert trajectory.input_tokens == input_tokens_value
    assert trajectory.output_tokens == output_tokens_value
    assert trajectory.cost == cost_value
    
    # Verify we can retrieve it from the database using the repository
    trajectory_from_db = repo.get(trajectory.id)
    assert trajectory_from_db.input_tokens == input_tokens_value
    assert trajectory_from_db.output_tokens == output_tokens_value
    assert trajectory_from_db.cost == cost_value


def test_update_trajectory_token_fields(setup_db):
    """Test updating a trajectory's token fields."""
    # Set up repository
    repo = TrajectoryRepository(db=setup_db)
    
    # Create a trajectory
    trajectory = repo.create(
        tool_name="test_tool",
        input_tokens=100,
        output_tokens=50
    )
    
    # Update the token fields
    new_input_tokens = 200
    new_output_tokens = 100
    updated_trajectory = repo.update(
        trajectory.id,
        input_tokens=new_input_tokens,
        output_tokens=new_output_tokens
    )
    
    # Verify the trajectory was updated correctly
    assert updated_trajectory is not None
    assert updated_trajectory.id == trajectory.id
    assert updated_trajectory.input_tokens == new_input_tokens
    assert updated_trajectory.output_tokens == new_output_tokens
    
    # Verify we can retrieve the updated fields from the database
    trajectory_from_db = repo.get(trajectory.id)
    assert trajectory_from_db.input_tokens == new_input_tokens
    assert trajectory_from_db.output_tokens == new_output_tokens


def test_get_parsed_trajectory_includes_token_fields(setup_db):
    """Test that get_parsed_trajectory includes the input_tokens and output_tokens fields."""
    # Set up repository
    repo = TrajectoryRepository(db=setup_db)
    
    # Create a trajectory with token fields
    input_tokens_value = 100
    output_tokens_value = 50
    trajectory = repo.create(
        tool_name="test_tool",
        input_tokens=input_tokens_value,
        output_tokens=output_tokens_value
    )
    
    # Get the parsed trajectory
    parsed_trajectory = repo.get_parsed_trajectory(trajectory.id)
    
    # Verify the token fields are included in the parsed output
    assert parsed_trajectory is not None
    assert "input_tokens" in parsed_trajectory
    assert "output_tokens" in parsed_trajectory
    assert parsed_trajectory["input_tokens"] == input_tokens_value
    assert parsed_trajectory["output_tokens"] == output_tokens_value
