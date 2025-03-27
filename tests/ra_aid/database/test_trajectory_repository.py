"""
Tests for the TrajectoryRepository class.
"""

import pytest
import json
import logging
from unittest.mock import patch, MagicMock, call


from ra_aid.database.connection import DatabaseManager, db_var
from ra_aid.database.models import Trajectory, HumanInput, Session, BaseModel
from ra_aid.database.repositories.trajectory_repository import (
    TrajectoryRepository,
    TrajectoryRepositoryManager,
    get_trajectory_repository,
    trajectory_repo_var,
)
from ra_aid.database.pydantic_models import TrajectoryModel

# Conftest fixture for mocking session repository
# from tests.conftest import mock_session_repository # Assuming this exists

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
    # No need to clear class hooks anymore

    # Run the test
    yield

    # Reset after the test
    trajectory_repo_var.set(None)
    # No need to clear class hooks anymore


@pytest.fixture
def setup_db(cleanup_db):
    """Set up an in-memory database with the necessary tables and patch the BaseModel.Meta.database."""
    # Initialize an in-memory database connection
    with DatabaseManager(in_memory=True) as db:
        # Patch the BaseModel.Meta.database to use our in-memory database
        with patch.object(BaseModel._meta, "database", db):
            # Create the required tables
            with db.atomic():
                db.create_tables([Trajectory, HumanInput, Session], safe=True)

                # Create a test session record
                Session.create(id=1, name="Test Session")

            yield db

            # Clean up
            with db.atomic():
                Trajectory.drop_table(safe=True)
                HumanInput.drop_table(safe=True)
                Session.drop_table(safe=True)


@pytest.fixture
def sample_human_input(setup_db):
    """Create a sample human input in the database."""
    return HumanInput.create(content="Test human input", source="test")


@pytest.fixture
def test_tool_parameters():
    """Return test tool parameters."""
    return {
        "pattern": "test pattern",
        "file_path": "/path/to/file",
        "options": {"case_sensitive": True, "whole_words": False},
    }

@pytest.fixture
def test_tool_parameters_with_include():
    """Return test tool parameters including include_paths."""
    return {
        "pattern": "test pattern",
        "file_path": "/path/to/file",
        "options": {"case_sensitive": True, "whole_words": False},
        "include_paths": ["src/", "tests/"],
        "fixed_string": False # Added fixed_string to match create call
    }


@pytest.fixture
def test_tool_result():
    """Return test tool result."""
    return {
        "matches": [
            {"line": 10, "content": "This is a test pattern"},
            {"line": 20, "content": "Another test pattern here"},
        ],
        "total_matches": 2,
        "execution_time": 0.5,
    }


@pytest.fixture
def test_step_data():
    """Return test step data for UI rendering."""
    return {
        "display_type": "text",
        "content": "Tool execution results",
        "highlights": [{"start": 10, "end": 15, "color": "red"}],
    }


@pytest.fixture
def sample_trajectory(
    setup_db, sample_human_input, test_tool_parameters, test_tool_result, test_step_data
):
    """Create a sample trajectory in the database."""
    return Trajectory.create(
        human_input=sample_human_input,
        session=1,
        tool_name="ripgrep_search",
        tool_parameters=json.dumps(test_tool_parameters),
        tool_result=json.dumps(test_tool_result),
        step_data=json.dumps(test_step_data),
        record_type="tool_execution",
        current_cost=0,
        input_tokens=0,
        output_tokens=0,
        is_error=False,
    )

# Mock session repository fixture (replace with your actual conftest setup if different)
@pytest.fixture
def mock_session_repository():
    mock_repo = MagicMock()
    mock_session_record = MagicMock()
    mock_session_record.get_id.return_value = 1 # Example session ID
    mock_repo.get_current_session_record.return_value = mock_session_record
    with patch("ra_aid.database.repositories.trajectory_repository.get_session_repository", return_value=mock_repo):
        yield mock_repo

def test_create_trajectory(
    setup_db,
    cleanup_repo, # Use cleanup_repo fixture
    sample_human_input,
    test_tool_parameters,
    test_tool_result,
    test_step_data,
    mock_session_repository,
):
    """Test creating a trajectory with all fields."""
    repo = TrajectoryRepository(db=setup_db)

    trajectory = repo.create(
        tool_name="ripgrep_search",
        tool_parameters=test_tool_parameters,
        tool_result=test_tool_result,
        step_data=test_step_data,
        record_type="tool_execution",
        human_input_id=sample_human_input.id,
        current_cost=0.001,
        input_tokens=200,
        output_tokens=100,
    )

    # Verify type is TrajectoryModel, not Trajectory
    assert isinstance(trajectory, TrajectoryModel)

    # Verify the trajectory was created correctly
    assert trajectory.id is not None
    assert trajectory.tool_name == "ripgrep_search"

    # Verify the JSON fields are dictionaries, not strings
    assert isinstance(trajectory.tool_parameters, dict)
    assert isinstance(trajectory.tool_result, dict)
    assert isinstance(trajectory.step_data, dict)

    # Verify the nested structure of tool parameters
    assert trajectory.tool_parameters["options"]["case_sensitive"] == True
    assert trajectory.tool_result["total_matches"] == 2
    assert trajectory.step_data["highlights"][0]["color"] == "red"
    assert trajectory.current_cost == 0.001
    assert trajectory.input_tokens == 200
    assert trajectory.output_tokens == 100

    # Verify foreign key reference
    assert trajectory.human_input_id == sample_human_input.id


def test_create_trajectory_with_include_paths(
    setup_db,
    cleanup_repo,
    sample_human_input,
    test_tool_parameters_with_include,
    test_tool_result,
    test_step_data,
    mock_session_repository,
):
    """Test creating a trajectory where tool_parameters includes include_paths."""
    repo = TrajectoryRepository(db=setup_db)

    trajectory = repo.create(
        tool_name="ripgrep_search",
        tool_parameters=test_tool_parameters_with_include,
        tool_result=test_tool_result,
        step_data=test_step_data,
        record_type="tool_execution",
        human_input_id=sample_human_input.id,
    )

    assert isinstance(trajectory, TrajectoryModel)
    assert trajectory.id is not None
    assert trajectory.tool_name == "ripgrep_search"
    assert isinstance(trajectory.tool_parameters, dict)

    # Verify include_paths is present and correct
    assert "include_paths" in trajectory.tool_parameters
    assert trajectory.tool_parameters["include_paths"] == ["src/", "tests/"]
    assert trajectory.tool_parameters["fixed_string"] is False

    # Verify other parts are still correct
    assert trajectory.tool_parameters["options"]["case_sensitive"] == True
    assert trajectory.human_input_id == sample_human_input.id


def test_create_trajectory_minimal(setup_db, cleanup_repo, mock_session_repository): # Use cleanup_repo fixture
    """Test creating a trajectory with minimal fields."""
    # Set up repository
    repo = TrajectoryRepository(db=setup_db)

    # Create a trajectory with minimal fields
    trajectory = repo.create(tool_name="simple_tool")

    # Verify type is TrajectoryModel, not Trajectory
    assert isinstance(trajectory, TrajectoryModel)

    # Verify the trajectory was created correctly
    assert trajectory.id is not None
    assert trajectory.tool_name == "simple_tool"

    # Verify optional fields are None or default
    assert trajectory.tool_parameters is None
    assert trajectory.tool_result is None
    assert trajectory.step_data is None
    assert trajectory.human_input_id is None
    assert trajectory.current_cost is None
    assert trajectory.output_tokens is None
    assert trajectory.input_tokens is None
    assert trajectory.is_error is False
    assert trajectory.session_id == 1 # Assuming mock returns session 1


def test_get_trajectory(
    setup_db, sample_trajectory, test_tool_parameters, test_tool_result, test_step_data
):
    """Test retrieving a trajectory by ID."""
    # Set up repository
    repo = TrajectoryRepository(db=setup_db)

    # Get the trajectory by ID
    trajectory = repo.get(sample_trajectory.id)

    # Verify type is TrajectoryModel, not Trajectory
    assert isinstance(trajectory, TrajectoryModel)

    # Verify the retrieved trajectory matches the original
    assert trajectory.id == sample_trajectory.id
    assert trajectory.tool_name == sample_trajectory.tool_name

    # Verify the JSON fields are dictionaries, not strings
    assert isinstance(trajectory.tool_parameters, dict)
    assert isinstance(trajectory.tool_result, dict)
    assert isinstance(trajectory.step_data, dict)

    # Verify the content of JSON fields
    assert trajectory.tool_parameters == test_tool_parameters
    assert trajectory.tool_result == test_tool_result
    assert trajectory.step_data == test_step_data

    # Verify non-existent trajectory returns None
    non_existent_trajectory = repo.get(999)
    assert non_existent_trajectory is None


def test_update_trajectory(setup_db, sample_trajectory):
    """Test updating a trajectory."""
    # Set up repository
    repo = TrajectoryRepository(db=setup_db)

    # New data for update
    new_tool_result = {
        "matches": [{"line": 15, "content": "Updated test pattern"}],
        "total_matches": 1,
        "execution_time": 0.3,
    }

    new_step_data = {
        "display_type": "html",
        "content": "Updated UI rendering",
        "highlights": [],
    }

    # Update the trajectory
    updated_trajectory = repo.update(
        trajectory_id=sample_trajectory.id,
        tool_result=new_tool_result,
        step_data=new_step_data,
        current_cost=0.002,
        input_tokens=2000,
        output_tokens=300,
        is_error=True,
        error_message="Test error",
        error_type="TestErrorType",
        error_details="Detailed error information",
    )

    # Verify type is TrajectoryModel, not Trajectory
    assert isinstance(updated_trajectory, TrajectoryModel)

    # Verify the fields were updated
    assert updated_trajectory.tool_result == new_tool_result
    assert updated_trajectory.step_data == new_step_data
    assert updated_trajectory.current_cost == 0.002
    assert updated_trajectory.input_tokens == 2000
    assert updated_trajectory.output_tokens == 300
    assert updated_trajectory.is_error is True
    assert updated_trajectory.error_message == "Test error"
    assert updated_trajectory.error_type == "TestErrorType"
    assert updated_trajectory.error_details == "Detailed error information"

    # Original tool parameters should not change
    # We need to parse the JSON string from the Peewee object for comparison
    original_params = json.loads(sample_trajectory.tool_parameters)
    assert updated_trajectory.tool_parameters == original_params

    # Verify updating a non-existent trajectory returns None
    non_existent_update = repo.update(trajectory_id=999, current_cost=0.005)
    assert non_existent_update is None


def test_delete_trajectory(setup_db, sample_trajectory):
    """Test deleting a trajectory."""
    # Set up repository
    repo = TrajectoryRepository(db=setup_db)

    # Verify the trajectory exists
    assert repo.get(sample_trajectory.id) is not None

    # Delete the trajectory
    result = repo.delete(sample_trajectory.id)

    # Verify the trajectory was deleted
    assert result is True
    assert repo.get(sample_trajectory.id) is None

    # Verify deleting a non-existent trajectory returns False
    result = repo.delete(999)
    assert result is False


def test_get_all_trajectories(setup_db, sample_human_input, mock_session_repository, cleanup_repo): # Use cleanup_repo fixture
    """Test retrieving all trajectories."""
    # Set up repository
    repo = TrajectoryRepository(db=setup_db)

    # Create multiple trajectories
    for i in range(3):
        repo.create(
            tool_name=f"tool_{i}",
            tool_parameters={"index": i},
            human_input_id=sample_human_input.id,
        )

    # Get all trajectories
    trajectories = repo.get_all()

    # Verify we got a dictionary of TrajectoryModel objects
    assert len(trajectories) == 3
    for trajectory_id, trajectory in trajectories.items():
        assert isinstance(trajectory, TrajectoryModel)
        assert isinstance(trajectory.tool_parameters, dict)

    # Verify the trajectories have the correct tool names
    tool_names = {trajectory.tool_name for trajectory in trajectories.values()}
    assert "tool_0" in tool_names
    assert "tool_1" in tool_names
    assert "tool_2" in tool_names


def test_get_trajectories_by_human_input(
    setup_db, sample_human_input, mock_session_repository, cleanup_repo # Use cleanup_repo fixture
):
    """Test retrieving trajectories by human input ID."""
    # Set up repository
    repo = TrajectoryRepository(db=setup_db)

    # Create another human input
    other_human_input = HumanInput.create(content="Another human input", source="test")

    # Create trajectories for both human inputs
    for i in range(2):
        repo.create(tool_name=f"tool_1_{i}", human_input_id=sample_human_input.id)

    for i in range(3):
        repo.create(tool_name=f"tool_2_{i}", human_input_id=other_human_input.id)

    # Get trajectories for the first human input
    trajectories = repo.get_trajectories_by_human_input(sample_human_input.id)

    # Verify we got a list of TrajectoryModel objects for the first human input
    assert len(trajectories) == 2
    for trajectory in trajectories:
        assert isinstance(trajectory, TrajectoryModel)
        assert trajectory.human_input_id == sample_human_input.id
        assert trajectory.tool_name.startswith("tool_1")

    # Get trajectories for the second human input
    trajectories = repo.get_trajectories_by_human_input(other_human_input.id)

    # Verify we got a list of TrajectoryModel objects for the second human input
    assert len(trajectories) == 3
    for trajectory in trajectories:
        assert isinstance(trajectory, TrajectoryModel)
        assert trajectory.human_input_id == other_human_input.id
        assert trajectory.tool_name.startswith("tool_2")


def test_get_parsed_trajectory(
    setup_db, sample_trajectory, test_tool_parameters, test_tool_result, test_step_data
):
    """Test retrieving a parsed trajectory."""
    # Set up repository
    repo = TrajectoryRepository(db=setup_db)

    # Get the parsed trajectory
    trajectory = repo.get_parsed_trajectory(sample_trajectory.id)

    # Verify type is TrajectoryModel, not Trajectory
    assert isinstance(trajectory, TrajectoryModel)

    # Verify the retrieved trajectory matches the original
    assert trajectory.id == sample_trajectory.id
    assert trajectory.tool_name == sample_trajectory.tool_name

    # Verify the JSON fields are dictionaries, not strings
    assert isinstance(trajectory.tool_parameters, dict)
    assert isinstance(trajectory.tool_result, dict)
    assert isinstance(trajectory.step_data, dict)

    # Verify the content of JSON fields
    assert trajectory.tool_parameters == test_tool_parameters
    assert trajectory.tool_result == test_tool_result
    assert trajectory.step_data == test_step_data

    # Verify non-existent trajectory returns None
    non_existent_trajectory = repo.get_parsed_trajectory(999)
    assert non_existent_trajectory is None


def test_get_session_usage_totals_empty(setup_db):
    """Test calculating session usage totals with no records."""
    # Set up repository
    repo = TrajectoryRepository(db=setup_db)

    # Get totals for a session with no records
    totals = repo.get_session_usage_totals(1)

    # Verify the totals are all zero
    assert totals["total_cost"] == 0.0
    assert totals["total_input_tokens"] == 0
    assert totals["total_output_tokens"] == 0
    assert totals["total_tokens"] == 0


def test_get_session_usage_totals(setup_db):
    """Test calculating session usage totals with multiple records."""
    # Set up repository
    repo = TrajectoryRepository(db=setup_db)

    # Create some model usage records for a session
    Trajectory.create(
        session=1,
        record_type="model_usage",
        current_cost=0.001,
        input_tokens=100,
        output_tokens=50,
    )
    Trajectory.create(
        session=1,
        record_type="model_usage",
        current_cost=0.002,
        input_tokens=200,
        output_tokens=100,
    )
    Trajectory.create(
        session=1,
        record_type="model_usage",
        current_cost=0.003,
        input_tokens=300,
        output_tokens=150,
    )

    # Create a record with a different record_type that should be ignored
    Trajectory.create(
        session=1,
        record_type="tool_execution",
        current_cost=0.999,
        input_tokens=999,
        output_tokens=999,
    )

    # Create a second session
    Session.create(id=2, name="Test Session 2")

    # Create a record for a different session that should be ignored
    Trajectory.create(
        session=2,
        record_type="model_usage",
        current_cost=0.999,
        input_tokens=999,
        output_tokens=999,
    )

    # Get totals for session 1
    totals = repo.get_session_usage_totals(1)

    # Verify the totals are calculated correctly
    assert totals["total_cost"] == pytest.approx(0.006)  # 0.001 + 0.002 + 0.003
    assert totals["total_input_tokens"] == 600  # 100 + 200 + 300
    assert totals["total_output_tokens"] == 300  # 50 + 100 + 150
    assert totals["total_tokens"] == 900  # 600 + 300

    # Get totals for session 2
    totals_session2 = repo.get_session_usage_totals(2)

    # Verify the totals for session 2
    assert totals_session2["total_cost"] == pytest.approx(0.999)  # Just the one record
    assert totals_session2["total_input_tokens"] == 999
    assert totals_session2["total_output_tokens"] == 999
    assert totals_session2["total_tokens"] == 1998  # 999 + 999


def test_get_session_usage_totals_with_nulls(setup_db):
    """Test calculating session usage totals with null values."""
    # Set up repository
    repo = TrajectoryRepository(db=setup_db)

    # Create records with some null values
    Trajectory.create(
        session=1,
        record_type="model_usage",
        current_cost=0.001,
        input_tokens=None,  # Null input tokens
        output_tokens=50,
    )
    Trajectory.create(
        session=1,
        record_type="model_usage",
        current_cost=None,  # Null cost
        input_tokens=200,
        output_tokens=None,  # Null output tokens
    )

    # Get totals for session 1
    totals = repo.get_session_usage_totals(1)

    # Verify the totals handle null values correctly
    assert totals["total_cost"] == pytest.approx(0.001)  # Only the non-null cost
    assert totals["total_input_tokens"] == 200  # Only the non-null input tokens
    assert totals["total_output_tokens"] == 50  # Only the non-null output tokens
    assert totals["total_tokens"] == 250  # 200 + 50


def test_get_trajectories_by_session(setup_db, mock_session_repository, cleanup_repo): # Use cleanup_repo fixture
    """Test retrieving trajectories by session ID."""
    # Set up repository
    repo = TrajectoryRepository(db=setup_db)

    # Create two sessions
    session1 = 1  # Using the default session from fixture
    session2 = Session.create(id=2).id

    # Create trajectories for both sessions
    for i in range(2):
        repo.create(tool_name=f"tool_s1_{i}", session_id=session1)

    for i in range(3):
        repo.create(tool_name=f"tool_s2_{i}", session_id=session2)

    # Get trajectories for the first session
    trajectories = repo.get_trajectories_by_session(session1)

    # Verify we got a list of TrajectoryModel objects for the first session
    assert len(trajectories) == 2
    for trajectory in trajectories:
        assert isinstance(trajectory, TrajectoryModel)
        assert trajectory.session_id == session1
        assert trajectory.tool_name.startswith("tool_s1")

    # Get trajectories for the second session
    trajectories = repo.get_trajectories_by_session(session2)

    # Verify we got a list of TrajectoryModel objects for the second session
    assert len(trajectories) == 3
    for trajectory in trajectories:
        assert isinstance(trajectory, TrajectoryModel)
        assert trajectory.session_id == session2
        assert trajectory.tool_name.startswith("tool_s2")


def test_trajectory_repository_manager(setup_db, cleanup_repo, mock_session_repository):
    """Test the TrajectoryRepositoryManager context manager."""
    # Use the context manager to create a repository
    with TrajectoryRepositoryManager(setup_db) as repo:
        # Verify the repository was created correctly
        assert isinstance(repo, TrajectoryRepository)
        assert repo.db is setup_db

        # Create a trajectory and verify it's a TrajectoryModel
        tool_parameters = {"test": "manager"}
        trajectory = repo.create(
            tool_name="manager_test", tool_parameters=tool_parameters
        )
        assert isinstance(trajectory, TrajectoryModel)
        assert trajectory.tool_parameters["test"] == "manager"

        # Verify we can get the repository using get_trajectory_repository
        repo_from_var = get_trajectory_repository()
        assert repo_from_var is repo

    # Verify the repository was removed from the context var
    with pytest.raises(RuntimeError) as excinfo:
        get_trajectory_repository()

    assert "No TrajectoryRepository available" in str(excinfo.value)


def test_repository_init_without_db():
    """Test that TrajectoryRepository raises an error when initialized without a db parameter."""
    # Attempt to create a repository without a database connection
    with pytest.raises(ValueError) as excinfo:
        TrajectoryRepository(db=None)

    # Verify the correct error message
    assert "Database connection is required" in str(excinfo.value)


# --- Tests for Hook Mechanism (Instance Level) ---

def test_register_create_hook_instance(setup_db, cleanup_repo):
    """Test registering a valid hook on an instance."""
    repo = TrajectoryRepository(db=setup_db)
    mock_hook = MagicMock()
    mock_hook.__name__ = 'mock_hook'
    repo.register_create_hook(mock_hook) # Call on instance
    assert mock_hook in repo._create_hooks # Check instance attribute

def test_register_invalid_hook_instance(setup_db, cleanup_repo):
    """Test registering an invalid (non-callable) hook on an instance."""
    repo = TrajectoryRepository(db=setup_db)
    with pytest.raises(TypeError, match="Hook must be callable"):
        repo.register_create_hook("not a function") # Call on instance

def test_create_hook_execution_instance(setup_db, cleanup_repo, mock_session_repository):
    """Test that instance hooks are executed upon trajectory creation."""
    repo = TrajectoryRepository(db=setup_db)

    # Create mock hooks
    mock_hook_1 = MagicMock(name='hook1')
    mock_hook_1.__name__ = 'mock_hook_1'
    mock_hook_2 = MagicMock(name="hook2")
    mock_hook_2.__name__ = 'mock_hook_2'

    # Register hooks on the instance
    repo.register_create_hook(mock_hook_1)
    repo.register_create_hook(mock_hook_2)

    # Create a trajectory
    trajectory_model = repo.create(tool_name="hook_test", current_cost=0.1)

    # Verify hooks were called once with the created model
    mock_hook_1.assert_called_once_with(trajectory_model)
    mock_hook_2.assert_called_once_with(trajectory_model)

    # Verify the model content passed to the hook
    args, _ = mock_hook_1.call_args
    passed_model = args[0]
    assert isinstance(passed_model, TrajectoryModel)
    assert passed_model.tool_name == "hook_test"
    assert passed_model.current_cost == 0.1

def test_create_hook_isolation(setup_db, cleanup_repo, mock_session_repository):
    """Test that hooks registered on one instance don't affect another."""
    repo1 = TrajectoryRepository(db=setup_db)
    repo2 = TrajectoryRepository(db=setup_db)

    hook_for_repo1 = MagicMock(name='hook_repo1')
    hook_for_repo1.__name__ = 'hook_for_repo1'

    repo1.register_create_hook(hook_for_repo1)

    # Create trajectory using repo2
    trajectory_model_2 = repo2.create(tool_name="repo2_test")

    # Verify hook for repo1 was NOT called
    hook_for_repo1.assert_not_called()

    # Create trajectory using repo1
    trajectory_model_1 = repo1.create(tool_name="repo1_test")

    # Verify hook for repo1 WAS called
    hook_for_repo1.assert_called_once_with(trajectory_model_1)


def test_create_hook_error_handling_instance(setup_db, cleanup_repo, mock_session_repository):
    """Test that instance hook errors are logged and don't stop other hooks or the create method."""
    repo = TrajectoryRepository(db=setup_db)

    # Mock logger to check error logging
    with patch("ra_aid.database.repositories.trajectory_repository.logger") as mock_logger:

        # Create mock hooks
        failing_hook = MagicMock(name="failing_hook", side_effect=ValueError("Hook failed!"))
        failing_hook.__name__ = 'failing_hook'
        successful_hook = MagicMock(name="successful_hook")
        successful_hook.__name__ = 'successful_hook'

        # Register hooks on the instance
        repo.register_create_hook(failing_hook)
        repo.register_create_hook(successful_hook)

        # Create a trajectory - this should not raise an exception
        trajectory_model = None
        try:
            trajectory_model = repo.create(tool_name="hook_error_test")
        except Exception as e:
            pytest.fail(f"repo.create raised an unexpected exception: {e}")

        # Verify the failing hook was called
        failing_hook.assert_called_once()

        # Verify the successful hook was still called after the failing one
        successful_hook.assert_called_once_with(trajectory_model)

        # Verify the error was logged
        mock_logger.error.assert_called_once()
        args, kwargs = mock_logger.error.call_args
        assert "Error executing trajectory create hook failing_hook" in args[0]
        assert "Hook failed!" in args[0]
        assert kwargs.get("exc_info") is True # Check if stack trace was included

        # Verify create still returned the model
        assert isinstance(trajectory_model, TrajectoryModel)
        assert trajectory_model.tool_name == "hook_error_test"

# --- End Tests for Hook Mechanism ---
