"""
Tests for the SessionRepository class.
"""

import pytest
import datetime
import json
from unittest.mock import patch

import peewee

from ra_aid.database.connection import DatabaseManager, db_var
from ra_aid.database.models import Session, BaseModel, HumanInput
from ra_aid.database.repositories.session_repository import (
    SessionRepository,
    SessionRepositoryManager,
    get_session_repository,
    session_repo_var
)
from ra_aid.database.pydantic_models import SessionModel


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
    session_repo_var.set(None)
    
    # Run the test
    yield
    
    # Reset after the test
    session_repo_var.set(None)


@pytest.fixture
def setup_db(cleanup_db):
    """Set up an in-memory database with the Session table and patch the BaseModel.Meta.database."""
    # Initialize an in-memory database connection
    with DatabaseManager(in_memory=True) as db:
        # Enable foreign keys
        db.execute_sql('PRAGMA foreign_keys = ON')
        
        # Patch the BaseModel.Meta.database to use our in-memory database
        with patch.object(BaseModel._meta, 'database', db):
            # Create the tables
            with db.atomic():
                db.create_tables([Session, HumanInput], safe=True)
            
            yield db
            
            # Clean up - drop tables in reverse order to handle foreign key constraints
            with db.atomic():
                try:
                    # First drop HumanInput (which depends on Session)
                    HumanInput.drop_table(safe=True)
                    # Then drop Session
                    Session.drop_table(safe=True)
                except Exception as e:
                    # Log error and continue
                    import logging
                    logging.error(f"Error dropping tables: {str(e)}")


@pytest.fixture
def test_metadata():
    """Return test metadata for sessions."""
    return {
        "os": "Test OS",
        "version": "1.0",
        "cpu_cores": 4,
        "memory_gb": 16,
        "additional_info": {
            "gpu": "Test GPU",
            "display_resolution": "1920x1080"
        }
    }


@pytest.fixture
def sample_session(setup_db, test_metadata):
    """Create a sample session in the database."""
    now = datetime.datetime.now()
    with setup_db.atomic():
        return Session.create(
            created_at=now,
            updated_at=now,
            start_time=now,
            command_line="python -m ra_aid.cli",
            program_version="0.1.0",
            machine_info=json.dumps(test_metadata)
        )


@pytest.fixture
def session_with_long_command(setup_db):
    """Create a session with a long command line."""
    now = datetime.datetime.now()
    long_command = "python -m ra_aid.cli " + "very_long_argument " * 20  # Will exceed 80 chars
    with setup_db.atomic():
        return Session.create(
            created_at=now,
            updated_at=now,
            start_time=now,
            command_line=long_command,
            program_version="0.1.0"
        )


@pytest.fixture
def human_input_model():
    """Create the HumanInput model in the database."""
    return HumanInput


@pytest.fixture
def session_with_human_input(setup_db):
    """Create a test session with a single human input."""
    # Create a session
    session = Session.create(
        start_time=datetime.datetime.now(),
        command_line="python -m ra_aid.cli",
        program_version="0.1.0",
    )
    
    # Create a human input linked to the session
    human_input = HumanInput.create(
        session=session,
        content="This is a test human input message",
        created_at=datetime.datetime.now(),
        source="cli",
    )
    
    # Return both objects for use in tests
    return session, human_input


@pytest.fixture
def session_with_long_human_input(setup_db):
    """Create a test session with a human input that has more than 80 characters."""
    # Create a session
    session = Session.create(
        start_time=datetime.datetime.now(),
        command_line="python -m ra_aid.cli",
        program_version="0.1.0",
    )
    
    # Create a long human input (> 80 chars)
    long_message = "This is a very long human input message that should be truncated in the display name since it's over 80 characters long."
    human_input = HumanInput.create(
        session=session,
        content=long_message,
        created_at=datetime.datetime.now(),
        source="chat",
    )
    
    # Return both objects for use in tests
    return session, human_input


@pytest.fixture
def multiple_human_inputs_session(setup_db):
    """Create a test session with multiple human inputs of different ages."""
    # Create a session
    session = Session.create(
        start_time=datetime.datetime.now(),
        command_line="python -m ra_aid.cli",
        program_version="0.1.0",
    )
    
    # Create human inputs with different timestamps
    human_inputs = []
    
    # Oldest input (created first, lowest ID)
    oldest_input = HumanInput.create(
        session=session,
        content="This is the oldest message and should be used for display name",
        created_at=datetime.datetime.now() - datetime.timedelta(hours=2),
        source="cli",
    )
    human_inputs.append(oldest_input)
    
    # Middle input
    middle_input = HumanInput.create(
        session=session,
        content="This is a newer message",
        created_at=datetime.datetime.now() - datetime.timedelta(hours=1),
        source="chat",
    )
    human_inputs.append(middle_input)
    
    # Newest input
    newest_input = HumanInput.create(
        session=session,
        content="This is the newest message",
        created_at=datetime.datetime.now(),
        source="hil",
    )
    human_inputs.append(newest_input)
    
    # Return session and all human inputs for use in tests
    return session, human_inputs


def test_create_session_with_metadata(setup_db, test_metadata):
    """Test creating a session with metadata."""
    # Set up repository
    repo = SessionRepository(db=setup_db)
    
    # Create a session with metadata
    session = repo.create_session(metadata=test_metadata)
    
    # Verify type is SessionModel, not Session
    assert isinstance(session, SessionModel)
    
    # Verify the session was created correctly
    assert session.id is not None
    assert session.command_line is not None
    assert session.program_version is not None
    
    # Verify machine_info is a dict, not a JSON string
    assert isinstance(session.machine_info, dict)
    assert session.machine_info == test_metadata
    
    # Verify the dictionary structure is preserved
    assert "additional_info" in session.machine_info
    assert session.machine_info["additional_info"]["gpu"] == "Test GPU"


def test_create_session_without_metadata(setup_db):
    """Test creating a session without metadata."""
    # Set up repository
    repo = SessionRepository(db=setup_db)
    
    # Create a session without metadata
    session = repo.create_session()
    
    # Verify type is SessionModel, not Session
    assert isinstance(session, SessionModel)
    
    # Verify the session was created correctly
    assert session.id is not None
    assert session.command_line is not None
    assert session.program_version is not None
    
    # Verify machine_info is None
    assert session.machine_info is None


def test_get_current_session(setup_db, sample_session):
    """Test retrieving the current session."""
    # Set up repository
    repo = SessionRepository(db=setup_db)
    
    # Set the current session
    repo.current_session = sample_session
    
    # Get the current session
    current_session = repo.get_current_session()
    
    # Verify type is SessionModel, not Session
    assert isinstance(current_session, SessionModel)
    
    # Verify the retrieved session matches the original
    assert current_session.id == sample_session.id
    assert current_session.command_line == sample_session.command_line
    assert current_session.program_version == sample_session.program_version
    
    # Verify machine_info is a dict, not a JSON string
    assert isinstance(current_session.machine_info, dict)


def test_get_current_session_from_db(setup_db, sample_session):
    """Test retrieving the current session from the database when no current session is set."""
    # Set up repository
    repo = SessionRepository(db=setup_db)
    
    # Get the current session (should retrieve the most recent from DB)
    current_session = repo.get_current_session()
    
    # Verify type is SessionModel, not Session
    assert isinstance(current_session, SessionModel)
    
    # Verify the retrieved session matches the sample session
    assert current_session.id == sample_session.id
    
    # Verify machine_info is a dict, not a JSON string
    assert isinstance(current_session.machine_info, dict)


def test_get_by_id(setup_db, sample_session):
    """Test retrieving a session by ID."""
    # Set up repository
    repo = SessionRepository(db=setup_db)
    
    # Get the session by ID
    session = repo.get(sample_session.id)
    
    # Verify type is SessionModel, not Session
    assert isinstance(session, SessionModel)
    
    # Verify the retrieved session matches the original
    assert session.id == sample_session.id
    assert session.command_line == sample_session.command_line
    assert session.program_version == sample_session.program_version
    
    # Verify machine_info is a dict, not a JSON string
    assert isinstance(session.machine_info, dict)
    
    # Verify getting a non-existent session returns None
    non_existent_session = repo.get(999)
    assert non_existent_session is None


def test_get_all(setup_db):
    """Test retrieving all sessions."""
    # Set up repository
    repo = SessionRepository(db=setup_db)
    
    # Create multiple sessions
    metadata1 = {"os": "Linux", "cpu_cores": 8}
    metadata2 = {"os": "Windows", "cpu_cores": 4}
    metadata3 = {"os": "macOS", "cpu_cores": 10}
    
    repo.create_session(metadata=metadata1)
    repo.create_session(metadata=metadata2)
    repo.create_session(metadata=metadata3)
    
    # Get all sessions with default pagination
    sessions, total_count = repo.get_all()
    
    # Verify total count
    assert total_count == 3
    
    # Verify we got a list of SessionModel objects
    assert len(sessions) == 3
    for session in sessions:
        assert isinstance(session, SessionModel)
        assert isinstance(session.machine_info, dict)
    
    # Verify the sessions are in descending order of creation time
    assert sessions[0].created_at >= sessions[1].created_at
    assert sessions[1].created_at >= sessions[2].created_at
    
    # Verify the machine_info fields
    os_values = [session.machine_info["os"] for session in sessions]
    assert "Linux" in os_values
    assert "Windows" in os_values
    assert "macOS" in os_values
    
    # Test pagination with limit
    sessions_limited, total_count = repo.get_all(limit=2)
    assert total_count == 3  # Total count should still be 3
    assert len(sessions_limited) == 2  # But only 2 returned
    
    # Test pagination with offset
    sessions_offset, total_count = repo.get_all(offset=1, limit=2)
    assert total_count == 3
    assert len(sessions_offset) == 2
    
    # The second item in the full list should be the first item in the offset list
    assert sessions[1].id == sessions_offset[0].id


def test_get_all_empty(setup_db):
    """Test retrieving all sessions when none exist."""
    # Set up repository
    repo = SessionRepository(db=setup_db)
    
    # Get all sessions
    sessions, total_count = repo.get_all()
    
    # Verify we got an empty list and zero count
    assert isinstance(sessions, list)
    assert len(sessions) == 0
    assert total_count == 0


def test_get_recent(setup_db):
    """Test retrieving recent sessions with a limit."""
    # Set up repository
    repo = SessionRepository(db=setup_db)
    
    # Create multiple sessions
    for i in range(5):
        metadata = {"index": i, "os": f"OS {i}"}
        repo.create_session(metadata=metadata)
    
    # Get recent sessions with limit=3
    sessions = repo.get_recent(limit=3)
    
    # Verify we got the correct number of SessionModel objects
    assert len(sessions) == 3
    for session in sessions:
        assert isinstance(session, SessionModel)
        assert isinstance(session.machine_info, dict)
    
    # Verify the sessions are in descending order and are the most recent ones
    indexes = [session.machine_info["index"] for session in sessions]
    assert indexes == [4, 3, 2]  # Most recent first


def test_session_repository_manager(setup_db, cleanup_repo):
    """Test the SessionRepositoryManager context manager."""
    # Create a real SessionRepository instance to use in the test
    real_repo = SessionRepository(db=setup_db)
    
    # Mock the SessionRepositoryManager.__enter__ method to return our real repo
    with patch('ra_aid.database.repositories.session_repository.SessionRepositoryManager.__enter__', 
               return_value=real_repo):
        # Use the context manager to create a repository
        with SessionRepositoryManager(setup_db) as repo:
            # Verify the repository was created correctly
            assert isinstance(repo, SessionRepository)
            assert repo.db is setup_db
            
            # Create a session and verify it's a SessionModel
            metadata = {"test": "manager"}
            session = repo.create_session(metadata=metadata)
            assert isinstance(session, SessionModel)
            assert session.machine_info["test"] == "manager"
            
            # Set the contextvar to our repo for the get_session_repository test
            session_repo_var.set(repo)
            
            # Verify we can get the repository using get_session_repository
            repo_from_var = get_session_repository()
            assert repo_from_var is repo
    
    # Reset the repository variable to avoid affecting other tests
    session_repo_var.set(None)
    
    # Verify the repository was removed from the context var
    with pytest.raises(RuntimeError) as excinfo:
        get_session_repository()
    
    assert "No SessionRepository available" in str(excinfo.value)


def test_repository_init_without_db():
    """Test that SessionRepository raises an error when initialized without a db parameter."""
    # Attempt to create a repository without a database connection
    with pytest.raises(ValueError) as excinfo:
        SessionRepository(db=None)
    
    # Verify the correct error message
    assert "Database connection is required" in str(excinfo.value)


def test_get_current_session_id(setup_db, sample_session):
    """Test retrieving the ID of the current session."""
    # Set up repository
    repo = SessionRepository(db=setup_db)
    
    # Set the current session
    repo.current_session = sample_session
    
    # Get the current session ID
    session_id = repo.get_current_session_id()
    
    # Verify the ID matches
    assert session_id == sample_session.id
    
    # Test when no current session exists
    repo.current_session = None
    # Delete all sessions
    Session.delete().execute()
    
    # Verify None is returned when no session exists
    session_id = repo.get_current_session_id()
    assert session_id is None


def test_display_name_from_command_line(setup_db, sample_session):
    """Test that display_name uses command_line when no human input exists."""
    # Create repository
    repo = SessionRepository(setup_db)
    
    # Get the session
    session_model = repo.get(sample_session.id)
    
    # Check display_name
    assert session_model is not None
    assert session_model.display_name == "python -m ra_aid.cli"


def test_display_name_from_long_command_line(setup_db, session_with_long_command):
    """Test that display_name truncates long command_line correctly."""
    # Create repository
    repo = SessionRepository(setup_db)
    
    # Get the session
    session_model = repo.get(session_with_long_command.id)
    
    # Check display_name
    assert session_model is not None
    assert len(session_model.display_name) <= 83  # 80 chars + "..."
    assert session_model.display_name.endswith("...")
    assert session_model.display_name.startswith("python -m ra_aid.cli very_long_argument")


def test_display_name_from_human_input(setup_db, session_with_human_input):
    """Test that display_name uses human input content when it exists."""
    session, human_input = session_with_human_input
    
    # Create repository
    repo = SessionRepository(setup_db)
    
    # Get the session
    session_model = repo.get(session.id)
    
    # Check display_name
    assert session_model is not None
    assert session_model.display_name == "This is a test human input message"


def test_display_name_from_long_human_input(setup_db, session_with_long_human_input):
    """Test that display_name truncates long human input correctly."""
    session, human_input = session_with_long_human_input
    
    # Create repository
    repo = SessionRepository(setup_db)
    
    # Get the session
    session_model = repo.get(session.id)
    
    # Check display_name
    assert session_model is not None
    assert len(session_model.display_name) <= 83  # 80 chars + "..."
    assert session_model.display_name.endswith("...")
    assert session_model.display_name.startswith("This is a very long human input message")


def test_display_name_from_oldest_human_input(setup_db, multiple_human_inputs_session):
    """Test that display_name uses the oldest human input when multiple exist."""
    session, human_inputs = multiple_human_inputs_session
    oldest_input = human_inputs[0]  # First input is the oldest
    
    # Create repository
    repo = SessionRepository(setup_db)
    
    # Get the session
    session_model = repo.get(session.id)
    
    # Check display_name
    assert session_model is not None
    assert session_model.display_name == "This is the oldest message and should be used for display name"


def test_display_name_in_get_all(setup_db):
    """Test that display_name is included in get_all results."""
    # Create multiple sessions
    session1 = Session.create(
        start_time=datetime.datetime.now() - datetime.timedelta(hours=2),
        command_line="python -m ra_aid.cli command1",
        program_version="0.1.0",
    )
    
    session2 = Session.create(
        start_time=datetime.datetime.now() - datetime.timedelta(hours=1),
        command_line="python -m ra_aid.cli command2",
        program_version="0.1.0",
    )
    
    # Add human input to session1
    human_input1 = HumanInput.create(
        session=session1,
        content="This is a human input for session 1",
        created_at=datetime.datetime.now() - datetime.timedelta(hours=2),
        source="cli",
    )
    
    # Create repository
    repo = SessionRepository(setup_db)
    
    # Get all sessions
    sessions, count = repo.get_all()
    
    # Check results
    assert len(sessions) == 2
    assert count == 2
    
    # Check that display_name is present in each session
    for session in sessions:
        assert session.display_name is not None
    
    # Find session1 in results (might be in any order)
    session1_result = next((s for s in sessions if s.id == session1.id), None)
    assert session1_result is not None
    assert session1_result.display_name == "This is a human input for session 1"
    
    # Find session2 in results
    session2_result = next((s for s in sessions if s.id == session2.id), None)
    assert session2_result is not None
    assert session2_result.display_name == "python -m ra_aid.cli command2"


def test_display_name_in_get_recent(setup_db):
    """Test that display_name is included in get_recent results."""
    # Create multiple sessions with different timestamps
    session1 = Session.create(
        start_time=datetime.datetime.now() - datetime.timedelta(hours=2),
        command_line="python -m ra_aid.cli command1",
        program_version="0.1.0",
    )
    
    session2 = Session.create(
        start_time=datetime.datetime.now() - datetime.timedelta(hours=1),
        command_line="python -m ra_aid.cli command2",
        program_version="0.1.0",
    )
    
    # Add human input to session2
    human_input2 = HumanInput.create(
        session=session2,
        content="This is a human input for session 2",
        created_at=datetime.datetime.now() - datetime.timedelta(hours=1),
        source="chat",
    )
    
    # Create repository
    repo = SessionRepository(setup_db)
    
    # Get recent sessions
    sessions = repo.get_recent()
    
    # Check results
    assert len(sessions) == 2
    
    # Check that display_name is present in each session
    for session in sessions:
        assert session.display_name is not None
    
    # Find session1 in results (might be in any order)
    session1_result = next((s for s in sessions if s.id == session1.id), None)
    assert session1_result is not None
    assert session1_result.display_name == "python -m ra_aid.cli command1"
    
    # Find session2 in results
    session2_result = next((s for s in sessions if s.id == session2.id), None)
    assert session2_result is not None
    assert session2_result.display_name == "This is a human input for session 2"
