"""
Tests for the SessionRepository class.
"""

import pytest
import datetime
import json
from unittest.mock import patch

import peewee

from ra_aid.database.connection import DatabaseManager, db_var
from ra_aid.database.models import Session, BaseModel
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
        # Patch the BaseModel.Meta.database to use our in-memory database
        with patch.object(BaseModel._meta, 'database', db):
            # Create the Session table
            with db.atomic():
                db.create_tables([Session], safe=True)
            
            yield db
            
            # Clean up
            with db.atomic():
                Session.drop_table(safe=True)


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
    return Session.create(
        start_time=now,
        command_line="ra-aid test",
        program_version="1.0.0",
        machine_info=json.dumps(test_metadata)
    )


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
        
        # Verify we can get the repository using get_session_repository
        repo_from_var = get_session_repository()
        assert repo_from_var is repo
    
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