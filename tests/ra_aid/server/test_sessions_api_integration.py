"""
Integration tests for the Sessions API endpoints.

This module contains integration tests for the API endpoints defined in ra_aid/server/api_v1_sessions.py.
It uses mocks to simulate the database interactions while testing the real API behavior.
"""

import pytest
import datetime
from typing import Dict, Any, List, Tuple
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from ra_aid.server.server import app
from ra_aid.database.pydantic_models import SessionModel
from ra_aid.server.api_v1_sessions import get_repository


# Mock session data for testing
@pytest.fixture
def mock_session():
    """Return a mock session for testing."""
    return SessionModel(
        id=1,
        created_at=datetime.datetime(2025, 1, 1, 0, 0, 0),
        updated_at=datetime.datetime(2025, 1, 1, 0, 0, 0),
        start_time=datetime.datetime(2025, 1, 1, 0, 0, 0),
        command_line="ra-aid test",
        program_version="1.0.0",
        machine_info={"os": "test"}
    )


@pytest.fixture
def mock_sessions():
    """Return a list of mock sessions for testing."""
    return [
        SessionModel(
            id=i+1,
            created_at=datetime.datetime(2025, 1, i+1, 0, 0, 0),
            updated_at=datetime.datetime(2025, 1, i+1, 0, 0, 0),
            start_time=datetime.datetime(2025, 1, i+1, 0, 0, 0),
            command_line=f"ra-aid test{i+1}",
            program_version="1.0.0",
            machine_info={"index": i}
        )
        for i in range(15)
    ]


@pytest.fixture
def mock_repo(mock_session, mock_sessions):
    """Create a mock repository with predefined responses."""
    repo = MagicMock()
    repo.get.return_value = mock_session
    repo.get_all.return_value = (mock_sessions[:10], len(mock_sessions))
    repo.create_session.return_value = mock_session
    
    # Add behavior for custom parameters
    def get_with_id(session_id):
        if session_id == 999999:
            return None
        for session in mock_sessions:
            if session.id == session_id:
                return session
        return mock_session
    
    def get_all_with_pagination(offset=0, limit=10):
        total = len(mock_sessions)
        sorted_sessions = sorted(mock_sessions, key=lambda s: s.id, reverse=True)
        return sorted_sessions[offset:offset+limit], total
    
    def create_with_metadata(metadata=None):
        if metadata is None:
            return SessionModel(
                id=16,
                created_at=datetime.datetime.now(),
                updated_at=datetime.datetime.now(),
                start_time=datetime.datetime.now(),
                command_line="ra-aid test-null",
                program_version="1.0.0",
                machine_info=None
            )
        return SessionModel(
            id=16,
            created_at=datetime.datetime.now(),
            updated_at=datetime.datetime.now(),
            start_time=datetime.datetime.now(),
            command_line="ra-aid test-custom",
            program_version="1.0.0",
            machine_info=metadata
        )
    
    repo.get.side_effect = get_with_id
    repo.get_all.side_effect = get_all_with_pagination
    repo.create_session.side_effect = create_with_metadata
    
    return repo


@pytest.fixture
def client(mock_repo):
    """Create a TestClient with the API and dependency overrides."""
    # Override the dependency to use our mock repository
    app.dependency_overrides[get_repository] = lambda: mock_repo
    
    # Create a test client
    with TestClient(app) as test_client:
        yield test_client
    
    # Clean up the dependency override
    app.dependency_overrides.clear()


@pytest.fixture
def sample_metadata() -> Dict[str, Any]:
    """Return sample metadata for session creation."""
    return {
        "os": "Test OS",
        "version": "1.0.0",
        "environment": "test",
        "cpu_cores": 4,
        "memory_gb": 16,
        "additional_info": {
            "gpu": "Test GPU",
            "display_resolution": "1920x1080"
        }
    }


def test_create_session_with_metadata(client, mock_repo, sample_metadata):
    """Test creating a session with metadata through the API endpoint."""
    # Send request to create a session with metadata
    response = client.post(
        "/v1/session",
        json={"metadata": sample_metadata}
    )
    
    # Verify response status code and structure
    assert response.status_code == 201
    data = response.json()
    
    # Verify the session was created with the expected fields
    assert data["id"] is not None
    assert data["command_line"] is not None
    assert data["program_version"] is not None
    assert data["created_at"] is not None
    assert data["updated_at"] is not None
    assert data["start_time"] is not None
    
    # Verify metadata was passed correctly to the repository
    mock_repo.create_session.assert_called_once_with(metadata=sample_metadata)


def test_create_session_without_metadata(client, mock_repo):
    """Test creating a session without metadata through the API endpoint."""
    # Send request without a body
    response = client.post("/v1/session")
    
    # Verify response status code and structure
    assert response.status_code == 201
    data = response.json()
    
    # Verify the session was created with the expected fields
    assert data["id"] is not None
    assert data["command_line"] is not None
    assert data["program_version"] is not None
    
    # Verify correct parameters were passed to the repository
    mock_repo.create_session.assert_called_once_with(metadata=None)


def test_get_session_by_id(client):
    """Test retrieving a session by ID through the API endpoint."""
    # Use a completely isolated, standalone test
    
    # For this test, let's focus on verifying the core functionality:
    # 1. The API endpoint receives a request for a specific session ID
    # 2. It calls the repository with that ID
    # 3. It returns a properly formatted response
    
    mock_repo = MagicMock()
    
    # Create a test session with a simple machine_info to reduce serialization issues
    test_session = SessionModel(
        id=42,
        created_at=datetime.datetime(2025, 1, 1, 0, 0, 0),
        updated_at=datetime.datetime(2025, 1, 1, 0, 0, 0),
        start_time=datetime.datetime(2025, 1, 1, 0, 0, 0),
        command_line="ra-aid specific-test",
        program_version="1.0.0-test",
        machine_info=None  # Use None to avoid serialization issues
    )
    
    # Configure the mock
    mock_repo.get.return_value = test_session
    
    # Override the dependency
    app.dependency_overrides[get_repository] = lambda: mock_repo
    
    try:
        # Retrieve the session through the API
        response = client.get(f"/v1/session/{test_session.id}")
        
        # Verify response status code
        assert response.status_code == 200
        
        # Parse the response data
        data = response.json()
        
        # Print for debugging
        import json
        print("Response JSON:", json.dumps(data, indent=2))
        
        # Verify the returned session matches what we expected
        assert data["id"] == test_session.id
        assert data["command_line"] == test_session.command_line
        assert data["program_version"] == test_session.program_version
        assert data["machine_info"] is None
        
        # Verify the repository was called with the correct ID
        mock_repo.get.assert_called_once_with(test_session.id)
    finally:
        # Clean up the override
        if get_repository in app.dependency_overrides:
            del app.dependency_overrides[get_repository]


def test_get_session_not_found(client, mock_repo):
    """Test the error handling when requesting a non-existent session."""
    # Try to get a session with a non-existent ID
    response = client.get("/v1/session/999999")
    
    # Verify response status code and error message
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]
    
    # Verify the repository was called with the correct ID
    mock_repo.get.assert_called_with(999999)


def test_list_sessions_empty(client, mock_repo):
    """Test listing sessions when no sessions exist."""
    # Reset the mock first to clear any previous calls/side effects
    mock_repo.reset_mock()
    
    # Configure the mock to return empty results
    # Both get and get_all should be set up to handle pagination
    def mock_get_empty(session_id):
        return None
    mock_repo.get.side_effect = mock_get_empty
    
    # Note: get_all is deprecated, but kept for backward compatibility
    mock_repo.get_all.side_effect = None  # Clear any previous side effects
    mock_repo.get_all.return_value = ([], 0)
    
    # Get the list of sessions
    response = client.get("/v1/session")
    
    # Verify response status code and structure
    assert response.status_code == 200
    data = response.json()
    
    # Verify the pagination response
    assert data["total"] == 0
    assert len(data["items"]) == 0
    assert data["limit"] == 10
    assert data["offset"] == 0
    
    # Verify the repository was called with the correct parameters
    mock_repo.get_all.assert_called_with(offset=0, limit=10)


def test_list_sessions_with_pagination(client, mock_repo, mock_sessions):
    """Test listing sessions with pagination parameters."""
    # Set up the repository mock to return specific results for different pagination parameters
    default_result = (mock_sessions[:10], len(mock_sessions))
    limit_5_result = (mock_sessions[:5], len(mock_sessions))
    offset_10_result = (mock_sessions[10:], len(mock_sessions))
    offset_5_limit_3_result = (mock_sessions[5:8], len(mock_sessions))
    
    pagination_responses = {
        (0, 10): default_result,
        (0, 5): limit_5_result,
        (10, 10): offset_10_result,
        (5, 3): offset_5_limit_3_result
    }
    
    def mock_get_all(offset=0, limit=10):
        return pagination_responses.get((offset, limit), ([], 0))
    
    mock_repo.get_all.side_effect = mock_get_all
    
    # Test default pagination (limit=10, offset=0)
    response = client.get("/v1/session")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == len(mock_sessions)
    assert len(data["items"]) == 10
    assert data["limit"] == 10
    assert data["offset"] == 0
    
    # Test with custom limit
    response = client.get("/v1/session?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == len(mock_sessions)
    assert len(data["items"]) == 5
    assert data["limit"] == 5
    assert data["offset"] == 0
    
    # Test with custom offset
    response = client.get("/v1/session?offset=10")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == len(mock_sessions)
    assert len(data["items"]) == 5  # Only 5 items left after offset 10
    assert data["limit"] == 10
    assert data["offset"] == 10
    
    # Test with both custom limit and offset
    response = client.get("/v1/session?limit=3&offset=5")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == len(mock_sessions)
    assert len(data["items"]) == 3
    assert data["limit"] == 3
    assert data["offset"] == 5


def test_list_sessions_invalid_parameters(client):
    """Test error handling for invalid pagination parameters."""
    # Test with negative offset
    response = client.get("/v1/session?offset=-1")
    assert response.status_code == 422
    
    # Test with negative limit
    response = client.get("/v1/session?limit=-5")
    assert response.status_code == 422
    
    # Test with zero limit
    response = client.get("/v1/session?limit=0")
    assert response.status_code == 422
    
    # Test with limit exceeding maximum
    response = client.get("/v1/session?limit=101")
    assert response.status_code == 422


def test_metadata_validation(client, mock_repo):
    """Test validation for different metadata formats in session creation."""
    # Create test sessions with different metadata
    null_metadata_session = SessionModel(
        id=20,
        created_at=datetime.datetime.now(),
        updated_at=datetime.datetime.now(),
        start_time=datetime.datetime.now(),
        command_line="ra-aid test-null",
        program_version="1.0.0",
        machine_info=None
    )
    
    empty_dict_metadata_session = SessionModel(
        id=21,
        created_at=datetime.datetime.now(),
        updated_at=datetime.datetime.now(),
        start_time=datetime.datetime.now(),
        command_line="ra-aid test-empty",
        program_version="1.0.0",
        machine_info={}
    )
    
    complex_metadata_session = SessionModel(
        id=22,
        created_at=datetime.datetime.now(),
        updated_at=datetime.datetime.now(),
        start_time=datetime.datetime.now(),
        command_line="ra-aid test-complex",
        program_version="1.0.0",
        machine_info={"level1": {"level2": {"level3": [1, 2, 3, {"key": "value"}]}}}
    )
    
    # Configure mock to return different sessions based on metadata
    def create_with_specific_metadata(metadata=None):
        if metadata is None:
            return null_metadata_session
        elif metadata == {}:
            return empty_dict_metadata_session
        elif isinstance(metadata, dict) and "level1" in metadata:
            return complex_metadata_session
        return SessionModel(
            id=23,
            created_at=datetime.datetime.now(),
            updated_at=datetime.datetime.now(),
            start_time=datetime.datetime.now(),
            command_line="ra-aid test-other",
            program_version="1.0.0",
            machine_info=metadata
        )
    
    mock_repo.create_session.side_effect = create_with_specific_metadata
    
    # Try to create a session with null metadata
    response = client.post(
        "/v1/session",
        json={"metadata": None}
    )
    
    # This should work fine
    assert response.status_code == 201
    mock_repo.create_session.assert_called_with(metadata=None)
    
    # Try to create a session with an empty metadata dict
    response = client.post(
        "/v1/session",
        json={"metadata": {}}
    )
    
    # This should work fine
    assert response.status_code == 201
    mock_repo.create_session.assert_called_with(metadata={})
    
    # Try to create a session with a complex nested metadata
    response = client.post(
        "/v1/session",
        json={"metadata": {
            "level1": {
                "level2": {
                    "level3": [1, 2, 3, {"key": "value"}]
                }
            }
        }}
    )
    
    # Verify the complex nested structure is preserved
    assert response.status_code == 201
    complex_metadata = {
        "level1": {
            "level2": {
                "level3": [1, 2, 3, {"key": "value"}]
            }
        }
    }
    mock_repo.create_session.assert_called_with(metadata=complex_metadata)


def test_integration_workflow(client, mock_repo):
    """Test a complete workflow of creating and retrieving sessions."""
    # Set up mock sessions for the workflow
    first_session = SessionModel(
        id=30,
        created_at=datetime.datetime.now(),
        updated_at=datetime.datetime.now(),
        start_time=datetime.datetime.now(),
        command_line="ra-aid workflow-1",
        program_version="1.0.0",
        machine_info={"workflow_test": True}
    )
    
    second_session = SessionModel(
        id=31,
        created_at=datetime.datetime.now(),
        updated_at=datetime.datetime.now(),
        start_time=datetime.datetime.now(),
        command_line="ra-aid workflow-2",
        program_version="1.0.0",
        machine_info={"workflow_test": False, "second": True}
    )
    
    # Configure mock for create_session
    create_calls = 0
    def create_session_for_workflow(metadata=None):
        nonlocal create_calls
        create_calls += 1
        if create_calls == 1:
            return first_session
        return second_session
    
    mock_repo.create_session.side_effect = create_session_for_workflow
    
    # Configure mock for get
    def get_session_for_workflow(session_id):
        if session_id == first_session.id:
            return first_session
        elif session_id == second_session.id:
            return second_session
        return None
    
    mock_repo.get.side_effect = get_session_for_workflow
    
    # Configure mock for get
    def get_pagination_for_workflow(offset=0, limit=10):
        if create_calls == 1:
            return first_session
        return second_session
    
    # Configure mock for get_all (deprecated but needed for backward compatibility)
    def get_all_for_workflow(offset=0, limit=10):
        if create_calls == 1:
            return [first_session], 1
        return [second_session, first_session], 2
    
    mock_repo.get_all.side_effect = get_all_for_workflow
    
    # 1. Create a session
    create_response = client.post(
        "/v1/session",
        json={"metadata": {"workflow_test": True}}
    )
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]
    assert session_id == first_session.id
    
    # 2. Retrieve the created session
    get_response = client.get(f"/v1/session/{session_id}")
    assert get_response.status_code == 200
    assert get_response.json()["id"] == session_id
    
    # 3. List all sessions and verify the created one is included
    list_response = client.get("/v1/session")
    assert list_response.status_code == 200
    items = list_response.json()["items"]
    assert len(items) == 1
    assert items[0]["id"] == session_id
    
    # 4. Create a second session
    create_response2 = client.post(
        "/v1/session",
        json={"metadata": {"workflow_test": False, "second": True}}
    )
    assert create_response2.status_code == 201
    session_id2 = create_response2.json()["id"]
    assert session_id2 == second_session.id
    
    # 5. List all sessions and verify both sessions are included
    list_response = client.get("/v1/session")
    assert list_response.status_code == 200
    data = list_response.json()
    assert data["total"] == 2
    items = data["items"]
    assert len(items) == 2
    assert items[0]["id"] == session_id2
    assert items[1]["id"] == session_id