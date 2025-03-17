"""
Tests for the Sessions API v1 endpoints.

This module contains tests for the sessions API endpoints in ra_aid/server/api_v1_sessions.py.
It tests the creation, listing, and retrieval of sessions through the API.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import datetime

from ra_aid.server.api_v1_sessions import router, get_repository
from ra_aid.database.pydantic_models import SessionModel


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
            id=1,
            created_at=datetime.datetime(2025, 1, 1, 0, 0, 0),
            updated_at=datetime.datetime(2025, 1, 1, 0, 0, 0),
            start_time=datetime.datetime(2025, 1, 1, 0, 0, 0),
            command_line="ra-aid test1",
            program_version="1.0.0",
            machine_info={"os": "test"}
        ),
        SessionModel(
            id=2,
            created_at=datetime.datetime(2025, 1, 2, 0, 0, 0),
            updated_at=datetime.datetime(2025, 1, 2, 0, 0, 0),
            start_time=datetime.datetime(2025, 1, 2, 0, 0, 0),
            command_line="ra-aid test2",
            program_version="1.0.0",
            machine_info={"os": "test"}
        )
    ]


@pytest.fixture
def mock_repo(mock_session, mock_sessions):
    """Mock the SessionRepository for testing."""
    repo = MagicMock()
    
    # Mock individual get method
    repo.get.return_value = mock_session
    
    # Note: get_all is deprecated, but kept for backward compatibility
    repo.get_all.return_value = (mock_sessions, len(mock_sessions))
    
    repo.create_session.return_value = mock_session
    return repo


@pytest.fixture
def client(mock_repo):
    """Return a TestClient for the API router with dependency override."""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    # Override the dependency
    app.dependency_overrides[get_repository] = lambda: mock_repo
    
    return TestClient(app)


def test_get_session(client, mock_repo, mock_session):
    """Test getting a specific session by ID."""
    response = client.get("/v1/sessions/1")
    
    assert response.status_code == 200
    assert response.json()["id"] == mock_session.id
    assert response.json()["command_line"] == mock_session.command_line
    mock_repo.get.assert_called_once_with(1)


def test_get_session_not_found(client, mock_repo):
    """Test getting a session that doesn't exist."""
    mock_repo.get.return_value = None
    
    response = client.get("/v1/sessions/999")
    
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]
    mock_repo.get.assert_called_once_with(999)


def test_list_sessions(client, mock_repo, mock_sessions):
    """Test listing sessions with pagination."""
    response = client.get("/v1/sessions?offset=0&limit=10")
    
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == len(mock_sessions)
    assert len(data["items"]) == len(mock_sessions)
    assert data["limit"] == 10
    assert data["offset"] == 0
    mock_repo.get_all.assert_called_once_with(offset=0, limit=10)


def test_create_session(client, mock_repo, mock_session):
    """Test creating a new session."""
    response = client.post(
        "/v1/sessions",
        json={"metadata": {"test": "data"}}
    )
    
    assert response.status_code == 201
    assert response.json()["id"] == mock_session.id
    mock_repo.create_session.assert_called_once_with(metadata={"test": "data"})


def test_create_session_no_body(client, mock_repo, mock_session):
    """Test creating a new session without a request body."""
    response = client.post("/v1/sessions")
    
    assert response.status_code == 201
    assert response.json()["id"] == mock_session.id
    mock_repo.create_session.assert_called_once_with(metadata=None)