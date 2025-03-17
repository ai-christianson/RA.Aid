"""
Tests for the Spawn Agent API v1 endpoint.

This module contains tests for the spawn-agent API endpoint in ra_aid/server/api_v1_spawn_agent.py.
It tests the creation of agent threads and session handling for the spawn-agent endpoint.
"""

import pytest
from unittest.mock import MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ra_aid.server.api_v1_sessions import get_repository
from ra_aid.server.api_v1_spawn_agent import router
from ra_aid.database.pydantic_models import SessionModel
import datetime
import ra_aid.server.api_v1_spawn_agent


@pytest.fixture
def mock_session():
    """Return a mock session for testing."""
    return SessionModel(
        id=123,
        created_at=datetime.datetime(2025, 1, 1, 0, 0, 0),
        updated_at=datetime.datetime(2025, 1, 1, 0, 0, 0),
        start_time=datetime.datetime(2025, 1, 1, 0, 0, 0),
        command_line="ra-aid test",
        program_version="1.0.0",
        machine_info={"agent_type": "research", "expert_enabled": True, "web_research_enabled": False}
    )


@pytest.fixture
def mock_thread():
    """Create a mock thread that does nothing when started."""
    mock = MagicMock()
    mock.daemon = True
    return mock


@pytest.fixture
def mock_repository(mock_session):
    """Create a mock repository for testing."""
    mock_repo = MagicMock()
    mock_repo.create_session.return_value = mock_session
    return mock_repo


@pytest.fixture
def mock_config_repository():
    """Create a mock config repository for testing."""
    mock_config = MagicMock()
    mock_config.get.side_effect = lambda key, default=None: {
        "expert_enabled": True,
        "web_research_enabled": False,
        "provider": "anthropic",
        "model": "claude-3-7-sonnet-20250219",
    }.get(key, default)
    return mock_config

@pytest.fixture
def client(mock_repository, mock_thread, mock_config_repository, monkeypatch):
    """Set up a test client with mocked dependencies."""
    # Create FastAPI app with router
    app = FastAPI()
    app.include_router(router)
    
    # Override the dependency to use our mock repository
    app.dependency_overrides[get_repository] = lambda: mock_repository
    
    # Mock run_agent_thread to be a no-op
    monkeypatch.setattr(
        "ra_aid.server.api_v1_spawn_agent.run_agent_thread", 
        lambda *args, **kwargs: None
    )
    
    # Mock get_config_repository to use our mock
    monkeypatch.setattr(
        "ra_aid.server.api_v1_spawn_agent.get_config_repository",
        lambda: mock_config_repository
    )
    
    # Mock threading.Thread to return our mock thread
    def mock_thread_constructor(*args, **kwargs):
        mock_thread.target = kwargs.get('target')
        mock_thread.args = kwargs.get('args')
        mock_thread.daemon = kwargs.get('daemon', False)
        return mock_thread
    
    monkeypatch.setattr(
        ra_aid.server.api_v1_spawn_agent,
        "threading",
        MagicMock(Thread=mock_thread_constructor)
    )
    
    client = TestClient(app)
    
    # Add mocks to client for test access
    client.mock_repo = mock_repository
    client.mock_thread = mock_thread
    client.mock_config = mock_config_repository
    
    yield client
    
    # Clean up the dependency override
    app.dependency_overrides.clear()


def test_spawn_agent(client, mock_repository, mock_thread):
    """Test spawning an agent with valid parameters."""
    # Create the request payload
    payload = {
        "message": "Test task for the agent",
        "research_only": False,
        "expert_enabled": True,
        "web_research_enabled": False
    }
    
    # Send the request
    response = client.post("/v1/spawn-agent", json=payload)
    
    # Verify response
    assert response.status_code == 201
    response_json = response.json()
    assert "session_id" in response_json
    
    # Verify session creation
    mock_repository.create_session.assert_called_once()
    
    # Verify thread was created with correct args
    assert mock_thread.args == ("Test task for the agent", "123", False)
    assert mock_thread.daemon is True
    
    # Verify thread.start was called
    mock_thread.start.assert_called_once()


def test_spawn_agent_missing_message(client):
    """Test spawning an agent with missing required message parameter."""
    # Create a request payload missing the required message
    payload = {
        "research_only": False,
        "expert_enabled": True,
        "web_research_enabled": False
    }
    
    # Send the request
    response = client.post("/v1/spawn-agent", json=payload)
    
    # Verify response indicates validation error
    assert response.status_code == 422
    error_detail = response.json().get("detail", [])
    assert any("message" in error.get("loc", []) for error in error_detail)
