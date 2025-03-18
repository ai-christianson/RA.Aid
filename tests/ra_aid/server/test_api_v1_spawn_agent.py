"""
Tests for the Spawn Agent API v1 endpoint.

This module contains tests for the spawn-agent API endpoint in ra_aid/server/api_v1_spawn_agent.py.
It tests the creation of agent threads and session handling for the spawn-agent endpoint.
"""

import pytest
from unittest.mock import MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ra_aid.database.repositories.session_repository import get_session_repository
from ra_aid.server.api_v1_sessions import get_repository
from ra_aid.server.api_v1_spawn_agent import router
from ra_aid.database.pydantic_models import SessionModel
import datetime
import ra_aid.server.api_v1_spawn_agent
from ra_aid.llm import get_model_default_temperature


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
    # Create a dictionary to simulate config
    config = {
        "expert_enabled": True,
        "web_research_enabled": False,
        "provider": "anthropic",
        "model": "claude-3-7-sonnet-20250219",
    }
    
    # Setup get method to return config values
    mock_config.get.side_effect = lambda key, default=None: config.get(key, default)
    
    # Note: get_all is deprecated, but kept for backward compatibility
    # Setup get_all method to return a reference to the config dict
    mock_config.get_all.return_value = config
    return mock_config


@pytest.fixture
def client(mock_repository, mock_thread, mock_config_repository, monkeypatch):
    """Set up a test client with mocked dependencies."""
    # Create FastAPI app with router
    app = FastAPI()
    app.include_router(router)
    
    # Override the dependency to use our mock repository
    app.dependency_overrides[get_session_repository] = lambda: mock_repository
    
    # Mock run_agent_thread to be a no-op
    monkeypatch.setattr(
        "ra_aid.server.api_v1_spawn_agent.run_agent_thread", 
        lambda message, session_id, source_config_repo, research_only=False, **kwargs: None
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


@pytest.mark.skip(reason="Test needs to be updated to match current implementation")
def test_spawn_agent(client, mock_repository, mock_thread):
    """Test spawning an agent with valid parameters."""
    # Create the request payload
    payload = {
        "message": "Test task for the agent",
        "research_only": False,
        "expert_enabled": True,
        "web_research_enabled": False
    }
    
    # Ensure create_session is called when the endpoint is hit
    mock_repository.create_session.return_value.id = 123
    
    # Send the request
    response = client.post("/v1/spawn-agent", json=payload)
    
    # Verify response
    assert response.status_code == 201
    response_json = response.json()
    assert "session_id" in response_json
    
    # Verify session creation
    assert mock_repository.create_session.called
    
    # Verify thread was created with correct args
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


def test_temperature_handling_in_agent_spawn(client, mock_repository, mock_thread, mock_config_repository, monkeypatch):
    """
    Test that the temperature handling logic in the spawn agent endpoint
    correctly uses the model's default temperature when none is provided.
    """
    # Patch the get_model_default_temperature function to return a specific value for testing
    monkeypatch.setattr(
        ra_aid.server.api_v1_spawn_agent, 
        "get_model_default_temperature", 
        lambda p, m: 0.9
    )
    
    # Mock config repository to return None for temperature
    mock_config_repository.get.side_effect = lambda key, default=None: {
        "provider": "anthropic",
        "model": "claude-3-7-sonnet-20250219",
        "temperature": None,  # No temperature provided
        "expert_enabled": True,
        "web_research_enabled": False
    }.get(key, default)
    
    # Create a spy for threading.Thread
    thread_spy = MagicMock()
    
    # Store the original Thread constructor
    original_Thread = ra_aid.server.api_v1_spawn_agent.threading.Thread
    
    # Create a new Thread constructor that records the kwargs
    def spy_Thread(*args, **kwargs):
        thread_spy(*args, **kwargs)
        return mock_thread
    
    # Replace the Thread constructor with our spy version
    monkeypatch.setattr(
        ra_aid.server.api_v1_spawn_agent.threading,
        "Thread",
        spy_Thread
    )
    
    # Make the API request
    response = client.post(
        "/v1/spawn-agent",
        json={"message": "Test message"}
    )
    
    # Check that the response is successful
    assert response.status_code == 201
    
    # Verify that Thread was called with the right temperature in kwargs
    _, kwargs = thread_spy.call_args
    assert kwargs.get('kwargs', {}).get('temperature') == 0.9, \
        f"Expected temperature 0.9, got {kwargs.get('kwargs', {}).get('temperature')}"
