"""
Tests for server.py FastAPI application.

This module tests the FastAPI application setup in server.py to ensure
that all routers are properly mounted and middleware is configured.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from ra_aid.server.server import app
from ra_aid.database.repositories.session_repository import session_repo_var


@pytest.fixture
def client():
    """Return a TestClient for the FastAPI app."""
    # Mock the session repository to avoid database dependency
    mock_repo = MagicMock()
    
    # Mock get method for session repository
    def mock_get(session_id):
        return None  # No session found 
    mock_repo.get.side_effect = mock_get
    
    # Note: get_all is deprecated, but kept for backward compatibility
    mock_repo.get_all.return_value = ([], 0)
    
    # Set the repository in the contextvar
    token = session_repo_var.set(mock_repo)
    
    yield TestClient(app)
    
    # Reset the contextvar after the test
    session_repo_var.reset(token)


def test_config_endpoint(client):
    """Test that the config endpoint returns server configuration."""
    response = client.get("/config")
    assert response.status_code == 200
    assert "host" in response.json()
    assert "port" in response.json()


def test_api_documentation(client):
    """Test that the OpenAPI documentation includes the sessions API."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    
    openapi_spec = response.json()
    assert "paths" in openapi_spec
    
    # Check that the sessions API paths are included
    assert "/v1/session" in openapi_spec["paths"]
    assert "/v1/session/{session_id}" in openapi_spec["paths"]
    
    # Verify that sessions API operations are documented
    assert "get" in openapi_spec["paths"]["/v1/session"]
    assert "post" in openapi_spec["paths"]["/v1/session"]
    assert "get" in openapi_spec["paths"]["/v1/session/{session_id}"]


@patch("ra_aid.database.repositories.session_repository.get_session_repository")
def test_sessions_api_mounted(mock_get_repo, client):
    """Test that the sessions API router is mounted correctly."""
    # Mock the repository for this specific test
    mock_repo = MagicMock()
    mock_repo.get_all.return_value = ([], 0)
    mock_get_repo.return_value = mock_repo
    
    # Test that the sessions list endpoint is accessible
    response = client.get("/v1/session")
    assert response.status_code == 200
    
    # Verify the response structure follows our expected format
    data = response.json()
    assert "total" in data
    assert "items" in data
    assert "limit" in data
    assert "offset" in data