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


@pytest.fixture
def mock_session():
    return SessionModel(
        id=1,
        created_at=datetime.datetime(2025, 1, 1),
        updated_at=datetime.datetime(2025, 1, 1),
        start_time=datetime.datetime(2025, 1, 1),
        command_line="ra-aid test",
        program_version="1.0.0",
        machine_info={"os": "test"},
        status='test_status'
    )


@pytest.fixture
def mock_sessions():
    return [
        SessionModel(
            id=i + 1,
            created_at=datetime.datetime(2025, 1, i + 1),
            updated_at=datetime.datetime(2025, 1, i + 1),
            start_time=datetime.datetime(2025, 1, i + 1),
            command_line=f"ra-aid test{i + 1}",
            program_version="1.0.0",
            machine_info={"index": i},
            status='test_status'
        )
        for i in range(15)
    ]


@pytest.fixture
def mock_repo(mock_session, mock_sessions):
    repo = MagicMock()
    repo.get.return_value = mock_session
    repo.get_all.return_value = (mock_sessions[:10], len(mock_sessions))
    repo.create_session.return_value = mock_session

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
        return sorted_sessions[offset:offset + limit], total

    def create_with_metadata(metadata=None):
        now = datetime.datetime.now()
        return SessionModel(
            id=16,
            created_at=now,
            updated_at=now,
            start_time=now,
            command_line="ra-aid test-custom" if metadata else "ra-aid test-null",
            program_version="1.0.0",
            machine_info=metadata,
            status='test_status'
        )

    repo.get.side_effect = get_with_id
    repo.get_all.side_effect = get_all_with_pagination
    repo.create_session.side_effect = create_with_metadata

    return repo


@pytest.fixture
def client(mock_repo):
    app.dependency_overrides[get_repository] = lambda: mock_repo
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def sample_metadata() -> Dict[str, Any]:
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
    response = client.post("/v1/session", json={"metadata": sample_metadata})
    assert response.status_code == 201
    data = response.json()
    assert data["id"] is not None
    assert data["command_line"] is not None
    assert data["program_version"] is not None
    assert data["created_at"] is not None
    assert data["updated_at"] is not None
    assert data["start_time"] is not None
    mock_repo.create_session.assert_called_once_with(metadata=sample_metadata)
