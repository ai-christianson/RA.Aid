"""Tests for version check module."""

from unittest.mock import Mock

import requests
import pytest

from ra_aid.version_check import check_for_newer_version

def test_newer_version_available(monkeypatch):
    """Test when a newer version is available."""
    # Mock the dependencies
    monkeypatch.setattr('ra_aid.version_check.current_version', '0.15.2')
    
    # Create a mock response
    mock_response = Mock()
    mock_response.json.return_value = {"version": "0.16.0"}
    
    # Create a mock function for requests.get
    def mock_get(*args, **kwargs):
        return mock_response
    
    # Set the mock function
    monkeypatch.setattr('ra_aid.version_check.requests.get', mock_get)
    
    result = check_for_newer_version()
    
    # Check that the message contains the new version
    assert "0.16.0" in result
    assert "A new version of RA.Aid is available" in result

def test_same_version(monkeypatch):
    """Test when the current version is the latest."""
    # Mock the dependencies
    monkeypatch.setattr('ra_aid.version_check.current_version', '0.15.2')
    
    # Create a mock response
    mock_response = Mock()
    mock_response.json.return_value = {"version": "0.15.2"}
    
    # Create a mock function for requests.get
    def mock_get(*args, **kwargs):
        return mock_response
    
    # Set the mock function
    monkeypatch.setattr('ra_aid.version_check.requests.get', mock_get)
    
    result = check_for_newer_version()
    
    # Check that no message is returned
    assert result == ""

def test_older_version(monkeypatch):
    """Test when the current version is newer than the latest."""
    # Mock the dependencies
    monkeypatch.setattr('ra_aid.version_check.current_version', '0.15.2')
    
    # Create a mock response
    mock_response = Mock()
    mock_response.json.return_value = {"version": "0.14.0"}
    
    # Create a mock function for requests.get
    def mock_get(*args, **kwargs):
        return mock_response
    
    # Set the mock function
    monkeypatch.setattr('ra_aid.version_check.requests.get', mock_get)
    
    result = check_for_newer_version()
    
    # Check that no message is returned
    assert result == ""

def test_connection_error(monkeypatch):
    """Test handling of connection errors."""
    # Create a mock function that raises an exception
    def mock_get(*args, **kwargs):
        raise requests.RequestException("Connection error")
    
    # Set the mock function
    monkeypatch.setattr('ra_aid.version_check.requests.get', mock_get)
    
    result = check_for_newer_version()
    
    # Check that no message is returned
    assert result == ""

def test_json_parse_error(monkeypatch):
    """Test handling of JSON parsing errors."""
    # Create a mock response
    mock_response = Mock()
    mock_response.json.side_effect = ValueError("Invalid JSON")
    
    # Create a mock function for requests.get
    def mock_get(*args, **kwargs):
        return mock_response
    
    # Set the mock function
    monkeypatch.setattr('ra_aid.version_check.requests.get', mock_get)
    
    result = check_for_newer_version()
    
    # Check that no message is returned
    assert result == ""

def test_missing_version_key(monkeypatch):
    """Test handling of missing version key in JSON."""
    # Create a mock response
    mock_response = Mock()
    mock_response.json.return_value = {"other_key": "value"}
    
    # Create a mock function for requests.get
    def mock_get(*args, **kwargs):
        return mock_response
    
    # Set the mock function
    monkeypatch.setattr('ra_aid.version_check.requests.get', mock_get)
    
    result = check_for_newer_version()
    
    # Check that no message is returned
    assert result == ""