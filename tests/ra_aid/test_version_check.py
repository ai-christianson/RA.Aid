"""Tests for version check module."""

from unittest.mock import Mock

import requests
import pytest

from ra_aid.version_check import check_for_newer_version

def test_newer_version_available(mocker):
    """Test when a newer version is available."""
    # Mock the dependencies
    mocker.patch('ra_aid.version_check.current_version', '0.15.2')
    mock_get = mocker.patch('ra_aid.version_check.requests.get')
    
    # Mock the response
    mock_response = Mock()
    mock_response.json.return_value = {"version": "0.16.0"}
    mock_get.return_value = mock_response
    
    result = check_for_newer_version()
    
    # Check that the message contains the new version
    assert "0.16.0" in result
    assert "A new version of RA.Aid is available" in result

def test_same_version(mocker):
    """Test when the current version is the latest."""
    # Mock the dependencies
    mocker.patch('ra_aid.version_check.current_version', '0.15.2')
    mock_get = mocker.patch('ra_aid.version_check.requests.get')
    
    # Mock the response
    mock_response = Mock()
    mock_response.json.return_value = {"version": "0.15.2"}
    mock_get.return_value = mock_response
    
    result = check_for_newer_version()
    
    # Check that no message is returned
    assert result == ""

def test_older_version(mocker):
    """Test when the current version is newer than the latest."""
    # Mock the dependencies
    mocker.patch('ra_aid.version_check.current_version', '0.15.2')
    mock_get = mocker.patch('ra_aid.version_check.requests.get')
    
    # Mock the response
    mock_response = Mock()
    mock_response.json.return_value = {"version": "0.14.0"}
    mock_get.return_value = mock_response
    
    result = check_for_newer_version()
    
    # Check that no message is returned
    assert result == ""

def test_connection_error(mocker):
    """Test handling of connection errors."""
    # Mock a connection error
    mock_get = mocker.patch('ra_aid.version_check.requests.get')
    mock_get.side_effect = requests.RequestException("Connection error")
    
    result = check_for_newer_version()
    
    # Check that no message is returned
    assert result == ""

def test_json_parse_error(mocker):
    """Test handling of JSON parsing errors."""
    # Mock the response with invalid JSON
    mock_get = mocker.patch('ra_aid.version_check.requests.get')
    mock_response = Mock()
    mock_response.json.side_effect = ValueError("Invalid JSON")
    mock_get.return_value = mock_response
    
    result = check_for_newer_version()
    
    # Check that no message is returned
    assert result == ""

def test_missing_version_key(mocker):
    """Test handling of missing version key in JSON."""
    # Mock the response with missing version key
    mock_get = mocker.patch('ra_aid.version_check.requests.get')
    mock_response = Mock()
    mock_response.json.return_value = {"other_key": "value"}
    mock_get.return_value = mock_response
    
    result = check_for_newer_version()
    
    # Check that no message is returned
    assert result == ""