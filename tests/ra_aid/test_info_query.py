"""Tests for the is_informational_query and is_stage_requested functions."""

import pytest

from ra_aid.__main__ import is_informational_query, is_stage_requested
from ra_aid.database.repositories.config_repository import ConfigRepositoryManager


@pytest.fixture
def config_repo():
    """Fixture for config repository."""
    with ConfigRepositoryManager() as repo:
        yield repo


def test_is_informational_query(config_repo):
    """Test that is_informational_query only depends on research_only config setting."""
    # When research_only is True, should return True
    config_repo.set("research_only", True)
    assert is_informational_query() is True
    
    # When research_only is False, should return False
    config_repo.set("research_only", False)
    assert is_informational_query() is False
    
    # When config is empty, should return False (default)
    config_repo.update({})
    assert is_informational_query() is False


def test_is_stage_requested():
    """Test that is_stage_requested always returns False now."""
    # Should always return False regardless of input
    assert is_stage_requested("implementation") is False
    assert is_stage_requested("anything_else") is False