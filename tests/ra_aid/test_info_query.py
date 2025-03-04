"""Tests for the is_informational_query and is_stage_requested functions."""

from ra_aid.__main__ import is_informational_query, is_stage_requested
from ra_aid.tools.memory import _global_memory


def test_is_informational_query():
    """Test that is_informational_query only depends on research_only config setting."""
    # Clear global memory to ensure clean state
    _global_memory.clear()
    
    # When research_only is True, should return True
    _global_memory["config"] = {"research_only": True}
    assert is_informational_query() is True
    
    # When research_only is False, should return False
    _global_memory["config"] = {"research_only": False}
    assert is_informational_query() is False
    
    # When config is empty, should return False (default)
    _global_memory.clear()
    _global_memory["config"] = {}
    assert is_informational_query() is False
    
    # When global memory is empty, should return False (default)
    _global_memory.clear()
    assert is_informational_query() is False


def test_is_stage_requested():
    """Test that is_stage_requested always returns False now."""
    # Clear global memory to ensure clean state
    _global_memory.clear()
    
    # Should always return False regardless of input
    assert is_stage_requested("implementation") is False
    assert is_stage_requested("anything_else") is False
    
    # Even if we set implementation_requested in global memory
    _global_memory["implementation_requested"] = True
    assert is_stage_requested("implementation") is False