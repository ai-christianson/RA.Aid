"""Tests for ConfigRepository."""

import pytest

from ra_aid.database.repositories.config_repository import (
    ConfigRepository,
    ConfigRepositoryManager,
    get_config_repository,
)


def test_config_repository_init():
    """Test ConfigRepository initialization with default values."""
    repo = ConfigRepository()
    assert repo.get("recursion_limit") is not None
    assert repo.get("max_test_cmd_retries") is not None
    assert repo.get("max_tool_failures") is not None


def test_config_repository_get_set():
    """Test ConfigRepository get and set methods."""
    repo = ConfigRepository()
    repo.set("test_key", "test_value")
    assert repo.get("test_key") == "test_value"
    
    # Test deep copy on get
    complex_value = {"a": [1, 2, 3], "b": {"c": "d"}}
    repo.set("complex", complex_value)
    result = repo.get("complex")
    
    # Verify we get a copy, not the original
    assert result == complex_value
    assert result is not complex_value
    
    # Modify the returned value and verify original is unchanged
    result["a"].append(4)
    assert repo.get("complex")["a"] == [1, 2, 3]


def test_config_repository_update():
    """Test ConfigRepository update method."""
    repo = ConfigRepository()
    repo.update({"key1": "value1", "key2": "value2"})
    assert repo.get("key1") == "value1"
    assert repo.get("key2") == "value2"
    
    # Test with complex values
    complex_dict = {
        "nested": {"a": 1, "b": 2},
        "list": [1, 2, 3]
    }
    repo.update(complex_dict)
    
    # Verify deep copy
    result = repo.get("nested")
    result["c"] = 3
    assert "c" not in repo.get("nested")


def test_config_repository_get_keys():
    """Test ConfigRepository get_keys method."""
    repo = ConfigRepository()
    repo.set("key1", "value1")
    repo.set("key2", "value2")
    
    keys = repo.get_keys()
    assert "key1" in keys
    assert "key2" in keys
    assert isinstance(keys, list)


def test_config_repository_deep_copy():
    """Test ConfigRepository deep_copy method."""
    repo = ConfigRepository()
    repo.set("key1", "value1")
    repo.set("complex", {"a": [1, 2, 3]})
    
    # Create a deep copy
    copy_repo = repo.deep_copy()
    
    # Verify values were copied
    assert copy_repo.get("key1") == "value1"
    assert copy_repo.get("complex") == {"a": [1, 2, 3]}
    
    # Modify original and verify copy is unchanged
    repo.set("key1", "changed")
    repo.get("complex")["a"].append(4)
    
    assert copy_repo.get("key1") == "value1"
    assert copy_repo.get("complex")["a"] == [1, 2, 3]
    
    # Modify copy and verify original is unchanged
    copy_repo.set("key1", "copy_changed")
    copy_repo.get("complex")["a"].append(5)
    
    assert repo.get("key1") == "changed"
    assert 5 not in repo.get("complex")["a"]


def test_config_repository_manager():
    """Test ConfigRepositoryManager."""
    with ConfigRepositoryManager() as repo:
        repo.set("test_key", "test_value")
        assert repo.get("test_key") == "test_value"
        
        # Test that the repository is available via the get_config_repository function
        assert get_config_repository() is repo
        
    # Test that the repository is no longer available after exiting the context
    with pytest.raises(RuntimeError):
        get_config_repository()


def test_config_repository_manager_with_source():
    """Test ConfigRepositoryManager with a source repository."""
    source_repo = ConfigRepository()
    source_repo.set("source_key", "source_value")
    source_repo.set("complex", {"a": [1, 2, 3]})
    
    with ConfigRepositoryManager(source_repo) as repo:
        # Verify values were copied from source
        assert repo.get("source_key") == "source_value"
        assert repo.get("complex") == {"a": [1, 2, 3]}
        
        # Modify the repo and verify source is unchanged
        repo.set("source_key", "changed")
        repo.get("complex")["a"].append(4)
        
        assert source_repo.get("source_key") == "source_value"
        assert source_repo.get("complex")["a"] == [1, 2, 3]
