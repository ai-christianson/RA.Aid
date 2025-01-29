import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from ra_aid.tools import list_directory_tree
from ra_aid.tools.list_directory import load_gitignore_patterns, should_ignore

EXPECTED_YEAR = str(datetime.now().year)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def create_test_directory_structure(path: Path):
    """Create a test directory structure"""
    # Create files
    (path / "file1.txt").write_text("content1")
    (path / "file2.py").write_text("content2")
    (path / ".hidden").write_text("hidden")

    # Create subdirectories
    subdir1 = path / "subdir1"
    subdir1.mkdir()
    (subdir1 / "subfile1.txt").write_text("subcontent1")
    (subdir1 / "subfile2.py").write_text("subcontent2")

    subdir2 = path / "subdir2"
    subdir2.mkdir()
    (subdir2 / ".git").mkdir()
    (subdir2 / "__pycache__").mkdir()


def test_list_directory_basic(temp_dir):
    """Test basic directory listing functionality"""
    create_test_directory_structure(temp_dir)

    result = list_directory_tree.invoke(
        {"path": str(temp_dir), "max_depth": 2, "follow_links": False}
    )

    # Check basic structure
    assert isinstance(result, str)
    assert "file1.txt" in result
    assert "file2.py" in result
    assert "subdir1" in result
    assert "subdir2" in result

    # Hidden files should be excluded by default
    assert ".hidden" not in result
    assert ".git" not in result
    assert "__pycache__" not in result

    # File details should not be present by default
    assert "bytes" not in result.lower()
    assert "2024-" not in result


def test_list_directory_with_details(temp_dir):
    """Test directory listing with file details"""
    create_test_directory_structure(temp_dir)

    result = list_directory_tree.invoke(
        {
            "path": str(temp_dir),
            "max_depth": 2,
            "show_size": True,
            "show_modified": True,
        }
    )

    # File details should be present
    assert "bytes" in result.lower() or "kb" in result.lower() or "b" in result.lower()
    assert f"{EXPECTED_YEAR}-" in result


def test_list_directory_depth_limit(temp_dir):
    """Test max_depth parameter"""
    create_test_directory_structure(temp_dir)

    # Test with depth 1 (default)
    result = list_directory_tree.invoke(
        {
            "path": str(temp_dir)  # Use defaults
        }
    )

    assert isinstance(result, str)
    assert "subdir1" in result  # Directory name should be visible
    assert "subfile1.txt" not in result  # But not its contents
    assert "subfile2.py" not in result


def test_list_directory_ignore_patterns(temp_dir):
    """Test exclude patterns"""
    create_test_directory_structure(temp_dir)

    result = list_directory_tree.invoke(
        {"path": str(temp_dir), "max_depth": 2, "exclude_patterns": ["*.py"]}
    )

    assert isinstance(result, str)
    assert "file1.txt" in result
    assert "file2.py" not in result
    assert "subfile2.py" not in result


def test_gitignore_patterns():
    """Test gitignore pattern loading and matching"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)

        # Create a .gitignore file
        (path / ".gitignore").write_text("*.log\n*.tmp\n")

        spec = load_gitignore_patterns(path)

        assert should_ignore("test.log", spec) is True
        assert should_ignore("test.tmp", spec) is True
        assert should_ignore("test.txt", spec) is False
        assert should_ignore("dir/test.log", spec) is True


def test_invalid_path():
    """Test error handling for invalid paths"""
    with pytest.raises(ValueError, match="Path does not exist"):
        list_directory_tree.invoke({"path": "/nonexistent/path"})

    with pytest.raises(ValueError, match="Path is not a directory"):
        list_directory_tree.invoke(
            {"path": __file__}
        )  # Try to list the test file itself
