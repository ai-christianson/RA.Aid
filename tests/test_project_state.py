"""Tests for project state detection functionality."""

import os

import pytest

from ra_aid.project_state import (
    DirectoryAccessError,
    DirectoryNotFoundError,
    ProjectStateError,
    is_new_project,
)


@pytest.fixture
def empty_dir(tmp_path):
    """Create an empty temporary directory."""
    return tmp_path


@pytest.fixture
def git_only_dir(tmp_path):
    """Create a directory with only git files."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("*.pyc\n")
    return tmp_path


@pytest.fixture
def project_dir(tmp_path):
    """Create a directory with some project files."""
    (tmp_path / "src").mkdir()
    (tmp_path / "README.md").write_text("# Test Project")
    return tmp_path


def test_empty_directory(empty_dir):
    """Test that an empty directory is considered a new project."""
    assert is_new_project(str(empty_dir)) is True


def test_git_only_directory(git_only_dir):
    """Test that a directory with only git files is considered a new project."""
    assert is_new_project(str(git_only_dir)) is True


@pytest.fixture
def git_dir_with_contents(tmp_path):
    """Create a directory with .git containing files and .gitignore."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    # Add some files inside .git
    (git_dir / "HEAD").write_text("ref: refs/heads/main")
    (git_dir / "config").write_text("[core]\n\trepositoryformatversion = 0")
    (git_dir / "refs").mkdir()
    # Add .gitignore
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("*.pyc\n")
    return tmp_path


def test_git_directory_with_contents(git_dir_with_contents):
    """Test that a directory is considered new even with files inside .git."""
    assert is_new_project(str(git_dir_with_contents)) is True


def test_existing_project_directory(project_dir):
    """Test that a directory with project files is not considered new."""
    assert is_new_project(str(project_dir)) is False


def test_nonexistent_directory():
    """Test that a non-existent directory raises appropriate error."""
    with pytest.raises(DirectoryNotFoundError):
        is_new_project("/nonexistent/path/123456")


def test_file_as_directory(tmp_path):
    """Test that passing a file instead of directory raises error."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test")

    with pytest.raises(ProjectStateError):
        is_new_project(str(test_file))


@pytest.mark.skipif(os.name == "nt", reason="Permission tests unreliable on Windows")
def test_permission_error(tmp_path):
    """Test handling of permission errors."""
    try:
        # Make directory unreadable
        os.chmod(tmp_path, 0o000)

        with pytest.raises(DirectoryAccessError):
            is_new_project(str(tmp_path))
    finally:
        # Restore permissions to allow cleanup
        os.chmod(tmp_path, 0o755)
