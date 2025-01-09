"""Tests for file listing functionality."""

import os
import pytest
from pathlib import Path
import subprocess

from ra_aid.file_listing import (
    get_file_listing,
    is_git_repo,
    DirectoryNotFoundError,
    DirectoryAccessError,
    GitCommandError,
    FileListerError
)


@pytest.fixture
def empty_git_repo(tmp_path):
    """Create an empty git repository."""
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    return tmp_path


@pytest.fixture
def sample_git_repo(empty_git_repo):
    """Create a git repository with sample files."""
    # Create some files
    files = [
        "README.md",
        "src/main.py",
        "src/utils.py",
        "tests/test_main.py",
        "docs/index.html"
    ]
    
    for file_path in files:
        full_path = empty_git_repo / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(f"Content of {file_path}")
    
    # Add and commit files
    subprocess.run(["git", "add", "."], cwd=empty_git_repo)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=empty_git_repo,
        env={"GIT_AUTHOR_NAME": "Test", 
             "GIT_AUTHOR_EMAIL": "test@example.com",
             "GIT_COMMITTER_NAME": "Test",
             "GIT_COMMITTER_EMAIL": "test@example.com"}
    )
    
    return empty_git_repo


def test_is_git_repo(sample_git_repo, tmp_path_factory):
    """Test git repository detection."""
    # Create a new directory that is not a git repository
    non_repo_dir = tmp_path_factory.mktemp("non_repo")
    # Assert that sample_git_repo is identified as a git repository
    assert is_git_repo(str(sample_git_repo)) is True
    # Assert that non_repo_dir is not identified as a git repository
    assert is_git_repo(str(non_repo_dir)) is False


def test_get_file_listing_no_limit(sample_git_repo):
    """Test getting complete file listing."""
    files, total = get_file_listing(str(sample_git_repo))
    assert len(files) == 5
    assert total == 5
    assert "README.md" in files
    assert "src/main.py" in files
    assert all(isinstance(f, str) for f in files)


def test_get_file_listing_with_limit(sample_git_repo):
    """Test file listing with limit."""
    files, total = get_file_listing(str(sample_git_repo), limit=2)
    assert len(files) == 2
    assert total == 5  # Total should still be 5


def test_empty_git_repo(empty_git_repo):
    """Test handling of empty git repository."""
    files, total = get_file_listing(str(empty_git_repo))
    assert len(files) == 0
    assert total == 0


def test_non_git_directory(tmp_path):
    """Test handling of non-git directory."""
    files, total = get_file_listing(str(tmp_path))
    assert len(files) == 0
    assert total == 0


def test_nonexistent_directory():
    """Test handling of non-existent directory."""
    with pytest.raises(DirectoryNotFoundError):
        get_file_listing("/nonexistent/path/123456")


def test_file_as_directory(tmp_path):
    """Test handling of file path instead of directory."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test")
    
    with pytest.raises(DirectoryNotFoundError):
        get_file_listing(str(test_file))


@pytest.mark.skipif(os.name == "nt", reason="Permission tests unreliable on Windows")
def test_permission_error(tmp_path):
    """Test handling of permission errors."""
    try:
        # Make directory unreadable
        os.chmod(tmp_path, 0o000)
        
        with pytest.raises(DirectoryAccessError):
            get_file_listing(str(tmp_path))
    finally:
        # Restore permissions to allow cleanup
        os.chmod(tmp_path, 0o755)
