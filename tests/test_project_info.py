"""Tests for project info functionality."""

import os
import subprocess

import pytest

from ra_aid.project_info import ProjectInfo, get_project_info
from ra_aid.project_state import DirectoryAccessError, DirectoryNotFoundError


@pytest.fixture
def empty_git_repo(tmp_path):
    """Create an empty git repository."""
    import subprocess

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
        "docs/index.html",
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
        env={
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        },
    )

    return empty_git_repo


def test_empty_git_repo(empty_git_repo):
    """Test project info for empty git repository."""
    info = get_project_info(str(empty_git_repo))
    assert isinstance(info, ProjectInfo)
    assert info.is_new is True
    assert len(info.files) == 0
    assert info.total_files == 0


def test_sample_git_repo(sample_git_repo):
    """Test project info for repository with files."""
    info = get_project_info(str(sample_git_repo))
    assert isinstance(info, ProjectInfo)
    assert info.is_new is False
    assert len(info.files) == 5
    assert info.total_files == 5
    assert "README.md" in info.files


def test_file_limit(sample_git_repo):
    """Test file listing with limit."""
    info = get_project_info(str(sample_git_repo), file_limit=2)
    assert len(info.files) == 2
    assert info.total_files == 5  # Total should still be 5


def test_nonexistent_directory():
    """Test handling of non-existent directory."""
    with pytest.raises(DirectoryNotFoundError):
        get_project_info("/nonexistent/path/123456")


def test_file_as_directory(tmp_path):
    """Test handling of file path instead of directory."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test")

    with pytest.raises(DirectoryNotFoundError):
        get_project_info(str(test_file))


@pytest.mark.skipif(os.name == "nt", reason="Permission tests unreliable on Windows")
def test_permission_error(tmp_path):
    """Test handling of permission errors."""
    try:
        # Make directory unreadable
        os.chmod(tmp_path, 0o000)

        with pytest.raises(DirectoryAccessError):
            get_project_info(str(tmp_path))
    finally:
        # Restore permissions to allow cleanup
        os.chmod(tmp_path, 0o755)
