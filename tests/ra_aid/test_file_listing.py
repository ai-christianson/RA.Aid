"""Tests for file listing functionality."""

import os
import pytest
from unittest.mock import patch, MagicMock
from ra_aid.file_listing import (
    get_file_listing,
    is_git_repo,
    GitCommandError,
    DirectoryNotFoundError,
    DirectoryAccessError,
    FileListerError,
)

# Constants for test data
DUMMY_PATH = "dummy/path"
EMPTY_FILE_LIST = []
EMPTY_FILE_TOTAL = 0
SINGLE_FILE_NAME = "file1.txt"
MULTI_FILE_NAMES = ["file1.txt", "file2.py", "file3.md"]

# Test cases for get_file_listing
FILE_LISTING_TEST_CASES = [
    {
        "name": "empty_repository",
        "git_output": "",
        "expected_files": EMPTY_FILE_LIST,
        "expected_total": EMPTY_FILE_TOTAL,
        "limit": None,
    },
    {
        "name": "single_file",
        "git_output": f"{SINGLE_FILE_NAME}\n",
        "expected_files": [SINGLE_FILE_NAME],
        "expected_total": 1,
        "limit": None,
    },
    {
        "name": "multiple_files",
        "git_output": "\n".join(MULTI_FILE_NAMES) + "\n",
        "expected_files": MULTI_FILE_NAMES,
        "expected_total": len(MULTI_FILE_NAMES),
        "limit": None,
    },
    {
        "name": "duplicate_files",
        "git_output": "\n".join([SINGLE_FILE_NAME, SINGLE_FILE_NAME] + MULTI_FILE_NAMES[1:]) + "\n",
        "expected_files": [SINGLE_FILE_NAME] + MULTI_FILE_NAMES[1:],
        "expected_total": 3,  # After deduplication
        "limit": None,
    },
    {
        "name": "with_limit",
        "git_output": "\n".join(MULTI_FILE_NAMES) + "\n",
        "expected_files": MULTI_FILE_NAMES[:2],
        "expected_total": len(MULTI_FILE_NAMES),
        "limit": 2,
    },
    {
        "name": "with_empty_lines",
        "git_output": f"\n{SINGLE_FILE_NAME}\n\n{MULTI_FILE_NAMES[1]}\n\n",
        "expected_files": [SINGLE_FILE_NAME, MULTI_FILE_NAMES[1]],
        "expected_total": 2,
        "limit": None,
    },
    {
        "name": "with_whitespace",
        "git_output": f"  {SINGLE_FILE_NAME}  \n  {MULTI_FILE_NAMES[1]}  \n",
        "expected_files": [SINGLE_FILE_NAME, MULTI_FILE_NAMES[1]],
        "expected_total": 2,
        "limit": None,
    },
    {
        "name": "limit_larger_than_total",
        "git_output": f"{SINGLE_FILE_NAME}\n{MULTI_FILE_NAMES[1]}\n",
        "expected_files": [SINGLE_FILE_NAME, MULTI_FILE_NAMES[1]],
        "expected_total": 2,
        "limit": 5,
    },
    {
        "name": "limit_zero",
        "git_output": "\n".join(MULTI_FILE_NAMES) + "\n",
        "expected_files": EMPTY_FILE_LIST,
        "expected_total": len(MULTI_FILE_NAMES),
        "limit": 0,
    },
    {
        "name": "nested_paths",
        "git_output": "dir1/file1.txt\ndir1/dir2/file2.py\nfile3.md\n",
        "expected_files": sorted(["dir1/file1.txt", "dir1/dir2/file2.py", "file3.md"]),
        "expected_total": 3,
        "limit": None,
    },
    {
        "name": "special_characters",
        "git_output": "file-1.txt\nfile_2.py\nfile 3.md\n",
        "expected_files": sorted(["file-1.txt", "file_2.py", "file 3.md"]),
        "expected_total": 3,
        "limit": None,
    },
    {
        "name": "duplicate_nested_paths",
        "git_output": "dir1/file1.txt\ndir1/file1.txt\ndir2/file1.txt\n",
        "expected_files": sorted(["dir1/file1.txt", "dir2/file1.txt"]),
        "expected_total": 2,
        "limit": None,
    },
]

def create_mock_process(git_output: str) -> MagicMock:
    """Create a mock process with the given git output."""
    mock_process = MagicMock()
    mock_process.stdout = git_output
    mock_process.returncode = 0
    return mock_process

@pytest.fixture
def mock_subprocess():
    """Fixture to mock subprocess.run."""
    with patch("subprocess.run") as mock_run:
        yield mock_run

@pytest.fixture
def mock_is_git_repo():
    """Fixture to mock is_git_repo function."""
    with patch("ra_aid.file_listing.is_git_repo") as mock:
        mock.return_value = True
        yield mock

@pytest.mark.parametrize("test_case", FILE_LISTING_TEST_CASES, ids=lambda x: x["name"])
def test_get_file_listing(test_case, mock_subprocess, mock_is_git_repo):
    """Test get_file_listing with various inputs."""
    mock_subprocess.return_value = create_mock_process(test_case["git_output"])
    files, total = get_file_listing(DUMMY_PATH, limit=test_case["limit"])
    assert files == test_case["expected_files"]
    assert total == test_case["expected_total"]

def test_get_file_listing_non_git_repo(mock_is_git_repo):
    """Test get_file_listing with non-git repository."""
    mock_is_git_repo.return_value = False
    files, total = get_file_listing(DUMMY_PATH)
    assert files == EMPTY_FILE_LIST
    assert total == EMPTY_FILE_TOTAL

def test_get_file_listing_git_error(mock_subprocess, mock_is_git_repo):
    """Test get_file_listing when git command fails."""
    mock_subprocess.side_effect = GitCommandError("Git command failed")
    with pytest.raises(GitCommandError):
        get_file_listing(DUMMY_PATH)

def test_get_file_listing_permission_error(mock_subprocess, mock_is_git_repo):
    """Test get_file_listing with permission error."""
    mock_subprocess.side_effect = PermissionError("Permission denied")
    with pytest.raises(DirectoryAccessError):
        get_file_listing(DUMMY_PATH)

def test_get_file_listing_unexpected_error(mock_subprocess, mock_is_git_repo):
    """Test get_file_listing with unexpected error."""
    mock_subprocess.side_effect = Exception("Unexpected error")
    with pytest.raises(FileListerError):
        get_file_listing(DUMMY_PATH)
