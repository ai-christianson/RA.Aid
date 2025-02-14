"""Tests for file listing functionality."""

import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from ra_aid.file_listing import (
    DirectoryAccessError,
    DirectoryNotFoundError,
    FileListerError,
    GitCommandError,
    get_file_listing,
    is_git_repo,
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


@pytest.fixture
def git_repo_with_untracked(sample_git_repo):
    """Create a git repository with both tracked and untracked files."""
    # Create untracked files
    untracked_files = ["untracked.txt", "src/untracked.py", "docs/draft.md"]

    for file_path in untracked_files:
        full_path = sample_git_repo / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(f"Untracked content of {file_path}")

    return sample_git_repo


@pytest.fixture
def git_repo_with_ignores(git_repo_with_untracked):
    """Create a git repository with .gitignore rules."""
    # Create .gitignore file
    gitignore_content = """
# Python
__pycache__/
*.pyc

# Project specific
*.log
temp/
ignored.txt
docs/draft.md
"""
    gitignore_path = git_repo_with_untracked / ".gitignore"
    gitignore_path.write_text(gitignore_content)

    # Add and commit .gitignore first
    subprocess.run(["git", "add", ".gitignore"], cwd=git_repo_with_untracked)
    subprocess.run(
        ["git", "commit", "-m", "Add .gitignore"],
        cwd=git_repo_with_untracked,
        env={
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        },
    )

    # Create some ignored files
    ignored_files = [
        "ignored.txt",
        "temp/temp.txt",
        "src/__pycache__/main.cpython-39.pyc",
    ]

    for file_path in ignored_files:
        full_path = git_repo_with_untracked / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(f"Ignored content of {file_path}")

    return git_repo_with_untracked


@pytest.fixture
def git_repo_with_aider_files(sample_git_repo):
    """Create a git repository with .aider files that should be ignored."""
    # Create .aider files
    aider_files = [
        ".aider.chat.history.md",
        ".aider.input.history",
        ".aider.tags.cache.v3/some_file",
        "src/.aider.local.settings",
    ]

    # Create regular files
    regular_files = ["main.cpp", "src/helper.cpp"]

    # Create all files
    for file_path in aider_files + regular_files:
        full_path = sample_git_repo / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(f"Content of {file_path}")

    # Add all files (both .aider and regular) to git
    subprocess.run(["git", "add", "."], cwd=sample_git_repo)
    subprocess.run(
        ["git", "commit", "-m", "Add files including .aider"],
        cwd=sample_git_repo,
        env={
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        },
    )

    return sample_git_repo


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
        "git_output": "\n".join(
            [SINGLE_FILE_NAME, SINGLE_FILE_NAME] + MULTI_FILE_NAMES[1:]
        )
        + "\n",
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


@pytest.fixture
def mock_os_path(monkeypatch):
    """Mock os.path functions."""

    def mock_exists(path):
        return True

    def mock_isdir(path):
        return True

    monkeypatch.setattr(os.path, "exists", mock_exists)
    monkeypatch.setattr(os.path, "isdir", mock_isdir)
    return monkeypatch


@pytest.mark.parametrize("test_case", FILE_LISTING_TEST_CASES, ids=lambda x: x["name"])
def test_get_file_listing(test_case, mock_subprocess, mock_is_git_repo, mock_os_path):
    """Test get_file_listing with various inputs."""
    mock_subprocess.return_value = create_mock_process(test_case["git_output"])
    files, total = get_file_listing(DUMMY_PATH, limit=test_case["limit"])

    assert files == test_case["expected_files"]
    assert total == test_case["expected_total"]


def test_get_file_listing_non_git_repo(mock_is_git_repo, mock_os_path):
    """Test get_file_listing with non-git repository."""
    mock_is_git_repo.return_value = False
    files, total = get_file_listing(DUMMY_PATH)
    assert files == []
    assert total == 0


def test_get_file_listing_git_error(mock_subprocess, mock_is_git_repo, mock_os_path):
    """Test get_file_listing when git command fails."""
    mock_subprocess.side_effect = GitCommandError("Git command failed")
    with pytest.raises(GitCommandError):
        get_file_listing(DUMMY_PATH)


def test_get_file_listing_permission_error(
    mock_subprocess, mock_is_git_repo, mock_os_path
):
    """Test get_file_listing with permission error."""
    mock_subprocess.side_effect = PermissionError("Permission denied")
    with pytest.raises(DirectoryAccessError):
        get_file_listing(DUMMY_PATH)


def test_get_file_listing_unexpected_error(
    mock_subprocess, mock_is_git_repo, mock_os_path
):
    """Test get_file_listing with unexpected error."""
    mock_subprocess.side_effect = Exception("Unexpected error")
    with pytest.raises(FileListerError):
        get_file_listing(DUMMY_PATH)


def test_get_file_listing_with_untracked(git_repo_with_untracked):
    """Test that file listing includes both tracked and untracked files."""
    files, count = get_file_listing(str(git_repo_with_untracked))

    # Check tracked files are present
    assert "README.md" in files
    assert "src/main.py" in files

    # Check untracked files are present
    assert "untracked.txt" in files
    assert "src/untracked.py" in files

    # Verify count includes both tracked and untracked
    expected_count = 8  # 5 tracked + 3 untracked (excluding .gitignore)
    assert count == expected_count


def test_get_file_listing_with_untracked_and_limit(git_repo_with_untracked):
    """Test that file listing with limit works correctly with untracked files."""
    limit = 3
    files, count = get_file_listing(str(git_repo_with_untracked), limit=limit)

    # Total count should still be full count
    assert count == 8  # 5 tracked + 3 untracked (excluding .gitignore)

    # Only limit number of files should be returned
    assert len(files) == limit

    # Files should be sorted, so we can check first 3
    assert files == sorted(files)


def test_get_file_listing_respects_gitignore(git_repo_with_ignores):
    """Test that file listing respects .gitignore rules."""
    # First test with hidden files excluded (default)
    files, count = get_file_listing(str(git_repo_with_ignores))

    # These files should be included (tracked or untracked but not ignored)
    assert "README.md" in files
    assert "src/main.py" in files
    assert "untracked.txt" in files
    assert "src/untracked.py" in files

    # .gitignore should be excluded as it's hidden
    assert ".gitignore" not in files

    # These files should be excluded (ignored)
    assert "ignored.txt" not in files
    assert "temp/temp.txt" not in files
    assert "src/__pycache__/main.cpython-39.pyc" not in files
    assert "docs/draft.md" not in files  # Explicitly ignored in .gitignore

    # Count should include non-ignored, non-hidden files
    expected_count = 7  # 4 tracked + 2 untracked (excluding .gitignore)
    assert count == expected_count

    # Now test with hidden files included
    files, count = get_file_listing(str(git_repo_with_ignores), include_hidden=True)

    # .gitignore should now be included
    assert ".gitignore" in files

    # Count should include non-ignored files plus .gitignore
    expected_count = 8  # 5 tracked + 2 untracked + .gitignore
    assert count == expected_count


def test_aider_files_excluded(git_repo_with_aider_files):
    """Test that .aider files are excluded from the file listing."""
    files, count = get_file_listing(str(git_repo_with_aider_files))

    # Regular files should be included
    assert "main.cpp" in files
    assert "src/helper.cpp" in files

    # .aider files should be excluded
    assert ".aider.chat.history.md" not in files
    assert ".aider.input.history" not in files
    assert ".aider.tags.cache.v3/some_file" not in files
    assert "src/.aider.local.settings" not in files

    # Only the regular files should be counted
    expected_count = 7  # 5 original files from sample_git_repo + 2 new regular files
    assert count == expected_count
    assert len(files) == expected_count


def test_hidden_files_excluded_by_default(git_repo_with_aider_files):
    """Test that hidden files are excluded by default."""
    # Create some hidden files
    hidden_files = [".config", ".env", "src/.local", ".gitattributes"]

    # Create regular files
    regular_files = ["main.cpp", "src/helper.cpp"]

    # Create all files
    for file_path in hidden_files + regular_files:
        full_path = git_repo_with_aider_files / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(f"Content of {file_path}")

    # Add all files to git
    subprocess.run(["git", "add", "."], cwd=git_repo_with_aider_files)
    subprocess.run(
        ["git", "commit", "-m", "Add files including hidden files"],
        cwd=git_repo_with_aider_files,
        env={
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        },
    )

    # Test default behavior (hidden files excluded)
    files, count = get_file_listing(str(git_repo_with_aider_files))

    # Regular files should be included
    assert "main.cpp" in files
    assert "src/helper.cpp" in files

    # Hidden files should be excluded
    for hidden_file in hidden_files:
        assert hidden_file not in files

    # Only regular files should be counted
    assert count == 7  # 5 original files + 2 new regular files

    # Test with include_hidden=True
    files, count = get_file_listing(str(git_repo_with_aider_files), include_hidden=True)

    # Both regular and hidden files should be included
    assert "main.cpp" in files
    assert "src/helper.cpp" in files
    for hidden_file in hidden_files:
        assert hidden_file in files

    # All files should be counted
    assert count == 11  # 5 original + 2 regular + 4 hidden
