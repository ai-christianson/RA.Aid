import os
import pytest
from unittest.mock import patch, MagicMock

from ra_aid.tools.programmer import (
    get_aider_executable,
    parse_aider_flags,
    run_programming_task,
)
from ra_aid.database.repositories.related_files_repository import get_related_files_repository
from ra_aid.database.repositories.config_repository import get_config_repository

@pytest.fixture(autouse=True)
def mock_config_repository():
    """Mock the ConfigRepository to avoid database operations during tests"""
    with patch('ra_aid.database.repositories.config_repository.config_repo_var') as mock_repo_var:
        # Setup a mock repository
        mock_repo = MagicMock()
        
        # Create a dictionary to simulate config
        config = {
            "recursion_limit": 2,
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-20241022",
            "temperature": 0.01,
            "aider_config": "/path/to/config.yml"
        }
        
        # Setup get_all method to return the config dict
        mock_repo.get_all.return_value = config
        
        # Setup get method to return config values
        def get_config(key, default=None):
            return config.get(key, default)
        mock_repo.get.side_effect = get_config
        
        # Make the mock context var return our mock repo
        mock_repo_var.get.return_value = mock_repo
        
        yield mock_repo

@pytest.fixture(autouse=True)
def mock_related_files_repository():
    """Mock the RelatedFilesRepository to avoid database operations during tests"""
    with patch('ra_aid.database.repositories.related_files_repository.related_files_repo_var') as mock_repo_var:
        # Setup a mock repository
        mock_repo = MagicMock()
        
        # Create a dictionary to simulate stored files
        related_files = {}
        
        # Setup get_all method to return the files dict
        mock_repo.get_all.return_value = related_files
        
        # Setup format_related_files method
        mock_repo.format_related_files.return_value = [f"ID#{file_id} {filepath}" for file_id, filepath in sorted(related_files.items())]
        
        # Setup add_file method
        def mock_add_file(filepath):
            normalized_path = os.path.abspath(filepath)
            # Check if path already exists
            for file_id, path in related_files.items():
                if path == normalized_path:
                    return file_id
            
            # Add new file
            file_id = len(related_files) + 1
            related_files[file_id] = normalized_path
            return file_id
        mock_repo.add_file.side_effect = mock_add_file
        
        # Setup remove_file method
        def mock_remove_file(file_id):
            if file_id in related_files:
                return related_files.pop(file_id)
            return None
        mock_repo.remove_file.side_effect = mock_remove_file
        
        # Make the mock context var return our mock repo
        mock_repo_var.get.return_value = mock_repo
        
        # Also patch the get_related_files_repository function
        with patch('ra_aid.tools.programmer.get_related_files_repository', return_value=mock_repo):
            yield mock_repo


# Test cases for parse_aider_flags function
test_cases = [
    # Test case format: (input_string, expected_output, test_description)
    (
        "yes-always,dark-mode",
        ["--yes-always", "--dark-mode"],
        "basic comma separated flags without dashes",
    ),
    (
        "--yes-always,--dark-mode",
        ["--yes-always", "--dark-mode"],
        "comma separated flags with dashes",
    ),
    (
        "yes-always, dark-mode",
        ["--yes-always", "--dark-mode"],
        "comma separated flags with space",
    ),
    (
        "--yes-always, --dark-mode",
        ["--yes-always", "--dark-mode"],
        "comma separated flags with dashes and space",
    ),
    ("", [], "empty string"),
    (
        "  yes-always  ,  dark-mode  ",
        ["--yes-always", "--dark-mode"],
        "flags with extra whitespace",
    ),
    ("--yes-always", ["--yes-always"], "single flag with dashes"),
    ("yes-always", ["--yes-always"], "single flag without dashes"),
    # New test cases for flags with values
    (
        "--analytics-log filename.json",
        ["--analytics-log", "filename.json"],
        "flag with value",
    ),
    (
        "--analytics-log filename.json, --model gpt4",
        ["--analytics-log", "filename.json", "--model", "gpt4"],
        "multiple flags with values",
    ),
    (
        "--dark-mode, --analytics-log filename.json",
        ["--dark-mode", "--analytics-log", "filename.json"],
        "mix of simple flags and flags with values",
    ),
    (
        "  --dark-mode  ,  --model  gpt4  ",
        ["--dark-mode", "--model", "gpt4"],
        "flags with values and extra whitespace",
    ),
    (
        "--analytics-log    filename.json",
        ["--analytics-log", "filename.json"],
        "multiple spaces between flag and value",
    ),
    (
        "---dark-mode,----model gpt4",
        ["--dark-mode", "--model", "gpt4"],
        "stripping extra dashes",
    ),
]


@pytest.mark.parametrize("input_flags,expected,description", test_cases)
def test_parse_aider_flags(input_flags, expected, description):
    """Table-driven test for parse_aider_flags function."""
    result = parse_aider_flags(input_flags)
    assert result == expected, f"Failed test case: {description}"


def test_aider_config_flag(monkeypatch, mock_config_repository, mock_related_files_repository):
    """Test that aider config flag is properly included in the command when specified."""
    # Config is mocked by mock_config_repository fixture

    # Mock the run_interactive_command to capture the command that would be run
    mock_run = MagicMock(return_value=(b"", 0))
    monkeypatch.setattr("ra_aid.tools.programmer.run_interactive_command", mock_run)

    run_programming_task.invoke({"instructions": "test instruction"})

    args = mock_run.call_args[0][0]  # Get the first positional arg (command list)
    assert "--config" in args
    config_index = args.index("--config")
    assert args[config_index + 1] == "/path/to/config.yml"


def test_path_normalization_and_deduplication(monkeypatch, tmp_path, mock_config_repository, mock_related_files_repository):
    """Test path normalization and deduplication in run_programming_task."""
    # Create a temporary test file
    test_file = tmp_path / "test.py"
    test_file.write_text("")
    new_file = tmp_path / "new.py"

    # Config is mocked by mock_config_repository fixture
    monkeypatch.setattr(
        "ra_aid.tools.programmer.get_aider_executable", 
        lambda: "/path/to/aider"
    )
    
    mock_run = MagicMock(return_value=(b"", 0))
    monkeypatch.setattr("ra_aid.tools.programmer.run_interactive_command", mock_run)

    # Test duplicate paths
    run_programming_task.invoke(
        {
            "instructions": "test instruction",
            "files": [str(test_file), str(test_file)],  # Same path twice
        }
    )

    # Get the command list passed to run_interactive_command
    cmd_args = mock_run.call_args[0][0]
    # Count occurrences of test_file path in command
    test_file_count = sum(1 for arg in cmd_args if arg == str(test_file))
    assert test_file_count == 1, "Expected exactly one instance of test_file path"

    # Test mixed paths
    run_programming_task.invoke(
        {
            "instructions": "test instruction",
            "files": [str(test_file), str(new_file)],  # Two different paths
        }
    )

    # Get the command list from the second call
    cmd_args = mock_run.call_args[0][0]
    # Verify both paths are present exactly once
    assert (
        sum(1 for arg in cmd_args if arg == str(test_file)) == 1
    ), "Expected one instance of test_file"
    assert (
        sum(1 for arg in cmd_args if arg == str(new_file)) == 1
    ), "Expected one instance of new_file"


def test_get_aider_executable(monkeypatch):
    """Test the get_aider_executable function."""
    # Create mock objects using standard unittest.mock.MagicMock
    mock_path_instance = MagicMock()
    mock_parent = MagicMock()
    mock_aider = MagicMock()
    
    # Setup sys mock
    mock_sys_attrs = {
        "executable": "/path/to/venv/bin/python",
        "platform": "linux"
    }
    monkeypatch.setattr("ra_aid.tools.programmer.sys", MagicMock(**mock_sys_attrs))
    
    # Setup Path mock
    monkeypatch.setattr("ra_aid.tools.programmer.Path", lambda x: mock_path_instance)
    mock_path_instance.parent = mock_parent
    mock_parent.__truediv__.return_value = mock_aider
    mock_aider.exists.return_value = True
    
    # Setup os mock
    mock_os = MagicMock()
    mock_os.access.return_value = True
    mock_os.X_OK = 1
    monkeypatch.setattr("ra_aid.tools.programmer.os", mock_os)

    # Test happy path on Linux
    aider_path = get_aider_executable()
    assert aider_path == str(mock_aider)
    mock_parent.__truediv__.assert_called_with("aider")

    # Test Windows path
    monkeypatch.setattr("ra_aid.tools.programmer.sys.platform", "win32")
    aider_path = get_aider_executable()
    mock_parent.__truediv__.assert_called_with("aider.exe")

    # Test executable not found
    mock_aider.exists.return_value = False
    with pytest.raises(RuntimeError, match="Could not find aider executable"):
        get_aider_executable()

    # Test not executable
    mock_aider.exists.return_value = True
    mock_os.access.return_value = False
    with pytest.raises(RuntimeError, match="is not executable"):
        get_aider_executable()