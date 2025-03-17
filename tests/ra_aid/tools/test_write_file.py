import os
from unittest.mock import patch

import pytest

from ra_aid.tools.write_file import put_complete_file_contents


@pytest.fixture(autouse=True)
def mock_related_files_repository():
    """Mock the RelatedFilesRepository to avoid database operations during tests"""
    with patch('ra_aid.tools.memory.get_related_files_repository') as mock_repo:
        # Setup the mock repository to behave like the original, but using memory
        related_files = {}  # Local in-memory storage
        id_counter = 0
        
        # Mock add_file method
        def mock_add_file(filepath):
            nonlocal id_counter
            # Check if normalized path already exists in values
            normalized_path = os.path.abspath(filepath)
            for file_id, path in related_files.items():
                if path == normalized_path:
                    return file_id
                    
            # First check if path exists
            if not os.path.exists(filepath):
                return None
                
            # Then check if it's a directory
            if os.path.isdir(filepath):
                return None
                
            # Validate it's a regular file
            if not os.path.isfile(filepath):
                return None
                
            # Check if it's a binary file (don't actually check in tests)
            # We'll mock is_binary_file separately when needed
            
            # Add new file
            file_id = id_counter
            id_counter += 1
            related_files[file_id] = normalized_path
            
            return file_id
        mock_repo.return_value.add_file.side_effect = mock_add_file
        
        # Mock get method for individual files
        def mock_get(file_id):
            return related_files.get(file_id)
        mock_repo.return_value.get.side_effect = mock_get
        
        # Note: get_all is deprecated, but kept for backward compatibility
        def mock_get_all():
            return related_files.copy()
        mock_repo.return_value.get_all.side_effect = mock_get_all
        
        # Mock remove_file method
        def mock_remove_file(file_id):
            if file_id in related_files:
                return related_files.pop(file_id)
            return None
        mock_repo.return_value.remove_file.side_effect = mock_remove_file
        
        # Mock format_related_files method
        def mock_format_related_files():
            return [f"ID#{file_id} {filepath}" for file_id, filepath in sorted(related_files.items())]
        mock_repo.return_value.format_related_files.side_effect = mock_format_related_files
        
        yield mock_repo


@pytest.fixture
def temp_test_dir(tmp_path):
    """Create a temporary test directory."""
    test_dir = tmp_path / "test_write_dir"
    test_dir.mkdir(exist_ok=True)
    return test_dir


def test_basic_write_functionality(temp_test_dir):
    """Test basic successful file writing."""
    test_file = temp_test_dir / "test.txt"
    content = "Hello, World!\nTest content"

    result = put_complete_file_contents(
        {"filepath": str(test_file), "complete_file_contents": content}
    )

    # Verify file contents
    assert test_file.read_text() == content

    # Verify return dict format
    assert isinstance(result, dict)
    assert result["success"] is True
    assert result["filepath"] == str(test_file)
    assert result["bytes_written"] == len(content.encode("utf-8"))
    assert "Successfully wrote" in result["message"]
    assert "bytes" in result["message"]


def test_directory_creation(temp_test_dir):
    """Test writing to a file in a non-existent directory."""
    nested_dir = temp_test_dir / "nested" / "subdirs"
    test_file = nested_dir / "test.txt"
    content = "Test content"

    result = put_complete_file_contents(
        {"filepath": str(test_file), "complete_file_contents": content}
    )

    assert test_file.exists()
    assert test_file.read_text() == content
    assert result["success"] is True


def test_different_encodings(temp_test_dir):
    """Test writing files with different encodings."""
    test_file = temp_test_dir / "encoded.txt"
    content = "Hello 世界"  # Mixed ASCII and Unicode

    # Test UTF-8
    result_utf8 = put_complete_file_contents(
        {
            "filepath": str(test_file),
            "complete_file_contents": content,
            "encoding": "utf-8",
        }
    )
    assert result_utf8["success"] is True
    assert test_file.read_text(encoding="utf-8") == content

    # Test UTF-16
    result_utf16 = put_complete_file_contents(
        {
            "filepath": str(test_file),
            "complete_file_contents": content,
            "encoding": "utf-16",
        }
    )
    assert result_utf16["success"] is True
    assert test_file.read_text(encoding="utf-16") == content


@patch("builtins.open")
def test_permission_error(mock_open_func, temp_test_dir):
    """Test handling of permission errors."""
    mock_open_func.side_effect = PermissionError("Permission denied")
    test_file = temp_test_dir / "noperm.txt"

    result = put_complete_file_contents(
        {"filepath": str(test_file), "complete_file_contents": "test content"}
    )

    assert result["success"] is False
    assert "Permission denied" in result["message"]
    assert result["error"] is not None


@patch("builtins.open")
def test_io_error(mock_open_func, temp_test_dir):
    """Test handling of IO errors."""
    mock_open_func.side_effect = IOError("IO Error occurred")
    test_file = temp_test_dir / "ioerror.txt"

    result = put_complete_file_contents(
        {"filepath": str(test_file), "complete_file_contents": "test content"}
    )

    assert result["success"] is False
    assert "IO Error" in result["message"]
    assert result["error"] is not None


def test_empty_content(temp_test_dir):
    """Test writing empty content to a file."""
    test_file = temp_test_dir / "empty.txt"

    result = put_complete_file_contents({"filepath": str(test_file)})

    assert test_file.exists()
    assert test_file.read_text() == ""
    assert result["success"] is True
    assert result["bytes_written"] == 0
    assert "initialized empty file" in result["message"].lower()


def test_write_empty_file_default(temp_test_dir):
    """Test creating an empty file using default parameter."""
    test_file = temp_test_dir / "empty_default.txt"

    result = put_complete_file_contents({"filepath": str(test_file)})

    assert test_file.exists()
    assert test_file.read_text() == ""
    assert result["success"] is True
    assert result["bytes_written"] == 0
    assert "initialized empty file" in result["message"].lower()


def test_overwrite_existing_file(temp_test_dir):
    """Test overwriting an existing file."""
    test_file = temp_test_dir / "overwrite.txt"

    # Write initial content
    test_file.write_text("Initial content")

    # Overwrite with new content
    new_content = "New content"
    result = put_complete_file_contents(
        {"filepath": str(test_file), "complete_file_contents": new_content}
    )

    assert test_file.read_text() == new_content
    assert result["success"] is True
    assert result["bytes_written"] == len(new_content.encode("utf-8"))


def test_large_file_write(temp_test_dir):
    """Test writing a large file and verify statistics."""
    test_file = temp_test_dir / "large.txt"
    content = "Large content\n" * 1000  # Create substantial content

    result = put_complete_file_contents(
        {"filepath": str(test_file), "complete_file_contents": content}
    )

    assert test_file.exists()
    assert test_file.read_text() == content
    assert result["success"] is True
    assert result["bytes_written"] == len(content.encode("utf-8"))
    assert os.path.getsize(test_file) == len(content.encode("utf-8"))


def test_invalid_path_characters(temp_test_dir):
    """Test handling of invalid path characters."""
    invalid_path = temp_test_dir / "invalid\0file.txt"

    result = put_complete_file_contents(
        {"filepath": str(invalid_path), "complete_file_contents": "test content"}
    )

    assert result["success"] is False
    assert "Invalid file path" in result["message"]


def test_write_to_readonly_directory(temp_test_dir):
    """Test writing to a readonly directory."""
    readonly_dir = temp_test_dir / "readonly"
    readonly_dir.mkdir()
    test_file = readonly_dir / "test.txt"

    # Make directory readonly
    os.chmod(readonly_dir, 0o444)

    try:
        result = put_complete_file_contents(
            {"filepath": str(test_file), "complete_file_contents": "test content"}
        )
        assert result["success"] is False
        assert "Permission" in result["message"]
    finally:
        # Restore permissions for cleanup
        os.chmod(readonly_dir, 0o755)
