import os
from unittest.mock import patch

import pytest

from ra_aid.tools.write_file import write_file_tool


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

    result = write_file_tool.invoke({"filepath": str(test_file), "content": content})

    # Verify file contents
    assert test_file.read_text() == content

    # Verify return dict format
    assert isinstance(result, dict)
    assert result["success"] is True
    assert result["filepath"] == str(test_file)
    assert result["bytes_written"] == len(content.encode("utf-8"))
    assert "Operation completed" in result["message"]


def test_directory_creation(temp_test_dir):
    """Test writing to a file in a non-existent directory."""
    nested_dir = temp_test_dir / "nested" / "subdirs"
    test_file = nested_dir / "test.txt"
    content = "Test content"

    result = write_file_tool.invoke({"filepath": str(test_file), "content": content})

    assert test_file.exists()
    assert test_file.read_text() == content
    assert result["success"] is True


def test_different_encodings(temp_test_dir):
    """Test writing files with different encodings."""
    test_file = temp_test_dir / "encoded.txt"
    content = "Hello 世界"  # Mixed ASCII and Unicode

    # Test UTF-8
    result_utf8 = write_file_tool.invoke(
        {"filepath": str(test_file), "content": content, "encoding": "utf-8"}
    )
    assert result_utf8["success"] is True
    assert test_file.read_text(encoding="utf-8") == content

    # Test UTF-16
    result_utf16 = write_file_tool.invoke(
        {"filepath": str(test_file), "content": content, "encoding": "utf-16"}
    )
    assert result_utf16["success"] is True
    assert test_file.read_text(encoding="utf-16") == content


@patch("builtins.open")
def test_permission_error(mock_open_func, temp_test_dir):
    """Test handling of permission errors."""
    mock_open_func.side_effect = PermissionError("Permission denied")
    test_file = temp_test_dir / "noperm.txt"

    result = write_file_tool.invoke(
        {"filepath": str(test_file), "content": "test content"}
    )

    assert result["success"] is False
    assert "Permission denied" in result["message"]
    assert result["error"] is not None


@patch("builtins.open")
def test_io_error(mock_open_func, temp_test_dir):
    """Test handling of IO errors."""
    mock_open_func.side_effect = IOError("IO Error occurred")
    test_file = temp_test_dir / "ioerror.txt"

    result = write_file_tool.invoke(
        {"filepath": str(test_file), "content": "test content"}
    )

    assert result["success"] is False
    assert "IO Error" in result["message"]
    assert result["error"] is not None


def test_empty_content(temp_test_dir):
    """Test writing empty content to a file."""
    test_file = temp_test_dir / "empty.txt"

    result = write_file_tool.invoke({"filepath": str(test_file), "content": ""})

    assert test_file.exists()
    assert test_file.read_text() == ""
    assert result["success"] is True
    assert result["bytes_written"] == 0


def test_overwrite_existing_file(temp_test_dir):
    """Test overwriting an existing file."""
    test_file = temp_test_dir / "overwrite.txt"

    # Write initial content
    test_file.write_text("Initial content")

    # Overwrite with new content
    new_content = "New content"
    result = write_file_tool.invoke(
        {"filepath": str(test_file), "content": new_content}
    )

    assert test_file.read_text() == new_content
    assert result["success"] is True
    assert result["bytes_written"] == len(new_content.encode("utf-8"))


def test_large_file_write(temp_test_dir):
    """Test writing a large file and verify statistics."""
    test_file = temp_test_dir / "large.txt"
    content = "Large content\n" * 1000  # Create substantial content

    result = write_file_tool.invoke({"filepath": str(test_file), "content": content})

    assert test_file.exists()
    assert test_file.read_text() == content
    assert result["success"] is True
    assert result["bytes_written"] == len(content.encode("utf-8"))
    assert os.path.getsize(test_file) == len(content.encode("utf-8"))


def test_invalid_path_characters(temp_test_dir):
    """Test handling of invalid path characters."""
    invalid_path = temp_test_dir / "invalid\0file.txt"

    result = write_file_tool.invoke(
        {"filepath": str(invalid_path), "content": "test content"}
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
        result = write_file_tool.invoke(
            {"filepath": str(test_file), "content": "test content"}
        )
        assert result["success"] is False
        assert "Permission" in result["message"]
    finally:
        # Restore permissions for cleanup
        os.chmod(readonly_dir, 0o755)
