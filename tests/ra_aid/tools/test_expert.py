from unittest.mock import patch

import pytest

from ra_aid.tools.expert import (
    emit_expert_context,
    expert_context,
    read_files_with_limit,
)


@pytest.fixture
def temp_test_files(tmp_path):
    """Create temporary test files with known content."""
    file1 = tmp_path / "test1.txt"
    file2 = tmp_path / "test2.txt"
    file3 = tmp_path / "test3.txt"

    file1.write_text("Line 1\nLine 2\nLine 3\n")
    file2.write_text("File 2 Line 1\nFile 2 Line 2\n")
    file3.write_text("")  # Empty file

    return tmp_path, [file1, file2, file3]


def test_read_files_with_limit_basic(temp_test_files):
    """Test basic successful reading of multiple files."""
    tmp_path, files = temp_test_files
    result = read_files_with_limit([str(f) for f in files])

    assert "## File:" in result
    assert "Line 1" in result
    assert "File 2 Line 1" in result
    assert str(files[0]) in result
    assert str(files[1]) in result


def test_read_files_with_limit_empty_file(temp_test_files):
    """Test handling of empty files."""
    tmp_path, files = temp_test_files
    result = read_files_with_limit([str(files[2])])  # Empty file
    assert result == ""  # Empty files should be excluded from output


def test_read_files_with_limit_nonexistent_file(temp_test_files):
    """Test handling of nonexistent files."""
    tmp_path, files = temp_test_files
    nonexistent = str(tmp_path / "nonexistent.txt")
    result = read_files_with_limit([str(files[0]), nonexistent])

    assert "Line 1" in result  # Should contain content from existing file
    assert "nonexistent.txt" not in result  # Shouldn't include non-existent file


def test_read_files_with_limit_line_limit(temp_test_files):
    """Test enforcement of line limit."""
    tmp_path, files = temp_test_files
    result = read_files_with_limit([str(files[0]), str(files[1])], max_lines=2)

    assert "truncated" in result
    assert "Line 1" in result
    assert "Line 2" in result
    assert "File 2 Line 1" not in result  # Should be truncated before reaching file 2


@patch("builtins.open")
def test_read_files_with_limit_permission_error(mock_open_func, temp_test_files):
    """Test handling of permission errors."""
    mock_open_func.side_effect = PermissionError("Permission denied")
    tmp_path, files = temp_test_files

    result = read_files_with_limit([str(files[0])])
    assert result == ""  # Should return empty string on permission error


@patch("builtins.open")
def test_read_files_with_limit_io_error(mock_open_func, temp_test_files):
    """Test handling of IO errors."""
    mock_open_func.side_effect = IOError("IO Error")
    tmp_path, files = temp_test_files

    result = read_files_with_limit([str(files[0])])
    assert result == ""  # Should return empty string on IO error


def test_read_files_with_limit_encoding_error(temp_test_files):
    """Test handling of encoding errors."""
    tmp_path, files = temp_test_files

    # Create a file with invalid UTF-8
    invalid_file = tmp_path / "invalid.txt"
    with open(invalid_file, "wb") as f:
        f.write(b"\xff\xfe\x00\x00")  # Invalid UTF-8

    result = read_files_with_limit([str(invalid_file)])
    assert result == ""  # Should return empty string on encoding error


def test_expert_context_management():
    """Test expert context global state management."""
    # Clear any existing context
    expert_context["text"].clear()
    expert_context["files"].clear()

    # Test adding context
    result1 = emit_expert_context.invoke("Test context 1")
    assert "Context added" in result1
    assert len(expert_context["text"]) == 1
    assert expert_context["text"][0] == "Test context 1"

    # Test adding multiple contexts
    result2 = emit_expert_context.invoke("Test context 2")
    assert "Context added" in result2
    assert len(expert_context["text"]) == 2
    assert expert_context["text"][1] == "Test context 2"

    # Test context accumulation
    assert all(
        ctx in expert_context["text"] for ctx in ["Test context 1", "Test context 2"]
    )
