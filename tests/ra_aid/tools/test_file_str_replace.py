import os
from unittest.mock import patch

import pytest

from ra_aid.tools.file_str_replace import file_str_replace


@pytest.fixture
def temp_test_dir(tmp_path):
    """Create a temporary test directory."""
    test_dir = tmp_path / "test_replace_dir"
    test_dir.mkdir(exist_ok=True)
    return test_dir


def test_basic_replacement(temp_test_dir):
    """Test basic string replacement functionality."""
    test_file = temp_test_dir / "test.txt"
    initial_content = "Hello world! This is a test."
    test_file.write_text(initial_content)

    result = file_str_replace.invoke(
        {"filepath": str(test_file), "old_str": "world", "new_str": "universe"}
    )

    assert result["success"] is True
    assert test_file.read_text() == "Hello universe! This is a test."
    assert "Successfully replaced" in result["message"]


def test_file_not_found():
    """Test handling of non-existent file."""
    result = file_str_replace.invoke(
        {"filepath": "nonexistent.txt", "old_str": "test", "new_str": "replacement"}
    )

    assert result["success"] is False
    assert "File not found" in result["message"]


def test_string_not_found(temp_test_dir):
    """Test handling of string not present in file."""
    test_file = temp_test_dir / "test.txt"
    test_file.write_text("Hello world!")

    result = file_str_replace.invoke(
        {"filepath": str(test_file), "old_str": "nonexistent", "new_str": "replacement"}
    )

    assert result["success"] is False
    assert "String not found" in result["message"]


def test_multiple_occurrences(temp_test_dir):
    """Test handling of multiple string occurrences."""
    test_file = temp_test_dir / "test.txt"
    test_file.write_text("test test test")

    result = file_str_replace.invoke(
        {"filepath": str(test_file), "old_str": "test", "new_str": "replacement"}
    )

    assert result["success"] is False
    assert "appears" in result["message"]
    assert "must be unique" in result["message"]


def test_empty_strings(temp_test_dir):
    """Test handling of empty strings."""
    test_file = temp_test_dir / "test.txt"
    test_file.write_text("Hello world!")

    # Test empty old string
    result1 = file_str_replace.invoke(
        {"filepath": str(test_file), "old_str": "", "new_str": "replacement"}
    )
    assert result1["success"] is False

    # Test empty new string
    result2 = file_str_replace.invoke(
        {"filepath": str(test_file), "old_str": "world", "new_str": ""}
    )
    assert result2["success"] is True
    assert test_file.read_text() == "Hello !"


def test_special_characters(temp_test_dir):
    """Test handling of special characters."""
    test_file = temp_test_dir / "test.txt"
    initial_content = "Hello\nworld!\t\r\nSpecial chars: $@#%"
    test_file.write_text(initial_content)

    result = file_str_replace.invoke(
        {
            "filepath": str(test_file),
            "old_str": "Special chars: $@#%",
            "new_str": "Replaced!",
        }
    )

    assert result["success"] is True
    assert "Special chars: $@#%" not in test_file.read_text()
    assert "Replaced!" in test_file.read_text()


@patch("pathlib.Path.read_text")
def test_io_error(mock_read_text, temp_test_dir):
    """Test handling of IO errors during read."""
    # Create and write to file first
    test_file = temp_test_dir / "test.txt"
    test_file.write_text("some test content")

    # Then mock read_text to raise error
    mock_read_text.side_effect = IOError("Failed to read file")

    result = file_str_replace.invoke(
        {"filepath": str(test_file), "old_str": "test", "new_str": "replacement"}
    )

    assert result["success"] is False
    assert "Failed to read file" in result["message"]


def test_permission_error(temp_test_dir):
    """Test handling of permission errors."""
    test_file = temp_test_dir / "readonly.txt"
    test_file.write_text("test content")
    os.chmod(test_file, 0o444)  # Make file read-only

    try:
        result = file_str_replace.invoke(
            {"filepath": str(test_file), "old_str": "test", "new_str": "replacement"}
        )

        assert result["success"] is False
        assert "Permission" in result["message"] or "Error" in result["message"]
    finally:
        os.chmod(test_file, 0o644)  # Restore permissions for cleanup


def test_unicode_strings(temp_test_dir):
    """Test handling of Unicode strings."""
    test_file = temp_test_dir / "unicode.txt"
    initial_content = "Hello 世界! Unicode テスト"
    test_file.write_text(initial_content)

    result = file_str_replace.invoke(
        {"filepath": str(test_file), "old_str": "世界", "new_str": "ワールド"}
    )

    assert result["success"] is True
    assert "世界" not in test_file.read_text()
    assert "ワールド" in test_file.read_text()


def test_long_string_truncation(temp_test_dir):
    """Test handling and truncation of very long strings."""
    test_file = temp_test_dir / "test.txt"
    long_string = "x" * 100
    test_file.write_text(f"prefix {long_string} suffix")

    result = file_str_replace.invoke(
        {"filepath": str(test_file), "old_str": long_string, "new_str": "replaced"}
    )

    assert result["success"] is True
    assert test_file.read_text() == "prefix replaced suffix"
