import pytest

from ra_aid.tools import read_file_tool


def test_basic_file_reading(tmp_path):
    """Test basic file reading functionality"""
    # Create a test file
    test_file = tmp_path / "test.txt"
    test_content = "Hello\nWorld\n"
    test_file.write_text(test_content)

    # Read the file
    result = read_file_tool.invoke({"filepath": str(test_file)})

    # Verify return format and content
    assert isinstance(result, dict)
    assert "content" in result
    assert result["content"] == test_content


def test_no_truncation(tmp_path):
    """Test that files under max_lines are not truncated"""
    # Create a test file with content under the limit
    test_file = tmp_path / "small.txt"
    line_count = 4000  # Well under 5000 limit
    test_content = "line\n" * line_count
    test_file.write_text(test_content)

    # Read the file
    result = read_file_tool.invoke({"filepath": str(test_file)})

    # Verify no truncation occurred
    assert isinstance(result, dict)
    assert "[lines of output truncated]" not in result["content"]
    assert len(result["content"].splitlines()) == line_count


def test_with_truncation(tmp_path):
    """Test that files over max_lines are properly truncated"""
    # Create a test file exceeding the limit
    test_file = tmp_path / "large.txt"
    line_count = 6000  # Exceeds 5000 limit
    test_content = "line\n" * line_count
    test_file.write_text(test_content)

    # Read the file
    result = read_file_tool.invoke({"filepath": str(test_file)})

    # Verify truncation occurred correctly
    assert isinstance(result, dict)
    assert "[1000 lines of output truncated]" in result["content"]
    assert (
        len(result["content"].splitlines()) == 5001
    )  # 5000 content lines + 1 truncation message


def test_nonexistent_file():
    """Test error handling for non-existent files"""
    with pytest.raises(FileNotFoundError):
        read_file_tool.invoke({"filepath": "/nonexistent/file.txt"})


def test_empty_file(tmp_path):
    """Test reading an empty file"""
    # Create an empty test file
    test_file = tmp_path / "empty.txt"
    test_file.write_text("")

    # Read the file
    result = read_file_tool.invoke({"filepath": str(test_file)})

    # Verify return format and empty content
    assert isinstance(result, dict)
    assert "content" in result
    assert result["content"] == ""
