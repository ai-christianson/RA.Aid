"""Tests for utility functions."""

from ra_aid.text.processing import truncate_output


def test_normal_truncation():
    """Test normal truncation behavior with more lines than max."""
    # Create input with 10 lines
    input_lines = [f"Line {i}\n" for i in range(10)]
    input_text = "".join(input_lines)

    # Truncate to 5 lines
    result = truncate_output(input_text, max_lines=5)

    # Verify truncation message and content
    assert "[5 lines of output truncated]" in result
    assert "Line 5\n" in result
    assert "Line 9\n" in result
    assert "Line 0\n" not in result
    assert "Line 4\n" not in result


def test_no_truncation_needed():
    """Test when input is shorter than max_lines."""
    input_text = "Line 1\nLine 2\nLine 3\n"
    result = truncate_output(input_text, max_lines=5)

    # Should return original text unchanged
    assert result == input_text
    assert "[lines of output truncated]" not in result


def test_empty_input():
    """Test with empty input."""
    assert truncate_output("") == ""
    assert truncate_output(None) == ""


def test_exact_max_lines():
    """Test when input is exactly max_lines."""
    # Create input with exactly 5 lines
    input_lines = [f"Line {i}\n" for i in range(5)]
    input_text = "".join(input_lines)

    result = truncate_output(input_text, max_lines=5)

    # Should return original text unchanged
    assert result == input_text
    assert "[lines of output truncated]" not in result


def test_different_line_endings():
    """Test with different line endings (\\n, \\r\\n, \\r)."""
    # Mix of different line endings
    input_text = "Line 1\nLine 2\r\nLine 3\rLine 4\nLine 5\r\nLine 6"

    result = truncate_output(input_text, max_lines=3)

    # Should preserve line endings in truncated output
    assert "[3 lines of output truncated]" in result
    assert "Line 4" in result
    assert "Line 6" in result
    assert "Line 1" not in result


def test_ansi_sequences():
    """Test with ANSI escape sequences."""
    # Input with ANSI color codes
    input_lines = [
        "\033[31mRed Line 1\033[0m\n",
        "\033[32mGreen Line 2\033[0m\n",
        "\033[34mBlue Line 3\033[0m\n",
        "\033[33mYellow Line 4\033[0m\n",
    ]
    input_text = "".join(input_lines)

    result = truncate_output(input_text, max_lines=2)

    # Should preserve ANSI sequences in truncated output
    assert "[2 lines of output truncated]" in result
    assert "\033[34mBlue Line 3\033[0m" in result
    assert "\033[33mYellow Line 4\033[0m" in result
    assert "\033[31mRed Line 1\033[0m" not in result


def test_custom_max_lines():
    """Test with custom max_lines value."""
    # Create input with 100 lines
    input_lines = [f"Line {i}\n" for i in range(100)]
    input_text = "".join(input_lines)

    # Test with custom max_lines=10
    result = truncate_output(input_text, max_lines=10)

    # Should have truncation message and last 10 lines
    assert "[90 lines of output truncated]" in result
    assert "Line 90\n" in result
    assert "Line 99\n" in result
    assert "Line 0\n" not in result
    assert "Line 89\n" not in result


def test_no_trailing_newline():
    """Test with input that doesn't end in newline."""
    input_lines = [f"Line {i}" for i in range(10)]
    input_text = "\n".join(input_lines)  # No trailing newline

    result = truncate_output(input_text, max_lines=5)

    # Should handle truncation correctly without trailing newline
    assert "[5 lines of output truncated]" in result
    assert "Line 5" in result
    assert "Line 9" in result
    assert "Line 0" not in result
    assert "Line 4" not in result
