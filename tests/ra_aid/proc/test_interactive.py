"""Tests for the interactive subprocess module."""

import os
import tempfile

import pytest

from ra_aid.proc.interactive import run_interactive_command


def test_basic_command():
    """Test running a basic command."""
    output, retcode = run_interactive_command(["echo", "hello world"])
    assert b"hello world" in output
    assert retcode == 0


def test_shell_pipeline():
    """Test running a shell pipeline command."""
    output, retcode = run_interactive_command(
        ["/bin/bash", "-c", "echo 'hello world' | grep 'world'"]
    )
    assert b"world" in output
    assert retcode == 0


def test_stderr_capture():
    """Test that stderr is properly captured in combined output."""
    # Use a command that definitely writes to stderr
    output, retcode = run_interactive_command(
        ["/bin/bash", "-c", "ls /nonexistent/path"]
    )
    assert b"No such file or directory" in output
    assert retcode != 0  # ls returns 0 upon success


def test_command_not_found():
    """Test handling of non-existent commands."""
    with pytest.raises(FileNotFoundError):
        run_interactive_command(["nonexistentcommand"])


def test_empty_command():
    """Test handling of empty commands."""
    with pytest.raises(ValueError):
        run_interactive_command([])


def test_interactive_command():
    """Test running an interactive command.

    This test verifies that output appears in real-time using process substitution.
    We use a command that prints to both stdout and stderr to verify capture."""
    output, retcode = run_interactive_command(
        ["/bin/bash", "-c", "echo stdout; echo stderr >&2"]
    )
    assert b"stdout" in output
    assert b"stderr" in output
    assert retcode == 0


def test_large_output():
    """Test handling of commands that produce large output."""
    # Generate a large output with predictable content
    cmd = 'for i in {1..10000}; do echo "Line $i of test output"; done'
    output, retcode = run_interactive_command(["/bin/bash", "-c", cmd])

    # Clean up specific artifacts (e.g., ^D)
    output_cleaned = output.lstrip(b"^D")  # Remove the leading ^D if present

    # Split and filter lines
    lines = [
        line.strip()
        for line in output_cleaned.splitlines()
        if b"Script" not in line and line.strip()
    ]

    # Verify we got all 10000 lines
    assert len(lines) == 10000, f"Expected 10000 lines, but got {len(lines)}"

    # Verify content of some lines
    assert lines[0] == b"Line 1 of test output", f"Unexpected line: {lines[0]}"
    assert lines[999] == b"Line 1000 of test output", f"Unexpected line: {lines[999]}"
    assert lines[-1] == b"Line 10000 of test output", f"Unexpected line: {lines[-1]}"

    # Verify return code
    assert retcode == 0, f"Unexpected return code: {retcode}"


def test_unicode_handling():
    """Test handling of unicode characters."""
    test_string = "Hello "
    output, retcode = run_interactive_command(
        ["/bin/bash", "-c", f"echo '{test_string}'"]
    )
    assert test_string.encode() in output
    assert retcode == 0


def test_multiple_commands():
    """Test running multiple commands in sequence."""
    output, retcode = run_interactive_command(
        ["/bin/bash", "-c", "echo 'first'; echo 'second'"]
    )
    assert b"first" in output
    assert b"second" in output
    assert retcode == 0


def test_cat_medium_file():
    """Test that cat command properly captures output for medium-length files."""
    # Create a temporary file with known content
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        for i in range(500):
            f.write(f"This is test line {i}\n")
        temp_path = f.name

    try:
        output, retcode = run_interactive_command(
            ["/bin/bash", "-c", f"cat {temp_path}"]
        )
        # Split by newlines and filter out script header/footer lines
        lines = [
            line
            for line in output.splitlines()
            if b"Script" not in line and line.strip()
        ]
        assert len(lines) == 500
        assert retcode == 0

        # Verify content integrity by checking first and last lines
        assert b"This is test line 0" in lines[0]
        assert b"This is test line 499" in lines[-1]
    finally:
        os.unlink(temp_path)


def test_realtime_output():
    """Test that output appears in real-time and is captured correctly."""
    # Create a command that sleeps briefly between outputs
    cmd = "echo 'first'; sleep 0.1; echo 'second'; sleep 0.1; echo 'third'"
    output, retcode = run_interactive_command(["/bin/bash", "-c", cmd])

    # Filter out script header/footer lines
    lines = [
        line for line in output.splitlines() if b"Script" not in line and line.strip()
    ]

    assert b"first" in lines[0]
    assert b"second" in lines[1]
    assert b"third" in lines[2]
    assert retcode == 0


def test_tty_available():
    """Test that commands have access to a TTY."""
    # Run the tty command
    output, retcode = run_interactive_command(["/bin/bash", "-c", "tty"])

    # Clean up specific artifacts (e.g., ^D)
    output_cleaned = output.lstrip(b"^D")  # Remove leading ^D if present

    # Debug: Print cleaned output
    print(f"Cleaned TTY Output: {output_cleaned}")

    # Check if the output contains a valid TTY path
    assert (
        b"/dev/pts/" in output_cleaned or b"/dev/ttys" in output_cleaned
    ), f"Unexpected TTY output: {output_cleaned}"
    assert retcode == 0
