from typing import Optional


def truncate_output(output: str, max_lines: Optional[int] = 5000) -> str:
    """Truncate output string to keep only the most recent lines if it exceeds max_lines.

    When truncation occurs, adds a message indicating how many lines were removed.
    Preserves original line endings and handles Unicode characters correctly.

    Args:
        output: The string output to potentially truncate
        max_lines: Maximum number of lines to keep (default: 5000)

    Returns:
        The truncated string if it exceeded max_lines, or the original string if not
    """
    # Handle empty output
    if not output:
        return ""

    # Set max_lines to default if None
    if max_lines is None:
        max_lines = 5000

    # Split while preserving line endings
    lines = output.splitlines(keepends=True)
    total_lines = len(lines)

    # Return original if under limit
    if total_lines <= max_lines:
        return output

    # Calculate lines to remove
    lines_removed = total_lines - max_lines

    # Keep only the most recent lines
    truncated_lines = lines[-max_lines:]

    # Add truncation message at start
    truncation_msg = f"[{lines_removed} lines of output truncated]\n"

    # Combine message with remaining lines
    return truncation_msg + "".join(truncated_lines)
