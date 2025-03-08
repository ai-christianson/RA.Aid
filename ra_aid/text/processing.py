from typing import Optional, Tuple
import re


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


def extract_think_tag(text: str) -> Tuple[Optional[str], str]:
    """Extract content from the first <think>...</think> tag at the start of a string.
    
    Args:
        text: Input string that may contain think tags
        
    Returns:
        A tuple containing:
            - The extracted content from the first think tag (None if no tag found)
            - The remaining string after the first think tag (or the original string if no tag found)
    """
    # Pattern to match think tags at the start of the string
    pattern = r'^\s*<think>(.*?)</think>'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        think_content = match.group(1)
        # Get the index where the think tag ends
        end_index = match.end()
        # Extract the remaining text
        remaining_text = text[end_index:]
        return think_content, remaining_text
    else:
        return None, text