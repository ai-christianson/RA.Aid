
from typing import Optional, Tuple, Union, List, Any
from ra_aid.console.formatting import cpm
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

    Uses greedy matching for the content within the tag.

    Args:
        text: Input string that may contain think tags

    Returns:
        A tuple containing:
            - The extracted content from the first think tag (None if no tag found)
            - The remaining string after the first think tag (or the original string if no tag found)
    """
    # Pattern to match think tags at the start of the string (greedy)
    pattern = r"^\s*<think>(.*)</think>"
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


def process_thinking_content(
    content: Union[str, List[Any]],
    supports_think_tag: Optional[bool] = None,
    supports_thinking: bool = False,
    panel_title: str = "💭 Thoughts",
    panel_style: str = None,
    show_thoughts: bool = None,
    logger=None,
) -> Tuple[Union[str, List[Any]], Optional[str]]:
    """Process model response content to extract and optionally display thinking content.

    Handles both string content with <think> tags and structured thinking content (lists).
    Supports implicit <think> tag detection when supports_think_tag is None.

    Args:
        content: The model response content (string or list)
        supports_think_tag: Whether the model supports <think> tags (True=always check,
                          False=never check, None=check if starts with <think>)
        supports_thinking: Whether the model supports structured thinking (list format)
        panel_title: Title to display in the thinking panel
        panel_style: Border style for the panel (None uses default)
        show_thoughts: Whether to display thinking content (if None, checks config)
        logger: Optional logger instance for debug messages

    Returns:
        A tuple containing:
            - The processed content with thinking removed
            - The extracted thinking content (None if no thinking found)
    """
    extracted_thinking = None

    # Determine whether to show thoughts
    if show_thoughts is None:
        try:
            from ra_aid.database.repositories.config_repository import (
                get_config_repository,
            )
            show_thoughts = get_config_repository().get("show_thoughts", False)
        except (ImportError, RuntimeError):
            show_thoughts = False # Default if config cannot be read

    # 1. Handle structured thinking (if supported and content is a list)
    if supports_thinking and isinstance(content, list):
        if logger: logger.debug("Checking for structured thinking content (list format)")
        thinking_items = []
        regular_items = []
        found_thinking = False

        for item in content:
            if isinstance(item, dict) and item.get("type") == "thinking":
                thinking_items.append(item.get("text", ""))
                found_thinking = True
            else:
                regular_items.append(item)

        if found_thinking:
            extracted_thinking = "\n\n".join(thinking_items)
            if logger: logger.debug(f"Found structured thinking content ({len(extracted_thinking)} chars)")

            # Display thinking content if enabled
            if show_thoughts:
                panel_kwargs = {"title": panel_title}
                if panel_style is not None:
                    panel_kwargs["border_style"] = panel_style
                cpm(extracted_thinking, **panel_kwargs)

            # Return remaining items as processed content and the extracted thinking
            return regular_items, extracted_thinking
        else:
            if logger: logger.debug("No structured thinking content found in list.")
        # If no structured thinking found in list, fall through to check string processing

    # 2. Handle string thinking (if content is a string)
    if isinstance(content, str):
        should_extract_tag = False
        # Determine if we should attempt <think> tag extraction based on config/content
        if supports_think_tag is True:
            if logger: logger.debug("Model config supports_think_tag=True, checking for tag.")
            should_extract_tag = True
        elif supports_think_tag is None:
            # Implicit detection: check only if content starts with tag
            if content.strip().startswith("<think>"):
                 if logger: logger.debug("Model config supports_think_tag=None and content starts with <think>, checking for tag.")
                 should_extract_tag = True
            else:
                 # Log only if implicit detection is active but tag not found at start
                 if logger: logger.debug("Model config supports_think_tag=None but content does not start with <think>, skipping tag check.")
        # else: supports_think_tag is False, should_extract_tag remains False. No log here needed for the "both False" case.

        if should_extract_tag:
            if logger: logger.debug("Attempting to extract <think> tag.")
            think_content, remaining_text = extract_think_tag(content)

            if think_content:
                extracted_thinking = think_content
                if logger: logger.debug(f"Found think tag content ({len(think_content)} chars).")

                # Display thinking content if enabled
                if show_thoughts:
                    panel_kwargs = {"title": panel_title}
                    if panel_style is not None:
                        panel_kwargs["border_style"] = panel_style
                    cpm(extracted_thinking, **panel_kwargs)

                # Return remaining text and the extracted thinking
                return remaining_text, extracted_thinking
            else:
                 if logger: logger.debug("No think tag content found despite check.")
        # If should_extract_tag is False, or if extraction failed, fall through.

    # 3. Return original content if no thinking was processed and returned above
    return content, None
