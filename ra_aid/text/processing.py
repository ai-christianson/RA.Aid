from typing import Optional, Tuple, Union, List, Any
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


def process_thinking_content(
    content: Union[str, List[Any]],
    supports_think_tag: bool = False,
    supports_thinking: bool = False,
    panel_title: str = "💭 Thoughts",
    panel_style: str = None,
    show_thoughts: bool = None,
    logger = None,
) -> Tuple[Union[str, List[Any]], Optional[str]]:
    """Process model response content to extract and optionally display thinking content.
    
    This function centralizes the logic for extracting and displaying thinking content
    from model responses, handling both string content with <think> tags and structured
    thinking content (lists).
    
    Args:
        content: The model response content (string or list)
        supports_think_tag: Whether the model supports <think> tags
        supports_thinking: Whether the model supports structured thinking
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
    
    # Skip processing if model doesn't support thinking features
    if not (supports_think_tag or supports_thinking):
        return content, extracted_thinking
    
    # Determine whether to show thoughts
    if show_thoughts is None:
        try:
            from ra_aid.database.repositories.config_repository import get_config_repository
            show_thoughts = get_config_repository().get("show_thoughts", False)
        except (ImportError, RuntimeError):
            show_thoughts = False
    
    # Handle structured thinking content (list format) from models like Claude 3.7
    if isinstance(content, list):
        # Extract thinking items and regular content
        thinking_items = []
        regular_items = []
        
        for item in content:
            if isinstance(item, dict) and item.get("type") == "thinking":
                thinking_items.append(item.get("text", ""))
            else:
                regular_items.append(item)
        
        # If we found thinking items, process them
        if thinking_items:
            extracted_thinking = "\n\n".join(thinking_items)
            
            if logger:
                logger.debug(f"Found structured thinking content ({len(extracted_thinking)} chars)")
            
            # Display thinking content if enabled
            if show_thoughts:
                from rich.panel import Panel
                from rich.markdown import Markdown
                from rich.console import Console
                
                console = Console()
                panel_kwargs = {"title": panel_title}
                if panel_style is not None:
                    panel_kwargs["border_style"] = panel_style
                
                console.print(Panel(Markdown(extracted_thinking), **panel_kwargs))
            
            # Return remaining items as processed content
            return regular_items, extracted_thinking
    
    # Handle string content with potential think tags
    elif isinstance(content, str):
        if logger:
            logger.debug("Checking for think tags in response")
        
        think_content, remaining_text = extract_think_tag(content)
        
        if think_content:
            extracted_thinking = think_content
            if logger:
                logger.debug(f"Found think tag content ({len(think_content)} chars)")
            
            # Display thinking content if enabled
            if show_thoughts:
                from rich.panel import Panel
                from rich.markdown import Markdown
                from rich.console import Console
                
                console = Console()
                panel_kwargs = {"title": panel_title}
                if panel_style is not None:
                    panel_kwargs["border_style"] = panel_style
                
                console.print(Panel(Markdown(think_content), **panel_kwargs))
            
            # Return remaining text as processed content
            return remaining_text, extracted_thinking
        elif logger:
            logger.debug("No think tag content found in response")
    
    # Return the original content if no thinking was found
    return content, extracted_thinking