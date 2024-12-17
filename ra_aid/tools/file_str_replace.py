from langchain_core.tools import tool
from typing import Dict
from pathlib import Path
from rich.panel import Panel
from ra_aid.console import console
from ra_aid.console.formatting import print_error

def truncate_display_str(s: str, max_length: int = 30) -> str:
    """Truncate a string for display purposes if it exceeds max length.
    
    Args:
        s: String to truncate
        max_length: Maximum length before truncating
        
    Returns:
        Truncated string with ellipsis if needed
    """
    if len(s) <= max_length:
        return s
    return s[:max_length] + "..."

def format_string_for_display(s: str, threshold: int = 30) -> str:
    """Format a string for display, showing either quoted string or length.
    
    Args:
        s: String to format
        threshold: Max length before switching to character count display
        
    Returns:
        Formatted string for display
    """
    if len(s) <= threshold:
        return f"'{s}'"
    return f'[{len(s)} characters]'

@tool
def file_str_replace(
    filepath: str,
    old_str: str,
    new_str: str
) -> Dict[str, any]:
    """Replace an exact string match in a file with a new string.
    Only performs replacement if the old string appears exactly once.
    
    Args:
        filepath: Path to the file to modify
        old_str: Exact string to replace
        new_str: String to replace with
        
    Returns:
        Dict containing:
            - success: Whether the operation succeeded
            - message: Success confirmation or error details
    """
    try:
        path = Path(filepath)
        if not path.exists():
            msg = f"File not found: {filepath}"
            print_error(msg)
            return {"success": False, "message": msg}
            
        content = path.read_text()
        count = content.count(old_str)
        
        if count == 0:
            msg = f"String not found: {truncate_display_str(old_str)}"
            print_error(msg)
            return {"success": False, "message": msg}
        elif count > 1:
            msg = f"String appears {count} times - must be unique"
            print_error(msg)
            return {"success": False, "message": msg}
            
        new_content = content.replace(old_str, new_str)
        path.write_text(new_content)
        
        console.print(Panel(
            f"Replaced in {filepath}:\n{format_string_for_display(old_str)} → {format_string_for_display(new_str)}",
            title="✓ String Replaced",
            border_style="bright_blue"
        ))
        return {
            "success": True,
            "message": f"Successfully replaced '{old_str}' with '{new_str}' in {filepath}"
        }
        
    except Exception as e:
        msg = f"Error: {str(e)}"
        print_error(msg)
        return {"success": False, "message": msg}
