from langchain_core.tools import tool
from typing import Dict
from pathlib import Path

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
            return {"success": False, "message": f"File not found: {filepath}"}
            
        content = path.read_text()
        count = content.count(old_str)
        
        if count == 0:
            return {"success": False, "message": f"String not found: {old_str}"}
        elif count > 1:
            return {"success": False, "message": f"String appears {count} times - must be unique"}
            
        new_content = content.replace(old_str, new_str)
        path.write_text(new_content)
        
        return {
            "success": True,
            "message": f"Successfully replaced '{old_str}' with '{new_str}' in {filepath}"
        }
        
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}
