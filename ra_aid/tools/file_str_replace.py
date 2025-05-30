from pathlib import Path
from typing import Dict

from langchain_core.tools import tool

from ra_aid.console import console
from ra_aid.console.formatting import print_error
from ra_aid.console.formatting import console_panel
from ra_aid.tools.memory import emit_related_files
from ra_aid.database.repositories.trajectory_repository import get_trajectory_repository
from ra_aid.database.repositories.human_input_repository import get_human_input_repository
import logging

logger = logging.getLogger(__name__)


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
    return f"[{len(s)} characters]"


@tool
def file_str_replace(filepath: str, old_str: str, new_str: str, *, replace_all: bool = False) -> Dict[str, any]:
    """Replace an exact string match in a file with a new string.
    Only performs replacement if the old string appears exactly once, or replace_all is True.

    Args:
        filepath: Path to the file to modify
        old_str: Exact string to replace
        new_str: String to replace with
        replace_all: If True, replace all occurrences of the string (default: False)
    """
    try:
        path = Path(filepath)
        if not path.exists():
            msg = f"File not found: {filepath}"
            
            # Record error in trajectory
            try:
                trajectory_repo = get_trajectory_repository()
                human_input_id = get_human_input_repository().get_most_recent_id()
                trajectory_repo.create(
                    step_data={
                        "error_message": msg,
                        "display_title": "Error: File Not Found",
                    },
                    record_type="error",
                    human_input_id=human_input_id,
                    is_error=True,
                    error_message=msg,
                    tool_name="file_str_replace",
                    tool_parameters={
                        "filepath": filepath,
                        "old_str": old_str,
                        "new_str": new_str,
                        "replace_all": replace_all
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to record error trajectory for file_str_replace: {e}")
            
            print_error(msg)
            return {"success": False, "message": msg}

        content = path.read_text()
        count = content.count(old_str)

        if count == 0:
            msg = f"String not found: {truncate_display_str(old_str)}"
            
            # Record error in trajectory
            try:
                trajectory_repo = get_trajectory_repository()
                human_input_id = get_human_input_repository().get_most_recent_id()
                trajectory_repo.create(
                    step_data={
                        "error_message": msg,
                        "display_title": "Error: String Not Found",
                    },
                    record_type="error",
                    human_input_id=human_input_id,
                    is_error=True,
                    error_message=msg,
                    tool_name="file_str_replace",
                    tool_parameters={
                        "filepath": filepath,
                        "old_str": old_str,
                        "new_str": new_str,
                        "replace_all": replace_all
                    }
                )
            except Exception as e:
                 logger.warning(f"Failed to record error trajectory for file_str_replace: {e}")
            
            print_error(msg)
            return {"success": False, "message": msg}
        elif count > 1 and not replace_all:
            msg = f"String appears {count} times - must be unique (use replace_all=True to replace all occurrences)"
            
            # Record error in trajectory
            try:
                trajectory_repo = get_trajectory_repository()
                human_input_id = get_human_input_repository().get_most_recent_id()
                trajectory_repo.create(
                    step_data={
                        "error_message": msg,
                        "display_title": "Error: String Not Unique",
                    },
                    record_type="error",
                    human_input_id=human_input_id,
                    is_error=True,
                    error_message=msg,
                    tool_name="file_str_replace",
                    tool_parameters={
                        "filepath": filepath,
                        "old_str": old_str,
                        "new_str": new_str,
                        "replace_all": replace_all
                    }
                )
            except Exception as e:
                 logger.warning(f"Failed to record error trajectory for file_str_replace: {e}")
            
            print_error(msg)
            return {"success": False, "message": msg}

        new_content = content.replace(old_str, new_str)
        path.write_text(new_content)

        replacement_msg = f"Replaced in {filepath}:"
        if count > 1 and replace_all:
            replacement_msg = f"Replaced {count} occurrences in {filepath}:"
            
        console_panel(
            f"{replacement_msg}\n{format_string_for_display(old_str)} → {format_string_for_display(new_str)}",
            title="✓ String Replaced",
            border_style="bright_blue"
        )
        
        success_msg = f"Successfully replaced '{old_str}' with '{new_str}' in {filepath}"
        if count > 1 and replace_all:
            success_msg = f"Successfully replaced {count} occurrences of '{old_str}' with '{new_str}' in {filepath}"
        
        # Record success trajectory
        try:
            trajectory_repo = get_trajectory_repository()
            human_input_id = get_human_input_repository().get_most_recent_id()
            trajectory_repo.create(
                record_type="file_str_replace",
                tool_name="file_str_replace",
                tool_parameters={
                    "filepath": filepath,
                    "old_str": old_str,
                    "new_str": new_str,
                    "replace_all": replace_all
                },
                step_data={
                    "filepath": filepath,
                    "old_str": truncate_display_str(old_str),
                    "new_str": truncate_display_str(new_str),
                    "count": count
                },
                human_input_id=human_input_id,
                is_error=False,
                display_title=f"Replaced string in {filepath}"
            )
        except Exception as e:
            logger.warning(f"Failed to record success trajectory for file_str_replace: {e}")
        
        # Add file to related files
        try:
            emit_related_files.invoke({"files": [filepath]})
        except Exception as e:
            # Don't let related files error affect main function success
            error_msg = f"Note: Could not add to related files: {str(e)}"
            
            # Record error in trajectory (for related files failure)
            try:
                trajectory_repo = get_trajectory_repository()
                human_input_id = get_human_input_repository().get_most_recent_id()
                trajectory_repo.create(
                    step_data={
                        "error_message": error_msg,
                        "display_title": "Error: Updating Related Files",
                    },
                    record_type="error",
                    human_input_id=human_input_id,
                    is_error=True,
                    error_message=error_msg,
                    tool_name="file_str_replace", # Attributing to the main tool
                    tool_parameters={ # Keep original tool params for context
                        "filepath": filepath,
                        "old_str": old_str,
                        "new_str": new_str,
                        "replace_all": replace_all
                    }
                )
            except Exception as trajectory_e:
                 logger.warning(f"Failed to record error trajectory for related files update: {trajectory_e}")
            
            print_error(error_msg)
            # Note: We still return success for the main operation even if related files fails
            
        return {
            "success": True,
            "message": success_msg,
        }

    except Exception as e:
        msg = f"Error: {str(e)}"
        
        # Record general error in trajectory
        try:
            trajectory_repo = get_trajectory_repository()
            human_input_id = get_human_input_repository().get_most_recent_id()
            trajectory_repo.create(
                step_data={
                    "error_message": msg,
                    "display_title": "Error: General Exception",
                },
                record_type="error",
                human_input_id=human_input_id,
                is_error=True,
                error_message=msg,
                tool_name="file_str_replace",
                tool_parameters={
                    "filepath": filepath,
                    "old_str": old_str, # Best effort, might not be defined if error is early
                    "new_str": new_str, # Best effort
                    "replace_all": replace_all # Best effort
                }
            )
        except Exception as trajectory_e:
             logger.warning(f"Failed to record general error trajectory for file_str_replace: {trajectory_e}")
        
        print_error(msg)
        return {"success": False, "message": msg}
