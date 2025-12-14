import logging
import os
import time
from typing import Dict

from langchain_core.tools import tool
from rich.console import Console
from rich.panel import Panel
from ra_aid.console.formatting import console_panel
from ra_aid.database.repositories.trajectory_repository import get_trajectory_repository  # Added import
from ra_aid.tools.memory import emit_related_files

console = Console()


# Guardrail thresholds
MIN_LINES_FOR_PROTECTION = 20  # Files with fewer lines aren't protected
MIN_BYTES_FOR_PROTECTION = 500  # Files smaller than this aren't protected
MAX_CONTENT_LOSS_RATIO = 0.5  # Refuse if new content is less than 50% of original


def _check_destructive_write(filepath: str, new_content: str, encoding: str = "utf-8") -> str | None:
    """
    Check if a write operation would be destructive.

    Returns:
        Error message if write should be blocked, None if safe to proceed.
    """
    if not os.path.exists(filepath):
        return None  # New file, always safe

    try:
        with open(filepath, "r", encoding=encoding) as f:
            old_content = f.read()
    except Exception:
        return None  # Can't read, let the write proceed

    old_lines = len(old_content.splitlines())
    old_bytes = len(old_content.encode(encoding))
    new_bytes = len(new_content.encode(encoding))

    # Skip protection for small files
    if old_lines < MIN_LINES_FOR_PROTECTION and old_bytes < MIN_BYTES_FOR_PROTECTION:
        return None

    # Check for destructive write (empty or massive reduction)
    if new_bytes == 0 and old_bytes > MIN_BYTES_FOR_PROTECTION:
        return (
            f"GUARDRAIL: Refusing to write empty content to {filepath} "
            f"which has {old_lines} lines ({old_bytes} bytes). "
            f"This looks like accidental file deletion."
        )

    if old_bytes > 0 and new_bytes < old_bytes * MAX_CONTENT_LOSS_RATIO:
        loss_percent = int((1 - new_bytes / old_bytes) * 100)
        return (
            f"GUARDRAIL: Refusing to reduce {filepath} from {old_bytes} to {new_bytes} bytes "
            f"({loss_percent}% content loss). This looks like accidental deletion. "
            f"If intentional, delete the file first then recreate it."
        )

    return None


@tool
def put_complete_file_contents(
    filepath: str,
    complete_file_contents: str = "",
    encoding: str = "utf-8",
) -> Dict[str, any]:
    """Write the complete contents of a file, creating it if it doesn't exist.
    This tool is specifically for writing the entire contents of a file at once,
    not for appending or partial writes.

    If you need to do anything other than write the complete contents use the run_programming_task tool instead.

    Args:
        filepath: (Required) Path to the file to write. Must be provided.
        complete_file_contents: Complete string content to write to the file. Defaults to
                              an empty string, which will create an empty file.
        encoding: File encoding to use (default: utf-8)
    """
    start_time = time.time()
    result = {
        "success": False,
        "bytes_written": 0,
        "elapsed_time": 0,
        "error": None,
        "filepath": None,
        "message": None,
    }

    try:
        # GUARDRAIL: Check for destructive writes
        guardrail_error = _check_destructive_write(filepath, complete_file_contents, encoding)
        if guardrail_error:
            console_panel(
                guardrail_error,
                title="🛡️ Write Blocked",
                border_style="red bold",
            )
            result["error"] = guardrail_error
            result["message"] = guardrail_error
            return result

        # Ensure directory exists if filepath contains directories
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        logging.debug(f"Starting to write file: {filepath}")

        with open(filepath, "w", encoding=encoding) as f:
            logging.debug(f"Writing {len(complete_file_contents)} bytes to {filepath}")
            f.write(complete_file_contents)
            result["bytes_written"] = len(complete_file_contents.encode(encoding))

        elapsed = time.time() - start_time
        bytes_written = result["bytes_written"]
        result["elapsed_time"] = elapsed
        result["success"] = True
        result["filepath"] = filepath
        result["message"] = (
            f"Successfully {'initialized empty file' if not complete_file_contents else f'wrote {bytes_written} bytes'} "
            f"at {filepath} in {result['elapsed_time']:.3f}s"
        )

        logging.debug(f"File write complete: {bytes_written} bytes in {elapsed:.2f}s")

        # Create trajectory record for successful file write
        trajectory_repo = get_trajectory_repository()
        trajectory_repo.create(
            record_type="file_write",
            tool_name="put_complete_file_contents",
            tool_parameters={"filepath": filepath, "encoding": encoding},
            step_data={"filepath": filepath, "bytes_written": result["bytes_written"]},
            is_error=False,
        )

        console_panel(
            f"{'Initialized empty file' if not complete_file_contents else f'Wrote {bytes_written} bytes'} at {filepath} in {elapsed:.2f}s",
            title="💾 File Write",
            border_style="bright_green",
        )

        # Add file to related files
        emit_related_files.invoke({"files": [filepath]})

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)

        result["elapsed_time"] = elapsed
        result["error"] = error_msg
        if "embedded null byte" in error_msg.lower():
            result["message"] = "Invalid file path: contains null byte character"
        else:
            result["message"] = error_msg

        console_panel(
            f"Failed to write {filepath}\nError: {error_msg}",
            title="❌ File Write Error",
            border_style="red",
        )

    return result
