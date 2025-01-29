import logging
import os
import time
from typing import Dict

from langchain_core.tools import tool
from rich.console import Console
from rich.panel import Panel

console = Console()


@tool
def write_file_tool(
    filepath: str, content: str, encoding: str = "utf-8", verbose: bool = True
) -> Dict[str, any]:
    """Write content to a text file.

    Args:
        filepath: Path to the file to write
        content: String content to write to the file
        encoding: File encoding to use (default: utf-8)
        verbose: Whether to display a Rich panel with write statistics (default: True)

    Returns:
        Dict containing:
            - success: Boolean indicating if write was successful
            - bytes_written: Number of bytes written
            - elapsed_time: Time taken in seconds
            - error: Error message if any (None if successful)

    Raises:
        RuntimeError: If file cannot be written
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
        # Ensure directory exists if filepath contains directories
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        logging.debug(f"Starting to write file: {filepath}")

        with open(filepath, "w", encoding=encoding) as f:
            f.write(content)
            result["bytes_written"] = len(content.encode(encoding))

        elapsed = time.time() - start_time
        result["elapsed_time"] = elapsed
        result["success"] = True
        result["filepath"] = filepath
        result["message"] = "Operation completed successfully"

        logging.debug(
            f"File write complete: {result['bytes_written']} bytes in {elapsed:.2f}s"
        )

        if verbose:
            console.print(
                Panel(
                    f"Wrote {result['bytes_written']} bytes to {filepath} in {elapsed:.2f}s",
                    title="💾 File Write",
                    border_style="bright_green",
                )
            )

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)

        result["elapsed_time"] = elapsed
        result["error"] = error_msg
        if "embedded null byte" in error_msg.lower():
            result["message"] = "Invalid file path: contains null byte character"
        else:
            result["message"] = error_msg

        if verbose:
            console.print(
                Panel(
                    f"Failed to write {filepath}\nError: {error_msg}",
                    title="❌ File Write Error",
                    border_style="red",
                )
            )

    return result
