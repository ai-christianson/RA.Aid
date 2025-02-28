import logging
import os
import time
from typing import Dict

from langchain_core.tools import tool
from rich.console import Console
from rich.panel import Panel

console = Console()


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

        console.print(
            Panel(
                f"{'Initialized empty file' if not complete_file_contents else f'Wrote {bytes_written} bytes'} at {filepath} in {elapsed:.2f}s",
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

        console.print(
            Panel(
                f"Failed to write {filepath}\nError: {error_msg}",
                title="❌ File Write Error",
                border_style="red",
            )
        )

    return result
