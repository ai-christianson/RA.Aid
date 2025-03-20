import logging
import os.path
import time
from typing import Dict, Optional

from langchain_core.tools import tool

from ra_aid.text.processing import truncate_output
from ra_aid.tools.memory import is_binary_file
from ra_aid.console.formatting import console_panel, cpm

# Standard buffer size for file reading
CHUNK_SIZE = 8192


def record_trajectory(
    tool_name: str,
    tool_parameters: Dict,
    step_data: Dict,
    record_type: str = "tool_execution",
    is_error: bool = False,
    error_message: Optional[str] = None,
    error_type: Optional[str] = None
) -> None:
    """
    Helper function to record trajectory information, handling the case when repositories are not available.
    
    Args:
        tool_name: Name of the tool
        tool_parameters: Parameters passed to the tool
        step_data: UI rendering data
        record_type: Type of trajectory record
        is_error: Flag indicating if this record represents an error
        error_message: The error message
        error_type: The type/class of the error
    """
    try:
        from ra_aid.database.repositories.trajectory_repository import get_trajectory_repository
        from ra_aid.database.repositories.human_input_repository import get_human_input_repository
        
        trajectory_repo = get_trajectory_repository()
        human_input_id = get_human_input_repository().get_most_recent_id()
        trajectory_repo.create(
            tool_name=tool_name,
            tool_parameters=tool_parameters,
            step_data=step_data,
            record_type=record_type,
            human_input_id=human_input_id,
            is_error=is_error,
            error_message=error_message,
            error_type=error_type
        )
    except (ImportError, RuntimeError):
        # If either the repository modules can't be imported or no repository is available,
        # just log and continue without recording trajectory
        logging.debug("Skipping trajectory recording: repositories not available")


@tool
def read_file_tool(filepath: str, encoding: str = "utf-8") -> Dict[str, str]:
    """Read and return the contents of a text file.

    Args:
        filepath: Path to the file to read
        encoding: File encoding to use (default: utf-8)
    
    DO NOT ATTEMPT TO READ BINARY FILES
    """
    start_time = time.time()
    try:
        if not os.path.exists(filepath):
            # Record error in trajectory
            record_trajectory(
                tool_name="read_file_tool",
                tool_parameters={
                    "filepath": filepath,
                    "encoding": encoding
                },
                step_data={
                    "filepath": filepath,
                    "display_title": "File Not Found",
                    "error_message": f"File not found: {filepath}"
                },
                is_error=True,
                error_message=f"File not found: {filepath}",
                error_type="FileNotFoundError"
            )
            raise FileNotFoundError(f"File not found: {filepath}")

        # Check if the file is binary
        if is_binary_file(filepath):
            # Record binary file error in trajectory
            record_trajectory(
                tool_name="read_file_tool",
                tool_parameters={
                    "filepath": filepath,
                    "encoding": encoding
                },
                step_data={
                    "filepath": filepath,
                    "display_title": "Binary File Detected",
                    "error_message": f"Cannot read binary file: {filepath}"
                },
                is_error=True,
                error_message="Cannot read binary file",
                error_type="BinaryFileError"
            )
            
            console_panel(
                f"Cannot read binary file: {filepath}",
                title="⚠ Binary File Detected",
                border_style="bright_red",
            )
            return {"error": "read_file failed because we cannot read binary files"}

        logging.debug(f"Starting to read file: {filepath}")
        content = []
        line_count = 0
        total_bytes = 0

        with open(filepath, "r", encoding=encoding) as f:
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break

                content.append(chunk)
                total_bytes += len(chunk)
                line_count += chunk.count("\n")

                logging.debug(
                    f"Read chunk: {len(chunk)} bytes, running total: {total_bytes} bytes"
                )

        full_content = "".join(content)
        elapsed = time.time() - start_time

        logging.debug(f"File read complete: {total_bytes} bytes in {elapsed:.2f}s")
        logging.debug(f"Pre-truncation stats: {total_bytes} bytes, {line_count} lines")

        # Record successful file read in trajectory
        record_trajectory(
            tool_name="read_file_tool",
            tool_parameters={
                "filepath": filepath,
                "encoding": encoding
            },
            step_data={
                "filepath": filepath,
                "display_title": "File Read",
                "line_count": line_count,
                "total_bytes": total_bytes,
                "elapsed_time": elapsed
            }
        )

        console_panel(
            f"Read {line_count} lines ({total_bytes} bytes) from {filepath}",
            title="📄 File Read",
            border_style="bright_blue",
        )

        # Truncate if needed
        truncated = truncate_output(full_content) if full_content else ""

        return {"content": truncated}

    except Exception as e:
        elapsed = time.time() - start_time
        
        # Record exception in trajectory (if it's not already a handled FileNotFoundError)
        if not isinstance(e, FileNotFoundError):
            record_trajectory(
                tool_name="read_file_tool",
                tool_parameters={
                    "filepath": filepath,
                    "encoding": encoding
                },
                step_data={
                    "filepath": filepath,
                    "display_title": "File Read Error",
                    "error_message": str(e)
                },
                is_error=True,
                error_message=str(e),
                error_type=type(e).__name__
            )
            
        raise
