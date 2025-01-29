"""
Module for running interactive subprocesses with output capture.
"""

import os
import re
import shlex
import shutil
import tempfile
from typing import List, Tuple

# Add macOS detection
IS_MACOS = os.uname().sysname == "Darwin"


def run_interactive_command(cmd: List[str]) -> Tuple[bytes, int]:
    """
    Runs an interactive command with a pseudo-tty, capturing combined output.

    Assumptions and constraints:
    - We are on a Linux system with script available
    - `cmd` is a non-empty list where cmd[0] is the executable
    - The executable and script are assumed to be on PATH
    - If anything is amiss (e.g., command not found), we fail early and cleanly

    The output is cleaned to remove ANSI escape sequences and control characters.

    Returns:
        Tuple of (cleaned_output, return_code)
    """
    # Fail early if cmd is empty
    if not cmd:
        raise ValueError("No command provided.")

    # Check that the command exists
    if shutil.which(cmd[0]) is None:
        raise FileNotFoundError(f"Command '{cmd[0]}' not found in PATH.")

    # Create temp files (we'll always clean them up)
    output_file = tempfile.NamedTemporaryFile(prefix="output_", delete=False)
    retcode_file = tempfile.NamedTemporaryFile(prefix="retcode_", delete=False)
    output_path = output_file.name
    retcode_path = retcode_file.name
    output_file.close()
    retcode_file.close()

    # Quote arguments for safety
    quoted_cmd = " ".join(shlex.quote(c) for c in cmd)
    # Use script to capture output with TTY and save return code
    shell_cmd = f"{quoted_cmd}; echo $? > {shlex.quote(retcode_path)}"

    def cleanup():
        for path in [output_path, retcode_path]:
            if os.path.exists(path):
                os.remove(path)

    try:
        # Disable pagers by setting environment variables
        os.environ["GIT_PAGER"] = ""
        os.environ["PAGER"] = ""

        # Run command with script for TTY and output capture
        if IS_MACOS:
            os.system(f"script -q {shlex.quote(output_path)} {shell_cmd}")
        else:
            os.system(
                f"script -q -c {shlex.quote(shell_cmd)} {shlex.quote(output_path)}"
            )

        # Read and clean the output
        with open(output_path, "rb") as f:
            output = f.read()

        # Clean ANSI escape sequences and control characters
        output = re.sub(rb"\x1b\[[0-9;]*[a-zA-Z]", b"", output)  # ANSI escape sequences
        output = re.sub(rb"[\x00-\x08\x0b\x0c\x0e-\x1f]", b"", output)  # Control chars

        # Get the return code
        with open(retcode_path, "r") as f:
            return_code = int(f.read().strip())

    except Exception as e:
        # If something goes wrong, cleanup and re-raise
        cleanup()
        raise RuntimeError("Error running interactive capture") from e
    finally:
        # Ensure files are removed no matter what
        cleanup()

    return output, return_code
