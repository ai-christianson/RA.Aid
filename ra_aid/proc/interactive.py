#!/usr/bin/env python3
"""
Module for running interactive subprocesses with output capture.

It uses a pseudo-tty and integrates pyte's HistoryScreen to
simulate a terminal and capture the final scrollback history (non-blank lines).
The interface remains compatible with external callers expecting a tuple (output, return_code),
where output is a bytes object (UTF-8 encoded).
"""

import os
import shlex
import shutil
import errno
import subprocess
from typing import List, Tuple

import pyte
from pyte.screens import HistoryScreen

def render_line(line, columns: int) -> str:
    """Render a single screen line from the pyte buffer (a mapping of column to Char)."""
    return "".join(line[x].data for x in range(columns))

def run_interactive_command(cmd: List[str]) -> Tuple[bytes, int]:
    """
    Runs an interactive command with a pseudo-tty, capturing final scrollback history.
    
    Assumptions and constraints:
      - Running on a Linux system.
      - `cmd` is a non-empty list where cmd[0] is the executable.
      - The executable is on PATH.
      
    Returns:
      A tuple of (captured_output, return_code), where captured_output is a UTF-8 encoded
      bytes object containing the trimmed non-empty history lines from the terminal session.
    
    Raises:
      ValueError: If no command is provided.
      FileNotFoundError: If the command is not found in PATH.
      RuntimeError: If an error occurs during execution.
    """
    # Fail early if cmd is empty.
    if not cmd:
        raise ValueError("No command provided.")
    
    # Check that the command exists.
    if shutil.which(cmd[0]) is None:
        raise FileNotFoundError(f"Command '{cmd[0]}' not found in PATH.")
    
    # Determine terminal dimensions; use os.get_terminal_size if available.
    try:
        term_size = os.get_terminal_size()
        cols, rows = term_size.columns, term_size.lines
    except OSError:
        cols, rows = 80, 24

    # Instantiate HistoryScreen with a large history (scrollback) buffer.
    screen = HistoryScreen(cols, rows, history=1000, ratio=0.5)
    stream = pyte.Stream(screen)

    # Open a new pseudo-tty.
    master_fd, slave_fd = os.openpty()
    
    # Spawn the subprocess with its stdio attached to the slave end.
    proc = subprocess.Popen(
        cmd,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True
    )
    os.close(slave_fd)  # Close slave in the parent.
    
    # Read output from the master file descriptor in real time.
    try:
        while True:
            try:
                data = os.read(master_fd, 1024)
            except OSError as e:
                if e.errno == errno.EIO:
                    # Expected error when the slave side is closed.
                    break
                else:
                    raise
            if not data:
                break
            # Feed the decoded data into pyte to update the screen and history.
            stream.feed(data.decode("utf-8", errors="ignore"))
            # Also write the raw data to stdout for live output.
            os.write(1, data)
    except KeyboardInterrupt:
        proc.terminate()
    finally:
        os.close(master_fd)
    proc.wait()
    
    # Assemble full scrollback: combine history.top, the current display, and history.bottom.
    top_lines = [render_line(line, cols) for line in screen.history.top]
    bottom_lines = [render_line(line, cols) for line in screen.history.bottom]
    display_lines = screen.display  # List of strings representing the current screen.
    all_lines = top_lines + display_lines + bottom_lines

    # Trim out empty lines to get only meaningful "history" lines.
    trimmed_lines = [line for line in all_lines if line.strip()]
    final_output = "\n".join(trimmed_lines)
    
    # Return as bytes for compatibility.
    return final_output.encode("utf-8"), proc.returncode

# if __name__ == "__main__":
#     # Test command: output 100 lines so that history goes beyond the screen height.
#     test_cmd = [
#         "bash",
#         "-c",
#         "for i in $(seq 1 100); do echo \"Line $i\"; sleep 0.05; done"
#     ]
#     output, ret = run_interactive_command(test_cmd)
#     print("\n=== Captured Scrollback (trimmed history lines) ===")
#     print(output.decode("utf-8"))
#     print("Return code:", ret)
