#!/usr/bin/env python3
"""
Module for running interactive subprocesses with output capture,
with full raw input passthrough for interactive commands.

It uses a pseudo-tty and integrates pyte's HistoryScreen to simulate
a terminal and capture the final scrollback history (non-blank lines).
The interface remains compatible with external callers expecting a tuple (output, return_code),
where output is a bytes object (UTF-8 encoded).
"""

import os
import shlex
import shutil
import errno
import sys
import io
import subprocess
import select
import termios
import tty
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
    if not cmd:
        raise ValueError("No command provided.")
    if shutil.which(cmd[0]) is None:
        raise FileNotFoundError(f"Command '{cmd[0]}' not found in PATH.")
    
    try:
        term_size = os.get_terminal_size()
        cols, rows = term_size.columns, term_size.lines
    except OSError:
        cols, rows = 80, 24

    # Set up pyte screen and stream to capture terminal output.
    screen = HistoryScreen(cols, rows, history=2000, ratio=0.5)
    stream = pyte.Stream(screen)

    # Open a new pseudo-tty.
    master_fd, slave_fd = os.openpty()

    try:
        stdin_fd = sys.stdin.fileno()
    except (AttributeError, io.UnsupportedOperation):
        stdin_fd = None

    # Set up environment variables for the subprocess using detected terminal size
    env = os.environ.copy()
    env.update({
        'DEBIAN_FRONTEND': 'noninteractive',
        'GIT_PAGER': '',
        'PYTHONUNBUFFERED': '1',
        'CI': 'true',
        'LANG': 'C.UTF-8',
        'LC_ALL': 'C.UTF-8',
        'COLUMNS': str(cols),
        'LINES': str(rows),
        'FORCE_COLOR': '1',
        'GIT_TERMINAL_PROMPT': '0',
        'PYTHONDONTWRITEBYTECODE': '1',
        'NODE_OPTIONS': '--unhandled-rejections=strict'
    })

    proc = subprocess.Popen(
        cmd,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        bufsize=0,
        close_fds=True,
        env=env
    )
    os.close(slave_fd)  # Close slave end in the parent process.

    captured_data = []

    # If we're in an interactive TTY, set raw mode and forward input.
    if stdin_fd is not None and sys.stdin.isatty():
        old_settings = termios.tcgetattr(stdin_fd)
        tty.setraw(stdin_fd)
        try:
            while True:
                rlist, _, _ = select.select([master_fd, stdin_fd], [], [])
                if master_fd in rlist:
                    try:
                        data = os.read(master_fd, 1024)
                    except OSError as e:
                        if e.errno == errno.EIO:
                            break
                        else:
                            raise
                    if not data:
                        break
                    captured_data.append(data)
                    # Update pyte's screen state.
                    stream.feed(data.decode("utf-8", errors="ignore"))
                    # Write to stdout for live output.
                    os.write(1, data)
                if stdin_fd in rlist:
                    try:
                        input_data = os.read(stdin_fd, 1024)
                    except OSError:
                        input_data = b""
                    if input_data:
                        # Forward raw keystrokes directly to the subprocess.
                        os.write(master_fd, input_data)
        except KeyboardInterrupt:
            proc.terminate()
        finally:
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_settings)
    else:
        # Non-interactive mode (e.g., during unit tests).
        try:
            while True:
                try:
                    data = os.read(master_fd, 1024)
                except OSError as e:
                    if e.errno == errno.EIO:
                        break
                    else:
                        raise
                if not data:
                    break
                captured_data.append(data)
                stream.feed(data.decode("utf-8", errors="ignore"))
                os.write(1, data)
        except KeyboardInterrupt:
            proc.terminate()

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
    
    return final_output.encode("utf-8"), proc.returncode

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: interactive.py <command> [args...]")
        sys.exit(1)
    output, return_code = run_interactive_command(sys.argv[1:])
    sys.exit(return_code)
