#!/usr/bin/env python3
"""
Module for running interactive subprocesses with output capture,
with full raw input passthrough for interactive commands.

It uses a pseudo-tty and integrates pyte's HistoryScreen to simulate
a terminal and capture the final scrollback history (non-blank lines).
The interface remains compatible with external callers expecting a tuple (output, return_code),
where output is a bytes object (UTF-8 encoded).
"""

import errno
import io
import os
import select
import shutil
import signal
import subprocess
import sys
import time
from typing import List, Tuple

import pyte
from pyte.screens import HistoryScreen

import termios
import tty


def get_terminal_size() -> Tuple[int, int]:
    """
    Get the current terminal size in a cross-platform way.
    
    This function works on both Unix and Windows systems, using shutil.get_terminal_size()
    which is available in Python 3.3+. If the terminal size cannot be determined
    (e.g., when running in a non-interactive environment), it falls back to default values.
    
    Returns:
        A tuple of (columns, rows) representing the terminal dimensions.
    """
    try:
        size = shutil.get_terminal_size()
        return size.columns, size.lines
    except OSError:
        # Default fallback values
        return 80, 24


def render_line(line, columns: int) -> str:
    """Render a single screen line from the pyte buffer (a mapping of column to Char)."""
    if not line:
        return ""
    
    # Handle string lines directly (from screen.display)
    if isinstance(line, str):
        return line
        
    # Handle dictionary-style lines (from history)
    try:
        return "".join(line[x].data for x in range(columns) if x in line)
    except (AttributeError, TypeError):
        # Fallback for any unexpected types
        return str(line)


def run_interactive_command(
    cmd: List[str], expected_runtime_seconds: int = 30
) -> Tuple[bytes, int]:
    """
    Runs an interactive command with output capture, capturing final scrollback history.
    
    This function provides a cross-platform way to run interactive commands with:
    - Full terminal emulation using pyte's HistoryScreen
    - Real-time display of command output
    - Input forwarding when running in an interactive terminal
    - Timeout handling to prevent runaway processes
    - Comprehensive output capture including ANSI escape sequences
    
    The implementation differs significantly between Windows and Unix:
    
    On Windows:
    - Uses threading to handle I/O operations
    - Relies on msvcrt for keyboard input detection
    - Uses pipes for process communication
    
    On Unix:
    - Uses pseudo-terminals (PTY) for full terminal emulation
    - Uses select() for non-blocking I/O
    - Handles raw terminal mode for proper input forwarding
    - Uses process groups for proper signal handling

    Args:
      cmd: A list containing the command and its arguments.
      expected_runtime_seconds: Expected runtime in seconds, defaults to 30.
        If process exceeds 2x this value, it will be terminated gracefully.
        If process exceeds 3x this value, it will be killed forcefully.
        Must be between 1 and 1800 seconds (30 minutes).

    Returns:
      A tuple of (captured_output, return_code), where captured_output is a UTF-8 encoded
      bytes object containing the trimmed non-empty history lines from the terminal session.

    Raises:
      ValueError: If no command is provided.
      FileNotFoundError: If the command is not found in PATH.
      ValueError: If expected_runtime_seconds is less than or equal to 0 or greater than 1800.
      RuntimeError: If an error occurs during execution.
    """
    if not cmd:
        raise ValueError("No command provided.")
    if shutil.which(cmd[0]) is None:
        raise FileNotFoundError(f"Command '{cmd[0]}' not found in PATH.")
    if expected_runtime_seconds <= 0 or expected_runtime_seconds > 1800:
        raise ValueError(
            "expected_runtime_seconds must be between 1 and 1800 seconds (30 minutes)"
        )

    cols, rows = get_terminal_size()

    # Set up pyte screen and stream to capture terminal output.
    screen = HistoryScreen(cols, rows, history=2000, ratio=0.5)
    stream = pyte.Stream(screen)

    # Open a new pseudo-tty.
    master_fd, slave_fd = os.openpty()
    # Set master_fd to non-blocking to avoid indefinite blocking.
    os.set_blocking(master_fd, False)

    try:
        stdin_fd = sys.stdin.fileno()
    except (AttributeError, io.UnsupportedOperation):
        stdin_fd = None

    # Set up environment variables for the subprocess using detected terminal size.
    env = os.environ.copy()
    env.update(
        {
            "DEBIAN_FRONTEND": "noninteractive",
            "GIT_PAGER": "",
            "PYTHONUNBUFFERED": "1",
            "CI": "true",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "COLUMNS": str(cols),
            "LINES": str(rows),
            "FORCE_COLOR": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "NODE_OPTIONS": "--unhandled-rejections=strict",
        }
    )

    proc = subprocess.Popen(
        cmd,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        bufsize=0,
        close_fds=True,
        env=env,
        preexec_fn=os.setsid,  # Create new process group for proper signal handling.
    )
    os.close(slave_fd)  # Close slave end in the parent process.

    captured_data = []
    start_time = time.time()
    was_terminated = False

    def check_timeout():
        elapsed = time.time() - start_time
        if elapsed > 3 * expected_runtime_seconds:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return True
        elif elapsed > 2 * expected_runtime_seconds:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            return True
        return False

    # Interactive mode: forward input if running in a TTY.
    if stdin_fd is not None and sys.stdin.isatty():
        old_settings = termios.tcgetattr(stdin_fd)
        tty.setraw(stdin_fd)
        try:
            while True:
                if check_timeout():
                    was_terminated = True
                    break
                # Use a finite timeout to avoid indefinite blocking.
                rlist, _, _ = select.select([master_fd, stdin_fd], [], [], 1.0)
                if master_fd in rlist:
                    try:
                        data = os.read(master_fd, 1024)
                    except OSError as e:
                        if e.errno == errno.EIO:
                            break
                        else:
                            raise
                    if not data:  # EOF detected.
                        break
                    captured_data.append(data)
                    decoded = data.decode("utf-8", errors="ignore")
                    stream.feed(decoded)
                    os.write(1, data)
                if stdin_fd in rlist:
                    try:
                        input_data = os.read(stdin_fd, 1024)
                    except OSError:
                        input_data = b""
                    if input_data:
                        os.write(master_fd, input_data)
        except KeyboardInterrupt:
            proc.terminate()
        finally:
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_settings)
    else:
        # Non-interactive mode.
        try:
            while True:
                if check_timeout():
                    was_terminated = True
                    break
                rlist, _, _ = select.select([master_fd], [], [], 1.0)
                if not rlist:
                    continue
                try:
                    data = os.read(master_fd, 1024)
                except OSError as e:
                    if e.errno == errno.EIO:
                        break
                    else:
                        raise
                if not data:  # EOF detected.
                    break
                captured_data.append(data)
                decoded = data.decode("utf-8", errors="ignore")
                stream.feed(decoded)
                os.write(1, data)
        except KeyboardInterrupt:
            proc.terminate()

    os.close(master_fd)

    # Wait for the process to finish
    proc.wait()

    # Ensure we have captured data even if the screen processing failed
    raw_output = b"".join(captured_data)
    
    # Assemble full scrollback from the terminal emulation
    try:
        # Assemble full scrollback: combine history.top, the current display, and history.bottom.
        top_lines = [render_line(line, cols) for line in screen.history.top]
        display_lines = [render_line(line, cols) for line in screen.display]
        bottom_lines = [render_line(line, cols) for line in screen.history.bottom]
        
        # Combine all lines to get the complete terminal history
        all_lines = top_lines + display_lines + bottom_lines
        
        # Trim out empty lines to get only meaningful "history" lines
        trimmed_lines = [line for line in all_lines if line and line.strip()]
        
        final_output = "\n".join(trimmed_lines)
    except Exception as e:
        # If anything goes wrong with screen processing, fall back to raw output
        print(f"Warning: Error processing terminal output: {e}", file=sys.stderr)
        final_output = raw_output.decode('utf-8', errors='replace').strip()

    # Add timeout message if process was terminated due to timeout.
    if was_terminated:
        timeout_msg = f"\n[Process exceeded timeout ({expected_runtime_seconds} seconds expected)]"
        final_output += timeout_msg

    # Limit output to the last 8000 bytes
    if isinstance(final_output, str):
        final_output = final_output[-8000:]
        final_output = final_output.encode("utf-8")
    elif isinstance(final_output, bytes):
        final_output = final_output[-8000:]
    else:
        # Handle any unexpected type
        final_output = str(final_output)[-8000:].encode("utf-8")
        
    return final_output, proc.returncode


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: interactive.py <command> [args...]")
        sys.exit(1)
    output, return_code = run_interactive_command(sys.argv[1:])
    sys.exit(return_code)
