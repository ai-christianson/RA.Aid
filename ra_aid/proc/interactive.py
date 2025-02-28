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
import shutil
import signal
import subprocess
import sys
import time
from typing import List, Optional, Tuple

import pyte
from pyte.screens import HistoryScreen

# Platform-specific imports
if sys.platform == "win32":
    import msvcrt
    import threading
else:
    import select
    import termios
    import tty


def create_process(
    cmd: List[str],
    env: Optional[dict] = None,
    cols: Optional[int] = None,
    rows: Optional[int] = None,
) -> Tuple[subprocess.Popen, Optional[int]]:
    """
    Create a subprocess with appropriate settings for the current platform.

    Args:
        cmd: Command to execute as a list of strings
        env: Environment variables dictionary, defaults to os.environ.copy()
        cols: Number of columns for the terminal, defaults to current terminal width
        rows: Number of rows for the terminal, defaults to current terminal height

    Returns:
        On Unix: (process, master_fd) where master_fd is the file descriptor for the pty master
        On Windows: (process, None) as Windows doesn't use ptys
    """
    # Set default values if not provided
    if env is None:
        env = os.environ.copy()
    if cols is None or rows is None:
        default_cols, default_rows = get_terminal_size()
        if cols is None:
            cols = default_cols
        if rows is None:
            rows = default_rows
    if sys.platform == "win32":
        # Windows-specific process creation
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            env=env,
            startupinfo=startupinfo,
            universal_newlines=False,
        )
        return proc, None
    else:
        # Unix-specific process creation with pty
        master_fd, slave_fd = os.openpty()
        # Set master_fd to non-blocking to avoid indefinite blocking
        os.set_blocking(master_fd, False)

        proc = subprocess.Popen(
            cmd,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            bufsize=0,
            close_fds=True,
            env=env,
            preexec_fn=os.setsid,  # Create new process group for proper signal handling
        )
        os.close(slave_fd)  # Close slave end in the parent process

        return proc, master_fd


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

    # Create process based on platform
    proc, master_fd = create_process(cmd, env, cols, rows)

    captured_data = []
    start_time = time.time()
    was_terminated = False

    def check_timeout():
        elapsed = time.time() - start_time
        if elapsed > 3 * expected_runtime_seconds:
            if sys.platform == "win32":
                proc.kill()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return True
        elif elapsed > 2 * expected_runtime_seconds:
            if sys.platform == "win32":
                proc.terminate()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            return True
        return False

    if sys.platform == "win32":
        # Windows implementation using threads for I/O
        running = True
        stdin_thread = None

        def read_stdout():
            nonlocal running
            while running and proc.poll() is None:
                try:
                    data = proc.stdout.read(1024)
                    if not data:
                        break
                    captured_data.append(data)
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
                except (OSError, IOError):
                    break
                except Exception as e:
                    print(f"Error reading stdout: {e}", file=sys.stderr)
                    break

        def read_stderr():
            nonlocal running
            while running and proc.poll() is None:
                try:
                    data = proc.stderr.read(1024)
                    if not data:
                        break
                    captured_data.append(data)
                    sys.stderr.buffer.write(data)
                    sys.stderr.buffer.flush()
                except (OSError, IOError):
                    break
                except Exception as e:
                    print(f"Error reading stderr: {e}", file=sys.stderr)
                    break

        def handle_input():
            nonlocal running
            try:
                while running and proc.poll() is None:
                    if msvcrt.kbhit():
                        char = msvcrt.getch()
                        proc.stdin.write(char)
                        proc.stdin.flush()
                    time.sleep(0.01)  # Small sleep to prevent CPU hogging
            except (OSError, IOError):
                pass
            except Exception as e:
                print(f"Error handling input: {e}", file=sys.stderr)

        # Start I/O threads
        stdout_thread = threading.Thread(target=read_stdout)
        stderr_thread = threading.Thread(target=read_stderr)
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()

        # Only start stdin thread if we're in an interactive terminal
        if sys.stdin.isatty():
            stdin_thread = threading.Thread(target=handle_input)
            stdin_thread.daemon = True
            stdin_thread.start()

        try:
            # Main thread monitors timeout
            while proc.poll() is None:
                if check_timeout():
                    was_terminated = True
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            proc.terminate()
        finally:
            running = False
            # Wait for threads to finish
            stdout_thread.join(1.0)
            stderr_thread.join(1.0)
            if stdin_thread:
                stdin_thread.join(1.0)

            # Close pipes
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()
            if proc.stdin:
                proc.stdin.close()
    else:
        # Unix implementation using select and pty
        try:
            stdin_fd = sys.stdin.fileno()
        except (AttributeError, io.UnsupportedOperation):
            stdin_fd = None

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
                    os.write(1, data)
            except KeyboardInterrupt:
                proc.terminate()

        os.close(master_fd)

    # Wait for the process to finish
    proc.wait()

    # Ensure we have captured data even if the screen processing failed
    raw_output = b"".join(captured_data)

    # Process the captured output through a fresh screen
    try:
        # Create a new screen and stream for final processing
        screen = HistoryScreen(cols, rows, history=2000, ratio=0.5)
        stream = pyte.Stream(screen)

        # Feed all captured data at once to get the final state
        raw_output = b"".join(captured_data)
        decoded = raw_output.decode("utf-8", errors="ignore")
        stream.feed(decoded)

        # Get all history lines (top and bottom) and current display
        all_lines = []

        # Add history.top lines (older history)
        if hasattr(screen.history.top, "keys"):
            # Dictionary-like object
            for line_num in sorted(screen.history.top.keys()):
                line = screen.history.top[line_num]
                all_lines.append(render_line(line, cols))
        else:
            # Deque or other iterable
            for i, line in enumerate(screen.history.top):
                all_lines.append(render_line(line, cols))

        # Add current display lines
        all_lines.extend([render_line(line, cols) for line in screen.display])

        # Add history.bottom lines (newer history)
        if hasattr(screen.history.bottom, "keys"):
            # Dictionary-like object
            for line_num in sorted(screen.history.bottom.keys()):
                line = screen.history.bottom[line_num]
                all_lines.append(render_line(line, cols))
        else:
            # Deque or other iterable
            for i, line in enumerate(screen.history.bottom):
                all_lines.append(render_line(line, cols))

        # Trim out empty lines to get only meaningful lines
        # Also strip trailing whitespace from each line
        trimmed_lines = [line.rstrip() for line in all_lines if line and line.strip()]

        final_output = "\n".join(trimmed_lines)
    except Exception as e:
        # If anything goes wrong with screen processing, fall back to raw output
        print(f"Warning: Error processing terminal output: {e}", file=sys.stderr)
        try:
            # Decode raw output, strip trailing whitespace from each line
            decoded = raw_output.decode("utf-8", errors="replace")
            lines = [line.rstrip() for line in decoded.splitlines()]
            final_output = "\n".join(lines)
        except Exception:
            # Ultimate fallback if line processing fails
            final_output = raw_output.decode("utf-8", errors="replace").strip()

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
