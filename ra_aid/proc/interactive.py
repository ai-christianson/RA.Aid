#!/usr/bin/env python3
"""
Module for running interactive subprocesses with output capture,
with full raw input passthrough for interactive commands.

It uses a pseudo-tty on Unix systems and direct pipes on Windows to simulate
a terminal and capture the final scrollback history (non-blank lines).
"""

import errno
import io
import os
import select
import signal
import subprocess
import sys
import time
import shutil
from typing import List, Tuple, Optional

import pyte

# Windows-specific imports
if sys.platform == "win32":
    try:
        # msvcrt: Provides Windows console I/O functionality
        import msvcrt
        # win32pipe, win32file: For low-level pipe operations
        import win32pipe
        import win32file
        # win32con: Windows API constants
        import win32con
        # win32process: Process management on Windows
        import win32process
    except ImportError as e:
        print("Error: Required Windows dependencies not found.")
        print("Please install the required packages using:")
        print("  pip install pywin32")
        sys.exit(1)
else:
    # Unix-specific imports for terminal handling
    import termios
    import fcntl
    import pty

def get_terminal_size():
    """Get the current terminal size."""
    if sys.platform == "win32":
        import shutil
        size = shutil.get_terminal_size()
        return size.columns, size.lines
    else:
        import struct
        try:
            with open(sys.stdout.fileno(), 'wb', buffering=0) as fd:
                size = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))
                return size[1], size[0]
        except (IOError, AttributeError):
            return 80, 24

def create_process(cmd: List[str]) -> Tuple[subprocess.Popen, Optional[int]]:
    """Create a subprocess with appropriate handling for the platform.
    
    On Windows:
    - Uses STARTUPINFO to hide the console window
    - Creates a new process group for proper signal handling
    - Returns direct pipe handles for I/O
    
    On Unix:
    - Creates a pseudo-terminal (PTY) for proper terminal emulation
    - Sets up process group for signal handling
    - Returns master PTY file descriptor for I/O
    """
    if sys.platform == "win32":
        # Windows process creation with hidden console
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # Hide the console window
        
        # Create process with proper pipe handling
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,   # Allow writing to stdin
            stdout=subprocess.PIPE,  # Capture stdout
            stderr=subprocess.PIPE,  # Capture stderr
            startupinfo=startupinfo,
            # CREATE_NEW_PROCESS_GROUP allows proper Ctrl+C handling
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
        return proc, None  # No PTY master_fd needed on Windows
    else:
        # Unix process creation with PTY
        master_fd, slave_fd = pty.openpty()
        proc = subprocess.Popen(
            cmd,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            preexec_fn=os.setsid
        )
        os.close(slave_fd)
        return proc, master_fd

def run_interactive_command(
    cmd: List[str], 
    expected_runtime_seconds: int = 1800, 
    ratio: float = 0.5
) -> Tuple[bytes, int]:
    """
    Runs an interactive command with a pseudo-tty, capturing final scrollback history.

    Assumptions and constraints:
      - Running on a Linux system or Windows.
      - `cmd` is a non-empty list where cmd[0] is the executable.
      - The executable is on PATH.

    Args:
      cmd: A list containing the command and its arguments.
      expected_runtime_seconds: Expected runtime in seconds, defaults to 1800.
        If process exceeds 2x this value, it will be terminated gracefully.
        If process exceeds 3x this value, it will be killed forcefully.
        Must be between 1 and 1800 seconds (30 minutes).
      ratio: Ratio of history to keep from top vs bottom (default: 0.5)

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

    try:
        term_size = get_terminal_size()
        cols, rows = term_size.columns, term_size.lines
    except OSError:
        cols, rows = 80, 24

    # Set up pyte screen and stream to capture terminal output.
    screen = pyte.HistoryScreen(cols, rows, history=2000, ratio=ratio)
    stream = pyte.Stream(screen)

    proc, master_fd = create_process(cmd)

    captured_data = []
    start_time = time.time()
    was_terminated = False
    timeout_type = None

    def check_timeout():
        nonlocal timeout_type
        elapsed = time.time() - start_time
        if elapsed > 3 * expected_runtime_seconds:
            if sys.platform == "win32":
                print("\nProcess exceeded hard timeout limit, forcefully terminating...")
                proc.terminate()
                time.sleep(0.5)
                if proc.poll() is None:
                    print("Process did not respond to termination, killing...")
                    proc.kill()
            else:
                print("\nProcess exceeded hard timeout limit, sending SIGKILL...")
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            timeout_type = "hard_timeout"
            return True
        elif elapsed > 2 * expected_runtime_seconds:
            if sys.platform == "win32":
                print("\nProcess exceeded soft timeout limit, attempting graceful termination...")
                proc.terminate()
            else:
                print("\nProcess exceeded soft timeout limit, sending SIGTERM...")
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            timeout_type = "soft_timeout"
            return True
        return False

    # Interactive mode: forward input if running in a TTY.
    if sys.platform == "win32":
        # Windows handling
        try:
            while True:
                if check_timeout():
                    was_terminated = True
                    break
                
                try:
                    # Check stdout with proper error handling
                    stdout_data = proc.stdout.read1(1024)
                    if stdout_data:
                        captured_data.append(stdout_data)
                        try:
                            stream.feed(stdout_data.decode(errors='ignore'))
                        except Exception as e:
                            print(f"Warning: Error processing stdout: {e}")

                    # Check stderr with proper error handling
                    stderr_data = proc.stderr.read1(1024)
                    if stderr_data:
                        captured_data.append(stderr_data)
                        try:
                            stream.feed(stderr_data.decode(errors='ignore'))
                        except Exception as e:
                            print(f"Warning: Error processing stderr: {e}")

                    # Check for input with proper error handling
                    if msvcrt.kbhit():
                        try:
                            char = msvcrt.getch()
                            proc.stdin.write(char)
                            proc.stdin.flush()
                        except (IOError, OSError) as e:
                            print(f"Warning: Error handling keyboard input: {e}")
                            break

                except (IOError, OSError) as e:
                    if isinstance(e, OSError) and e.winerror == 6:  # Invalid handle
                        break
                    print(f"Warning: I/O error during process communication: {e}")
                    break

        except Exception as e:
            print(f"Error in Windows process handling: {e}")
            proc.terminate()
    else:
        # Unix handling
        import tty
        try:
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin)
            while True:
                if check_timeout():
                    was_terminated = True
                    break
                rlist, _, _ = select.select([master_fd, sys.stdin], [], [], 0.1)
                
                for fd in rlist:
                    try:
                        if fd == master_fd:
                            data = os.read(master_fd, 1024)
                            if not data:
                                break
                            captured_data.append(data)
                            stream.feed(data.decode(errors='ignore'))
                        else:
                            data = os.read(fd, 1024)
                            os.write(master_fd, data)
                    except (IOError, OSError):
                        break

        except KeyboardInterrupt:
            proc.terminate()
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    if not sys.platform == "win32" and master_fd is not None:
        os.close(master_fd)

    proc.wait()

    # Assemble full scrollback: combine history.top, the current display, and history.bottom.
    def render_line(line, width):
        return ''.join(char.data for char in line[:width]).rstrip()

    # Combine history and current screen content
    final_output = []
    
    # Add lines from history
    history_lines = [render_line(line, cols) for line in screen.history.top]
    final_output.extend(line for line in history_lines if line.strip())
    
    # Add current screen content
    screen_lines = [render_line(line, cols) for line in screen.display]
    final_output.extend(line for line in screen_lines if line.strip())
    
    # Add bottom history
    bottom_lines = [render_line(line, cols) for line in screen.history.bottom]
    final_output.extend(line for line in bottom_lines if line.strip())

    # Add timeout message if process was terminated
    if was_terminated:
        if timeout_type == "hard_timeout":
            timeout_msg = f"\n[Process forcefully terminated after exceeding {3 * expected_runtime_seconds:.1f} seconds (expected: {expected_runtime_seconds} seconds)]"
        else:
            timeout_msg = f"\n[Process gracefully terminated after exceeding {2 * expected_runtime_seconds:.1f} seconds (expected: {expected_runtime_seconds} seconds)]"
        final_output.append(timeout_msg)

    # Limit output size
    final_output = final_output[-8000:]
    final_output = '\n'.join(final_output)

    return final_output.encode('utf-8'), proc.returncode


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: interactive.py <command> [args...]")
        sys.exit(1)
    output, return_code = run_interactive_command(sys.argv[1:])
    sys.exit(return_code)
