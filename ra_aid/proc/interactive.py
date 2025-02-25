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
import threading
import time
from typing import List, Tuple, Optional, Any

import pyte
from pyte.screens import HistoryScreen

# Import platform-specific modules
if sys.platform == "win32":
    import msvcrt
else:
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


def create_process(cmd: List[str]) -> Tuple[subprocess.Popen, Any]:
    """
    Create a subprocess with appropriate platform-specific settings.
    
    This function handles the platform-specific differences between Windows and Unix:
    
    On Windows:
    - Creates a process with pipes for stdin/stdout
    - Uses STARTF_USESHOWWINDOW to prevent console windows from appearing
    - Returns the process and None (no PTY on Windows)
    
    On Unix:
    - Creates a pseudo-terminal (PTY) for full terminal emulation
    - Sets up non-blocking I/O on the master file descriptor
    - Configures environment variables for consistent behavior
    - Creates a new process group for proper signal handling
    
    Args:
        cmd: A list containing the command and its arguments.
        
    Returns:
        A tuple of (process, master_fd) where:
        - process is the subprocess.Popen object
        - master_fd is the master file descriptor (on Unix) or None (on Windows)
    """
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
            startupinfo=startupinfo
        )
        return proc, None
    else:
        # Unix-specific process creation with pty
        master_fd, slave_fd = os.openpty()
        os.set_blocking(master_fd, False)
        
        env = os.environ.copy()
        env.update({
            "DEBIAN_FRONTEND": "noninteractive",
            "GIT_PAGER": "",
            "PYTHONUNBUFFERED": "1",
            "CI": "true",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "COLUMNS": str(get_terminal_size()[0]),
            "LINES": str(get_terminal_size()[1]),
            "FORCE_COLOR": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "NODE_OPTIONS": "--unhandled-rejections=strict",
        })
        
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
        
        return proc, master_fd


def render_line(line, columns: int) -> str:
    """
    Render a single screen line from the pyte buffer.
    
    This function handles different types of line representations from pyte:
    - String lines (from screen.display)
    - Dictionary-style lines (from history, mapping column indices to Char objects)
    
    Args:
        line: A line from pyte's screen buffer (string or dict mapping column to Char)
        columns: Maximum number of columns to render
        
    Returns:
        A string representation of the line with proper character data
    """
    if not line:
        return ""
    
    # Handle string lines directly (from screen.display)
    if isinstance(line, str):
        return line
        
    # Handle dictionary-style lines (from history)
    try:
        max_col = max(line.keys()) if line else -1
        result = ""
        for x in range(min(columns, max_col + 1)):
            if x in line:
                result += line[x].data
        return result
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
    # Increase history size to capture more lines (from 2000 to 5000)
    screen = HistoryScreen(cols, rows, history=5000, ratio=0.8)
    stream = pyte.Stream(screen)

    # Create process with platform-specific settings
    proc, master_fd = create_process(cmd)

    captured_data = []
    start_time = time.time()
    was_terminated = False

    def check_timeout():
        """
        Check if the process has exceeded its timeout limits and terminate if necessary.
        
        Returns:
            True if the process was terminated due to timeout, False otherwise.
        """
        elapsed = time.time() - start_time
        if sys.platform == "win32":
            # Windows process termination
            if elapsed > 3 * expected_runtime_seconds:
                # Hard kill after 3x the expected time
                proc.kill()
                return True
            elif elapsed > 2 * expected_runtime_seconds:
                # Graceful termination after 2x the expected time
                proc.terminate()
                return True
        else:
            # Unix process termination (using process groups)
            if elapsed > 3 * expected_runtime_seconds:
                # Hard kill with SIGKILL after 3x the expected time
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                return True
            elif elapsed > 2 * expected_runtime_seconds:
                # Graceful termination with SIGTERM after 2x the expected time
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                return True
        return False

    # Windows implementation
    if sys.platform == "win32":
        try:
            stdin_fd = None
            if sys.stdin and sys.stdin.isatty():
                try:
                    stdin_fd = sys.stdin.fileno()
                except (AttributeError, io.UnsupportedOperation):
                    stdin_fd = None

            # Function to read output from the process
            def read_output():
                """
                Thread function to continuously read and process output from the subprocess.
                
                This function:
                1. Reads data from the process stdout in chunks
                2. Adds the data to captured_data for later processing
                3. Feeds the data to the terminal emulator (pyte)
                4. Writes the data to stdout for real-time display
                5. Handles process termination and cleanup
                """
                while proc.poll() is None:
                    try:
                        data = proc.stdout.read(1024)
                        if not data:
                            break
                        captured_data.append(data)
                        decoded = data.decode("utf-8", errors="ignore")
                        stream.feed(decoded)
                        # Write to stdout for real-time display
                        try:
                            sys.stdout.buffer.write(data)
                            sys.stdout.buffer.flush()
                        except (OSError, IOError):
                            pass  # Ignore errors writing to stdout
                    except (OSError, IOError):
                        break
                    except Exception as e:
                        print(f"Error reading output: {e}", file=sys.stderr)
                        break
                
                # Try to read any remaining data after process ends
                try:
                    remaining = proc.stdout.read()
                    if remaining:
                        captured_data.append(remaining)
                        stream.feed(remaining.decode("utf-8", errors="ignore"))
                except Exception:
                    pass

            # Start a thread to read output
            output_thread = threading.Thread(target=read_output)
            output_thread.daemon = True
            output_thread.start()

            # Main loop for input and timeout checking
            while proc.poll() is None:
                if check_timeout():
                    was_terminated = True
                    break

                # Check for input if we have a TTY
                if stdin_fd is not None and msvcrt.kbhit():
                    try:
                        char = msvcrt.getch()
                        proc.stdin.write(char)
                        proc.stdin.flush()
                    except (OSError, IOError):
                        break

                time.sleep(0.1)  # Small sleep to prevent CPU hogging

            # Wait for the output thread to finish
            output_thread.join(timeout=1.0)

        except KeyboardInterrupt:
            proc.terminate()
        finally:
            if proc.stdin:
                proc.stdin.close()
            if proc.stdout:
                proc.stdout.close()

    # Unix implementation
    else:
        try:
            stdin_fd = None
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
                        try:
                            os.write(1, data)
                        except (OSError, IOError):
                            pass  # Ignore errors writing to stdout
                except KeyboardInterrupt:
                    proc.terminate()

            os.close(master_fd)
        except Exception as e:
            print(f"Error in Unix implementation: {e}", file=sys.stderr)

    # Wait for the process to finish
    proc.wait()

    # Ensure we have captured data even if the screen processing failed
    raw_output = b"".join(captured_data)
    
    # Try to assemble full scrollback from the terminal emulation
    try:
        # Assemble full scrollback: combine history.top, the current display, and history.bottom.
        top_lines = []
        display_lines = []
        bottom_lines = []
        
        # Safely extract history.top (scrollback buffer above visible area)
        if hasattr(screen, 'history') and hasattr(screen.history, 'top'):
            top_lines = [render_line(line, cols) for line in screen.history.top]
        
        # Safely extract current display (visible terminal area)
        if hasattr(screen, 'display'):
            display_lines = [render_line(line, cols) for line in screen.display]
        
        # Safely extract history.bottom (scrollback buffer below visible area)
        if hasattr(screen, 'history') and hasattr(screen.history, 'bottom'):
            bottom_lines = [render_line(line, cols) for line in screen.history.bottom]
        
        # Combine all lines to get the complete terminal history
        all_lines = top_lines + display_lines + bottom_lines
        
        # Trim out empty lines to get only meaningful "history" lines
        # This is important for commands that don't fill the entire terminal
        trimmed_lines = [line for line in all_lines if line and line.strip()]
        
        # IMPORTANT: Always check if we have meaningful content from the screen
        if trimmed_lines and any(line.strip() for line in trimmed_lines):
            final_output = "\n".join(trimmed_lines)
        else:
            # Fall back to raw output if no meaningful lines from screen
            # This is critical for simple commands like "echo hello world"
            raw_decoded = raw_output.decode('utf-8', errors='replace')
            final_output = raw_decoded.strip()
            
            # If raw output is also empty, try to extract any content from the screen
            if not final_output and display_lines:
                final_output = "\n".join(display_lines)
    except Exception as e:
        # If anything goes wrong with screen processing, fall back to raw output
        print(f"Warning: Error processing terminal output: {e}", file=sys.stderr)
        final_output = raw_output.decode('utf-8', errors='replace').strip()

    # Add timeout message if process was terminated due to timeout.
    if was_terminated:
        timeout_msg = f"\n[Process exceeded timeout ({expected_runtime_seconds} seconds expected)]"
        final_output += timeout_msg

    # Limit output to the last 8000 bytes, but try to keep complete lines
    # This ensures we don't exceed memory limits while preserving readable output
    if len(final_output) > 8000:
        # Find a newline near the 8000-byte cutoff point
        cutoff = max(0, len(final_output) - 8000)
        # Try to find a newline after the cutoff to avoid cutting in the middle of a line
        newline_pos = final_output.find('\n', cutoff)
        if newline_pos != -1 and newline_pos < cutoff + 200:  # Don't look too far ahead
            cutoff = newline_pos + 1
        final_output = final_output[cutoff:]
    
    # Ensure we're returning bytes with consistent encoding
    if isinstance(final_output, str):
        # Make sure we have content in the final output
        if not final_output.strip() and raw_output:
            # Fall back to raw output if processed output is empty
            final_output = raw_output.decode('utf-8', errors='replace').strip()
        final_output = final_output.encode("utf-8")
    elif not isinstance(final_output, bytes):
        # Handle any unexpected type by converting to string and then bytes
        final_output = str(final_output).encode("utf-8")
        
    # Ensure we have at least some output, even if the command produced none
    # This is important for error reporting and debugging
    if not final_output or final_output.strip() == b"":
        # Last resort: use raw output directly
        if raw_output:
            final_output = raw_output
        else:
            final_output = b"[No output captured]"
        
    return final_output, proc.returncode


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: interactive.py <command> [args...]")
        sys.exit(1)
    output, return_code = run_interactive_command(sys.argv[1:])
    sys.exit(return_code)
