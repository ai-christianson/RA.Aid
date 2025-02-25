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
    import tty

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
    
    Args:
        cmd: A list containing the command and its arguments.
        expected_runtime_seconds: Expected runtime in seconds, defaults to 1800.
        ratio: Ratio of history to keep from top vs bottom (default: 0.5)
        
    Returns:
        A tuple of (captured_output, return_code)
    """
    if not cmd:
        raise ValueError("No command provided")
    if not 0 < expected_runtime_seconds <= 1800:
        raise ValueError("Expected runtime must be between 1 and 1800 seconds")
        
    try:
        term_size = os.get_terminal_size()
        cols, rows = term_size.columns, term_size.lines
    except OSError:
        cols, rows = 80, 24
        
    screen = pyte.HistoryScreen(cols, rows, history=2000, ratio=ratio)
    stream = pyte.Stream(screen)

    # Set up environment variables for the subprocess
    env = os.environ.copy()
    env.update({
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
    })

    # Create process with proper PTY handling
    if sys.platform == "win32":
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            bufsize=0,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
        master_fd = None
    else:
        master_fd, slave_fd = os.openpty()
        os.set_blocking(master_fd, False)
        
        proc = subprocess.Popen(
            cmd,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            env=env,
            bufsize=0,
            close_fds=True,
            preexec_fn=os.setsid
        )
        os.close(slave_fd)

    try:
        stdin_fd = sys.stdin.fileno()
    except (AttributeError, io.UnsupportedOperation):
        stdin_fd = None
    
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
        # Windows handling
        try:
            while True:
                if check_timeout():
                    was_terminated = True
                    break
                    
                if proc.poll() is not None:
                    break
                    
                try:
                    output = proc.stdout.read1(1024)
                    if output:
                        captured_data.append(output)
                        stream.feed(output.decode('utf-8', errors='replace'))
                        os.write(1, output)  # Write to stdout
                        
                    if msvcrt.kbhit():
                        char = msvcrt.getch()
                        proc.stdin.write(char)
                        proc.stdin.flush()
                except (IOError, OSError) as e:
                    break
                    
        except KeyboardInterrupt:
            proc.terminate()
            
    else:
        # Unix handling with proper TTY passthrough
        if stdin_fd is not None and sys.stdin.isatty():
            old_settings = termios.tcgetattr(stdin_fd)
            tty.setraw(stdin_fd)
            try:
                while True:
                    if check_timeout():
                        was_terminated = True
                        break
                        
                    rlist, _, _ = select.select([master_fd, stdin_fd], [], [], 0.1)
                    
                    if master_fd in rlist:
                        try:
                            data = os.read(master_fd, 1024)
                        except OSError as e:
                            if e.errno == errno.EIO:
                                break
                            raise
                            
                        if not data:
                            break
                            
                        captured_data.append(data)
                        stream.feed(data.decode('utf-8', errors='replace'))
                        os.write(1, data)  # Write to stdout
                        
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
            # Non-interactive mode
            try:
                while True:
                    if check_timeout():
                        was_terminated = True
                        break
                        
                    rlist, _, _ = select.select([master_fd], [], [], 0.1)
                    if not rlist:
                        continue
                        
                    try:
                        data = os.read(master_fd, 1024)
                    except OSError as e:
                        if e.errno == errno.EIO:
                            break
                        raise
                        
                    if not data:
                        break
                        
                    captured_data.append(data)
                    stream.feed(data.decode('utf-8', errors='replace'))
                    os.write(1, data)  # Write to stdout
                    
            except KeyboardInterrupt:
                proc.terminate()
    
    # Cleanup
    if master_fd is not None:
        os.close(master_fd)
    
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    
    # Get the final screen content
    def render_line(line, width):
        return ''.join(char.data for char in line[:width]).rstrip()
    
    # Combine history and current screen content
    top_lines = [render_line(line, cols) for line in screen.history.top]
    bottom_lines = [render_line(line, cols) for line in screen.history.bottom]
    display_lines = screen.display
    all_lines = top_lines + display_lines + bottom_lines
    
    # Filter empty lines
    trimmed_lines = [line for line in all_lines if line.strip()]
    final_output = '\n'.join(trimmed_lines)
    
    # Add timeout message if process was terminated
    if was_terminated:
        timeout_msg = f"\n[Process exceeded timeout ({expected_runtime_seconds} seconds expected)]"
        final_output += timeout_msg
    
    # Limit output size
    final_output = final_output[-8000:]
    
    return final_output.encode('utf-8'), proc.returncode


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: interactive.py <command> [args...]")
        sys.exit(1)
    output, return_code = run_interactive_command(sys.argv[1:])
    sys.exit(return_code)
