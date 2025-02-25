"""Tests for Windows-specific functionality."""

import os
import sys
import subprocess
import pytest
from unittest.mock import patch, MagicMock

from ra_aid.proc.interactive import get_terminal_size, create_process, run_interactive_command

@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific tests")
class TestWindowsCompatibility:
    """Test suite for Windows-specific functionality."""

    def test_get_terminal_size(self):
        """Test terminal size detection on Windows."""
        with patch('shutil.get_terminal_size') as mock_get_size:
            mock_get_size.return_value = MagicMock(columns=120, lines=30)
            cols, rows = get_terminal_size()
            assert cols == 120
            assert rows == 30
            mock_get_size.assert_called_once()

    def test_create_process(self):
        """Test process creation on Windows."""
        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_popen.return_value = mock_process
            
            proc, _ = create_process(['echo', 'test'])
            
            assert mock_popen.called
            args, kwargs = mock_popen.call_args
            assert kwargs['stdin'] == subprocess.PIPE
            assert kwargs['stdout'] == subprocess.PIPE
            assert kwargs['stderr'] == subprocess.PIPE
            assert 'startupinfo' in kwargs
            assert kwargs['startupinfo'].dwFlags & subprocess.STARTF_USESHOWWINDOW

    def test_run_interactive_command(self):
        """Test running an interactive command on Windows."""
        test_output = "Test output\n"
        
        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.stdout = MagicMock()
            mock_process.stdout.read.return_value = test_output.encode()
            mock_process.wait.return_value = 0
            mock_popen.return_value = mock_process
            
            output, return_code = run_interactive_command(['echo', 'test'])
            assert return_code == 0
            assert "Test output" in output.decode()

    def test_windows_dependencies(self):
        """Test that required Windows dependencies are available."""
        if sys.platform == "win32":
            import msvcrt
            import win32pipe
            import win32file
            import win32con
            import win32process
            
            # If we get here without ImportError, the test passes
            assert True
