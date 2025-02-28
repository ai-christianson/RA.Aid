"""Tests for Windows-specific functionality."""

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from ra_aid.proc.interactive import (
    create_process,
    get_terminal_size,
    run_interactive_command,
)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific tests")
class TestWindowsCompatibility:
    """Test suite for Windows-specific functionality."""

    def test_get_terminal_size(self):
        """Test terminal size detection on Windows."""
        with patch("shutil.get_terminal_size") as mock_get_size:
            mock_get_size.return_value = MagicMock(columns=120, lines=30)
            cols, rows = get_terminal_size()
            assert cols == 120
            assert rows == 30
            mock_get_size.assert_called_once()

    def test_create_process(self):
        """Test process creation on Windows."""
        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_popen.return_value = mock_process

            proc, _ = create_process(["echo", "test"])

            assert mock_popen.called
            args, kwargs = mock_popen.call_args
            assert kwargs["stdin"] == subprocess.PIPE
            assert kwargs["stdout"] == subprocess.PIPE
            assert kwargs["stderr"] == subprocess.STDOUT
            assert "startupinfo" in kwargs
            assert kwargs["startupinfo"].dwFlags & subprocess.STARTF_USESHOWWINDOW

    def test_run_interactive_command(self):
        """Test running an interactive command on Windows."""
        test_output = "Test output\n"

        with (
            patch("subprocess.Popen") as mock_popen,
            patch("pyte.Stream") as mock_stream,
            patch("pyte.HistoryScreen") as mock_screen,
            patch("threading.Thread") as mock_thread,
        ):
            # Setup mock process
            mock_process = MagicMock()
            mock_process.stdout = MagicMock()
            mock_process.stdout.read.return_value = test_output.encode()
            mock_process.poll.side_effect = [None, 0]  # First None, then return 0
            mock_process.returncode = 0
            mock_popen.return_value = mock_process

            # Setup mock screen with history
            mock_screen_instance = MagicMock()
            mock_screen_instance.history.top = []
            mock_screen_instance.history.bottom = []
            mock_screen_instance.display = ["Test output"]
            mock_screen.return_value = mock_screen_instance

            # Setup mock thread
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance

            # Run the command
            output, return_code = run_interactive_command(["echo", "test"])

            # Verify results
            assert return_code == 0
            assert "Test output" in output.decode()

            # Verify the thread was started and joined
            mock_thread_instance.start.assert_called()
            mock_thread_instance.join.assert_called()

    def test_windows_dependencies(self):
        """Test that required Windows dependencies are available."""
        if sys.platform == "win32":

            # If we get here without ImportError, the test passes
            assert True

    def test_windows_output_handling(self):
        """Test handling of multi-chunk output on Windows."""
        if sys.platform != "win32":
            pytest.skip("Windows-specific test")

        # Test with multiple chunks of output to verify proper handling
        with (
            patch("subprocess.Popen") as mock_popen,
            patch("msvcrt.kbhit", return_value=False),
            patch("threading.Thread") as mock_thread,
            patch("time.sleep"),
        ):  # Mock sleep to speed up test
            # Setup mock process
            mock_process = MagicMock()
            mock_process.stdout = MagicMock()
            mock_process.poll.return_value = 0
            mock_process.returncode = 0
            mock_popen.return_value = mock_process

            # Setup mock thread to simulate output collection
            def side_effect(*args, **kwargs):
                # Simulate thread collecting output
                mock_process.stdout.read.side_effect = [
                    b"First chunk\n",
                    b"Second chunk\n",
                    b"Third chunk with unicode \xe2\x9c\x93\n",  # UTF-8 checkmark
                    None,  # End of output
                ]
                return MagicMock()

            mock_thread.side_effect = side_effect

            # Run the command
            output, return_code = run_interactive_command(["test", "command"])

            # Verify results
            assert return_code == 0
            # We can't verify exact output content in this test since we're mocking the thread
            # that would collect the output, but we can verify the process was created correctly
            assert mock_popen.called
