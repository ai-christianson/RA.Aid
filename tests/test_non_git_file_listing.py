"""Test file listing for non-git directories."""

import os
import tempfile
from pathlib import Path

import pytest

from ra_aid.file_listing import get_file_listing


def test_non_git_file_listing():
    """Test that file listing works correctly for non-git directories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a few test files
        file1 = Path(temp_dir) / "file1.txt"
        file2 = Path(temp_dir) / "file2.py"
        file3 = Path(temp_dir) / ".hidden_file"  # Hidden file
        
        # Create a subdirectory with files
        subdir = Path(temp_dir) / "subdir"
        os.makedirs(subdir)
        file4 = subdir / "file4.js"
        
        # Create excluded directories
        ra_aid_dir = Path(temp_dir) / ".ra-aid"
        venv_dir = Path(temp_dir) / ".venv"
        os.makedirs(ra_aid_dir)
        os.makedirs(venv_dir)
        
        # Create files in excluded directories
        ra_aid_file = ra_aid_dir / "config.json"
        venv_file = venv_dir / "activate"
        
        # Write content to all files
        for file_path in [file1, file2, file3, file4, ra_aid_file, venv_file]:
            with open(file_path, "w") as f:
                f.write("test content")
        
        # Test regular file listing (should exclude hidden files and directories)
        files, count = get_file_listing(temp_dir)
        assert count == 3  # file1.txt, file2.py, subdir/file4.js
        assert set(files) == {"file1.txt", "file2.py", os.path.join("subdir", "file4.js")}
        
        # Test with include_hidden=True
        files_with_hidden, count_with_hidden = get_file_listing(temp_dir, include_hidden=True)
        assert count_with_hidden == 4  # Including .hidden_file
        assert ".hidden_file" in files_with_hidden
        
        # Test with limit
        files_limited, count_limited = get_file_listing(temp_dir, limit=2)
        assert len(files_limited) == 2
        assert count_limited == 3  # Total count should still be 3


def test_non_git_directory_with_excluded_dirs():
    """Test that excluded directories are properly handled in non-git directories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create regular files
        file1 = Path(temp_dir) / "file1.txt"
        with open(file1, "w") as f:
            f.write("test content")
            
        # Create excluded directories with files
        excluded_dirs = [".ra-aid", ".venv", ".git", ".aider", "__pycache__"]
        for excluded_dir in excluded_dirs:
            dir_path = Path(temp_dir) / excluded_dir
            os.makedirs(dir_path)
            with open(dir_path / "test_file.txt", "w") as f:
                f.write("test content")
                
        # Get file listing
        files, count = get_file_listing(temp_dir)
        
        # Should only include the regular file
        assert count == 1
        assert files == ["file1.txt"]


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
