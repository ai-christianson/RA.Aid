"""Test for list_directory_tree with file path support."""

import tempfile
import os
from pathlib import Path

from ra_aid.tools import list_directory_tree


def test_list_directory_tree_with_file():
    """Test that list_directory_tree works with a file path."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(b"Some test content")
        tmp_file_path = tmp_file.name
    
    try:
        # Test with file path
        result = list_directory_tree.invoke({"path": tmp_file_path})
        
        # Basic verification that the output contains the filename
        filename = os.path.basename(tmp_file_path)
        assert filename in result
        
        # Test with size option
        result_with_size = list_directory_tree.invoke({"path": tmp_file_path, "show_size": True})
        assert "(" in result_with_size  # Size information should be present
        
        # Test with modified time option
        result_with_time = list_directory_tree.invoke({"path": tmp_file_path, "show_modified": True})
        assert "Modified:" in result_with_time
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


if __name__ == "__main__":
    test_list_directory_tree_with_file()
    print("All tests passed!")
