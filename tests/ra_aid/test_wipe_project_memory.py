"""Tests for wipe_project_memory functionality."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from ra_aid.__main__ import wipe_project_memory, parse_arguments


def test_wipe_project_memory_function():
    """Test that wipe_project_memory function correctly deletes the database file."""
    # Create a temporary directory to simulate the project directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a fake .ra-aid directory and pk.db file
        ra_aid_dir = Path(os.path.join(temp_dir, ".ra-aid"))
        os.makedirs(ra_aid_dir, exist_ok=True)
        db_path = os.path.join(ra_aid_dir, "pk.db")
        
        # Create an empty file
        with open(db_path, "w") as f:
            f.write("")
        
        # Verify the file exists
        assert os.path.exists(db_path)
        
        # Mock getcwd to return our temp directory
        with patch("os.getcwd", return_value=temp_dir):
            # Call the function
            result = wipe_project_memory()
            
            # Verify the file no longer exists
            assert not os.path.exists(db_path)
            assert result == "Project memory wiped successfully."


def test_wipe_project_memory_no_file():
    """Test wipe_project_memory when no database file exists."""
    # Create a temporary directory without a pk.db file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock getcwd to return our temp directory
        with patch("os.getcwd", return_value=temp_dir):
            # Call the function
            result = wipe_project_memory()
            
            # Verify the result message
            assert result == "No project memory found to wipe."


def test_wipe_project_memory_flag():
    """Test that the --wipe-project-memory flag is properly parsed."""
    # Test without the flag
    args = parse_arguments(["-m", "test message"])
    assert not hasattr(args, "wipe_project_memory") or not args.wipe_project_memory
    
    # Test with the flag
    args = parse_arguments(["-m", "test message", "--wipe-project-memory"])
    assert args.wipe_project_memory is True


def test_build_status_shows_reset_option():
    """Test that build_status function shows reset option when there are items in memory."""
    from unittest.mock import patch, MagicMock
    
    from ra_aid.__main__ import build_status
    
    # Mock repositories to return different numbers of items
    with patch("ra_aid.__main__.get_key_fact_repository") as mock_fact_repo, \
         patch("ra_aid.__main__.get_key_snippet_repository") as mock_snippet_repo, \
         patch("ra_aid.__main__.get_research_note_repository") as mock_note_repo, \
         patch("ra_aid.__main__.get_config_repository") as mock_config_repo:
         
        # Set up mock repositories to return specific results with get and get_all
        # For key_fact_repository
        def mock_fact_get(fact_id):
            if 1 <= fact_id <= 3:
                return MagicMock(id=fact_id, content=f"Fact {fact_id}")
            return None
        mock_fact_repo.return_value.get.side_effect = mock_fact_get
        mock_fact_repo.return_value.get_all.return_value = [1, 2, 3]  # 3 facts
        
        # For key_snippet_repository
        def mock_snippet_get(snippet_id):
            if snippet_id == 1:
                return MagicMock(id=1, filepath="test.py", line_number=1, snippet="test")
            return None
        mock_snippet_repo.return_value.get.side_effect = mock_snippet_get
        mock_snippet_repo.return_value.get_all.return_value = [1]  # 1 snippet
        
        # For research_note_repository
        def mock_note_get(note_id):
            if 1 <= note_id <= 2:
                return MagicMock(id=note_id, content=f"Note {note_id}")
            return None
        mock_note_repo.return_value.get.side_effect = mock_note_get
        mock_note_repo.return_value.get_all.return_value = [1, 2]  # 2 notes
        mock_config_repo.return_value.get.return_value = None
        
        # Call build_status
        status = build_status()
        
        # Convert status to string for easier assertion
        status_str = str(status)
        
        # Verify it includes the memory statistics with reset option
        assert "Memory: 3 facts, 1 snippets, 2 notes" in status_str
        assert "use --wipe-project-memory to reset" in status_str
        
        # Test with empty memory - should not show reset option
        # Update both get and get_all mocks
        mock_fact_repo.return_value.get.side_effect = lambda fact_id: None
        mock_fact_repo.return_value.get_all.return_value = []
        
        mock_snippet_repo.return_value.get.side_effect = lambda snippet_id: None
        mock_snippet_repo.return_value.get_all.return_value = []
        
        mock_note_repo.return_value.get.side_effect = lambda note_id: None
        mock_note_repo.return_value.get_all.return_value = []
        
        # Call build_status again
        status = build_status()
        status_str = str(status)
        
        # Verify it doesn't include the reset option
        assert "Memory: 0 facts, 0 snippets, 0 notes" in status_str
        assert "use --wipe-project-memory to reset" not in status_str


def test_main_with_wipe_project_memory_flag():
    """Test that the main function properly calls wipe_project_memory when flag is set."""
    from ra_aid.__main__ import main

    # Create a mock args object with wipe_project_memory=True and project_state_dir=None
    mock_args = MagicMock()
    mock_args.wipe_project_memory = True
    mock_args.project_state_dir = None
    
    # Mock the wipe_project_memory function to raise SystemExit after being called
    def mock_wipe_side_effect(custom_dir=None):
        raise SystemExit(0)
    
    mock_wipe = MagicMock(side_effect=mock_wipe_side_effect)
    
    # Mock all necessary dependencies to prevent actual operations
    with patch("ra_aid.__main__.wipe_project_memory", mock_wipe), \
         patch("ra_aid.__main__.parse_arguments", return_value=mock_args), \
         patch("ra_aid.__main__.setup_logging"), \
         patch("ra_aid.__main__.get_config_repository"), \
         patch("ra_aid.__main__.launch_server"), \
         patch("ra_aid.__main__.DatabaseManager"):
        
        # Call main() and catch SystemExit since we're raising it
        try:
            main()
        except SystemExit:
            pass
        
        # Verify wipe_project_memory was called with the custom_dir parameter
        mock_wipe.assert_called_once_with(custom_dir=mock_args.project_state_dir)