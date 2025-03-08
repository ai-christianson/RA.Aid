"""Unit tests for the --show-thoughts CLI flag."""

import pytest
from unittest.mock import patch, MagicMock

from ra_aid.__main__ import parse_arguments


def test_show_thoughts_flag():
    """Test that the --show-thoughts flag is correctly parsed."""
    # Test default value (False)
    args = parse_arguments(["-m", "test message"])
    assert args.show_thoughts is False
    
    # Test with flag (True)
    args = parse_arguments(["-m", "test message", "--show-thoughts"])
    assert args.show_thoughts is True


@pytest.fixture(autouse=True)
def mock_config_repository():
    """Mock the ConfigRepository to avoid database operations during tests"""
    with patch('ra_aid.database.repositories.config_repository.config_repo_var') as mock_repo_var:
        # Setup a mock repository
        mock_repo = MagicMock()
        
        # Create a dictionary to simulate config
        config = {}
        
        # Setup get method to return config values
        def get_config(key, default=None):
            return config.get(key, default)
        mock_repo.get.side_effect = get_config
        
        # Setup set method to update config values
        def set_config(key, value):
            config[key] = value
        mock_repo.set.side_effect = set_config
        
        # Setup update method to update multiple config values
        def update_config(config_dict):
            config.update(config_dict)
        mock_repo.update.side_effect = update_config
        
        # Setup get_all method to return the config dict
        def get_all_config():
            return config.copy()
        mock_repo.get_all.side_effect = get_all_config
        
        # Make the mock context var return our mock repo
        mock_repo_var.get.return_value = mock_repo
        
        yield mock_repo


def test_show_thoughts_config(mock_config_repository):
    """Test that the show_thoughts flag is correctly stored in config."""
    import sys
    from unittest.mock import patch
    
    from ra_aid.__main__ import main
    
    # Reset mocks
    mock_config_repository.set.reset_mock()
    
    # For testing, we need to patch ConfigRepositoryManager.__enter__ to return our mock
    with patch('ra_aid.database.repositories.config_repository.ConfigRepositoryManager.__enter__', return_value=mock_config_repository):
        # Test with --show-thoughts flag
        with patch.object(sys, "argv", ["ra-aid", "-m", "test message", "--show-thoughts"]):
            with patch("ra_aid.__main__.run_research_agent", return_value=None):
                main()
                # Verify the show_thoughts flag is set to True in config
                mock_config_repository.set.assert_any_call("show_thoughts", True)
        
        # Reset mocks
        mock_config_repository.set.reset_mock()
        
        # Test without --show-thoughts flag (default: False)
        with patch.object(sys, "argv", ["ra-aid", "-m", "test message"]):
            with patch("ra_aid.__main__.run_research_agent", return_value=None):
                main()
                # Verify the show_thoughts flag is set to False in config
                mock_config_repository.set.assert_any_call("show_thoughts", False)