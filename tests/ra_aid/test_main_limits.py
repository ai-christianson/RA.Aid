"""Unit tests for cost and token limit CLI arguments in __main__.py."""

import contextlib
import pytest
from unittest.mock import patch, MagicMock

from ra_aid.__main__ import parse_arguments


def test_max_cost_argument():
    """Test --max-cost argument parsing and validation."""
    # Test valid positive cost
    args = parse_arguments(["-m", "test", "--max-cost", "5.0"])
    assert args.max_cost == 5.0
    
    # Test default None value
    args = parse_arguments(["-m", "test"])
    assert args.max_cost is None
    
    # Test negative cost raises error
    with pytest.raises(SystemExit):
        parse_arguments(["-m", "test", "--max-cost", "-1.0"])
    
    # Test zero cost raises error
    with pytest.raises(SystemExit):
        parse_arguments(["-m", "test", "--max-cost", "0"])
    
    # Test very small positive cost
    args = parse_arguments(["-m", "test", "--max-cost", "0.01"])
    assert args.max_cost == 0.01


def test_max_tokens_argument():
    """Test --max-tokens argument parsing and validation."""
    # Test valid positive tokens
    args = parse_arguments(["-m", "test", "--max-tokens", "10000"])
    assert args.max_tokens == 10000
    
    # Test default None value
    args = parse_arguments(["-m", "test"])
    assert args.max_tokens is None
    
    # Test negative tokens raises error
    with pytest.raises(SystemExit):
        parse_arguments(["-m", "test", "--max-tokens", "-100"])
    
    # Test zero tokens raises error
    with pytest.raises(SystemExit):
        parse_arguments(["-m", "test", "--max-tokens", "0"])
    
    # Test minimum positive tokens
    args = parse_arguments(["-m", "test", "--max-tokens", "1"])
    assert args.max_tokens == 1


def test_exit_at_limit_argument():
    """Test --exit-at-limit flag parsing."""
    # Test flag enabled
    args = parse_arguments(["-m", "test", "--exit-at-limit"])
    assert args.exit_at_limit is True
    
    # Test default False value
    args = parse_arguments(["-m", "test"])
    assert args.exit_at_limit is False


def test_combined_limit_arguments():
    """Test using multiple limit arguments together."""
    args = parse_arguments([
        "-m", "test",
        "--max-cost", "2.5",
        "--max-tokens", "50000",
        "--exit-at-limit"
    ])
    assert args.max_cost == 2.5
    assert args.max_tokens == 50000
    assert args.exit_at_limit is True


def test_limit_arguments_with_other_flags():
    """Test limit arguments work with other CLI flags."""
    args = parse_arguments([
        "-m", "test",
        "--max-cost", "10.0",
        "--show-cost",
        "--cowboy-mode",
        "--provider", "openai"
    ])
    assert args.max_cost == 10.0
    assert args.show_cost is True
    assert args.cowboy_mode is True
    assert args.provider == "openai"


def test_invalid_cost_float():
    """Test that invalid float values for max-cost raise errors."""
    with pytest.raises(SystemExit):
        parse_arguments(["-m", "test", "--max-cost", "not-a-number"])


def test_invalid_tokens_int():
    """Test that invalid integer values for max-tokens raise errors."""
    with pytest.raises(SystemExit):
        parse_arguments(["-m", "test", "--max-tokens", "not-a-number"])


@pytest.fixture(autouse=True)
def mock_config_repository():
    """Mock the ConfigRepository to avoid database operations during tests"""
    with patch('ra_aid.database.repositories.config_repository.config_repo_var') as mock_repo_var:
        # Setup a mock repository
        mock_repo = MagicMock()
        
        # Create a dictionary to simulate config
        config = {}
        
        # Setup set method to update config values
        def set_config(key, value):
            config[key] = value
        mock_repo.set.side_effect = set_config
        
        # Setup update method to update multiple config values
        def update_config(config_dict):
            for k, v in config_dict.items():
                config[k] = v
        mock_repo.update.side_effect = update_config
        
        # Setup get method to return config values
        def get_config(key, default=None):
            return config.get(key, default)
        mock_repo.get.side_effect = get_config
        
        # Make the mock context var return our mock repo
        mock_repo_var.get.return_value = mock_repo
        
        yield mock_repo


def test_limit_config_storage(mock_config_repository):
    """Test that limit arguments are stored in config repository."""
    import sys
    from ra_aid.__main__ import main
    
    with contextlib.ExitStack() as stack:
        # Mock session manager to return a valid session ID
        mock_session = MagicMock()
        mock_session.get_current_session_id.return_value = 1
        mock_session.create_session.return_value = None
        
        # Mock context manager __enter__ methods to return mock repositories
        stack.enter_context(patch('ra_aid.database.connection.DatabaseManager.__enter__', return_value=MagicMock()))
        stack.enter_context(patch('ra_aid.database.repositories.session_repository.SessionRepositoryManager.__enter__', return_value=mock_session))
        stack.enter_context(patch('ra_aid.database.repositories.key_fact_repository.KeyFactRepositoryManager.__enter__', return_value=MagicMock()))
        stack.enter_context(patch('ra_aid.database.repositories.key_snippet_repository.KeySnippetRepositoryManager.__enter__', return_value=MagicMock()))
        stack.enter_context(patch('ra_aid.database.repositories.human_input_repository.HumanInputRepositoryManager.__enter__', return_value=MagicMock()))
        stack.enter_context(patch('ra_aid.database.repositories.research_note_repository.ResearchNoteRepositoryManager.__enter__', return_value=MagicMock()))
        stack.enter_context(patch('ra_aid.database.repositories.related_files_repository.RelatedFilesRepositoryManager.__enter__', return_value=MagicMock()))
        stack.enter_context(patch('ra_aid.database.repositories.trajectory_repository.TrajectoryRepositoryManager.__enter__', return_value=MagicMock()))
        stack.enter_context(patch('ra_aid.database.repositories.work_log_repository.WorkLogRepositoryManager.__enter__', return_value=MagicMock()))
        stack.enter_context(patch('ra_aid.database.repositories.config_repository.ConfigRepositoryManager.__enter__', return_value=mock_config_repository))
        stack.enter_context(patch('ra_aid.env_inv_context.EnvInvManager.__enter__', return_value=MagicMock()))
        
        # Mock the repository getter functions that access contextvars
        stack.enter_context(patch('ra_aid.__main__.get_trajectory_repository', return_value=MagicMock()))
        stack.enter_context(patch('ra_aid.__main__.get_human_input_repository', return_value=MagicMock()))
        stack.enter_context(patch('ra_aid.__main__.get_key_fact_repository', return_value=MagicMock()))
        stack.enter_context(patch('ra_aid.__main__.get_key_snippet_repository', return_value=MagicMock()))
        stack.enter_context(patch('ra_aid.__main__.get_research_note_repository', return_value=MagicMock()))
        
        # Mock other dependencies
        stack.enter_context(patch('ra_aid.__main__.check_dependencies'))
        stack.enter_context(patch('ra_aid.__main__.validate_environment', return_value=(True, [], True, [])))
        stack.enter_context(patch('ra_aid.__main__.run_research_agent'))
        stack.enter_context(patch('ra_aid.__main__.ensure_migrations_applied', return_value=True))
        stack.enter_context(patch('ra_aid.__main__.EnvDiscovery'))
        
        stack.enter_context(patch.object(sys, "argv", [
            "ra-aid", "-m", "test",
            "--max-cost", "5.0",
            "--max-tokens", "10000", 
            "--exit-at-limit"
        ]))
        
        main()
        
        # Verify config values were set
        mock_config_repository.set.assert_any_call("max_cost", 5.0)
        mock_config_repository.set.assert_any_call("max_tokens", 10000)
        mock_config_repository.set.assert_any_call("exit_at_limit", True)


def test_limit_config_storage_none_values(mock_config_repository):
    """Test that None values are stored when limits not specified."""
    import sys
    from ra_aid.__main__ import main
    
    with contextlib.ExitStack() as stack:
        # Mock session manager to return a valid session ID
        mock_session = MagicMock()
        mock_session.get_current_session_id.return_value = 1
        mock_session.create_session.return_value = None
        
        # Mock context manager __enter__ methods to return mock repositories
        stack.enter_context(patch('ra_aid.database.connection.DatabaseManager.__enter__', return_value=MagicMock()))
        stack.enter_context(patch('ra_aid.database.repositories.session_repository.SessionRepositoryManager.__enter__', return_value=mock_session))
        stack.enter_context(patch('ra_aid.database.repositories.key_fact_repository.KeyFactRepositoryManager.__enter__', return_value=MagicMock()))
        stack.enter_context(patch('ra_aid.database.repositories.key_snippet_repository.KeySnippetRepositoryManager.__enter__', return_value=MagicMock()))
        stack.enter_context(patch('ra_aid.database.repositories.human_input_repository.HumanInputRepositoryManager.__enter__', return_value=MagicMock()))
        stack.enter_context(patch('ra_aid.database.repositories.research_note_repository.ResearchNoteRepositoryManager.__enter__', return_value=MagicMock()))
        stack.enter_context(patch('ra_aid.database.repositories.related_files_repository.RelatedFilesRepositoryManager.__enter__', return_value=MagicMock()))
        stack.enter_context(patch('ra_aid.database.repositories.trajectory_repository.TrajectoryRepositoryManager.__enter__', return_value=MagicMock()))
        stack.enter_context(patch('ra_aid.database.repositories.work_log_repository.WorkLogRepositoryManager.__enter__', return_value=MagicMock()))
        stack.enter_context(patch('ra_aid.database.repositories.config_repository.ConfigRepositoryManager.__enter__', return_value=mock_config_repository))
        stack.enter_context(patch('ra_aid.env_inv_context.EnvInvManager.__enter__', return_value=MagicMock()))
        
        # Mock the repository getter functions that access contextvars
        stack.enter_context(patch('ra_aid.__main__.get_trajectory_repository', return_value=MagicMock()))
        stack.enter_context(patch('ra_aid.__main__.get_human_input_repository', return_value=MagicMock()))
        stack.enter_context(patch('ra_aid.__main__.get_key_fact_repository', return_value=MagicMock()))
        stack.enter_context(patch('ra_aid.__main__.get_key_snippet_repository', return_value=MagicMock()))
        stack.enter_context(patch('ra_aid.__main__.get_research_note_repository', return_value=MagicMock()))
        
        # Mock other dependencies
        stack.enter_context(patch('ra_aid.__main__.check_dependencies'))
        stack.enter_context(patch('ra_aid.__main__.validate_environment', return_value=(True, [], True, [])))
        stack.enter_context(patch('ra_aid.__main__.run_research_agent'))
        stack.enter_context(patch('ra_aid.__main__.ensure_migrations_applied', return_value=True))
        stack.enter_context(patch('ra_aid.__main__.EnvDiscovery'))
        
        stack.enter_context(patch.object(sys, "argv", ["ra-aid", "-m", "test"]))
        
        main()
        
        # Verify None values were set
        mock_config_repository.set.assert_any_call("max_cost", None)
        mock_config_repository.set.assert_any_call("max_tokens", None)
        mock_config_repository.set.assert_any_call("exit_at_limit", False)
