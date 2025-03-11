"""Unit tests for __main__.py argument parsing."""

import pytest
from unittest.mock import patch, MagicMock

from ra_aid.__main__ import parse_arguments
from ra_aid.config import DEFAULT_RECURSION_LIMIT
from ra_aid.database.repositories.work_log_repository import WorkLogEntry
from ra_aid.database.repositories.config_repository import ConfigRepositoryManager, get_config_repository


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


@pytest.fixture
def mock_dependencies(monkeypatch):
    """Mock all dependencies needed for main()."""
    # Mock dependencies that interact with external systems
    monkeypatch.setattr("ra_aid.__main__.check_dependencies", lambda: None)
    monkeypatch.setattr("ra_aid.__main__.validate_environment", lambda args: (True, [], True, []))
    monkeypatch.setattr("ra_aid.__main__.create_agent", lambda *args, **kwargs: None)
    monkeypatch.setattr("ra_aid.__main__.run_agent_with_retry", lambda *args, **kwargs: None)
    monkeypatch.setattr("ra_aid.__main__.run_research_agent", lambda *args, **kwargs: None)
    monkeypatch.setattr("ra_aid.agents.planning_agent.run_planning_agent", lambda *args, **kwargs: None)
    
    # Mock LLM initialization
    def mock_config_update(*args, **kwargs):
        config_repo = get_config_repository()
        if kwargs.get("temperature"):
            config_repo.set("temperature", kwargs["temperature"])
        return None

    monkeypatch.setattr("ra_aid.__main__.initialize_llm", mock_config_update)


@pytest.fixture(autouse=True)
def mock_related_files_repository():
    """Mock the RelatedFilesRepository to avoid database operations during tests"""
    with patch('ra_aid.database.repositories.related_files_repository.related_files_repo_var') as mock_repo_var:
        # Setup a mock repository
        mock_repo = MagicMock()
        
        # Create a dictionary to simulate stored files
        related_files = {}
        
        # Setup get_all method to return the files dict
        mock_repo.get_all.return_value = related_files
        
        # Setup format_related_files method
        mock_repo.format_related_files.return_value = [f"ID#{file_id} {filepath}" for file_id, filepath in sorted(related_files.items())]
        
        # Make the mock context var return our mock repo
        mock_repo_var.get.return_value = mock_repo
        
        yield mock_repo


@pytest.fixture(autouse=True)
def mock_work_log_repository():
    """Mock the WorkLogRepository to avoid database operations during tests"""
    with patch('ra_aid.database.repositories.work_log_repository.work_log_repo_var') as mock_repo_var:
        # Setup a mock repository
        mock_repo = MagicMock()
        
        # Setup local in-memory storage
        entries = []
        
        # Mock add_entry method
        def mock_add_entry(event):
            from datetime import datetime
            entry = {"timestamp": datetime.now().isoformat(), "event": event}
            entries.append(entry)
        mock_repo.add_entry.side_effect = mock_add_entry
        
        # Mock get_all method
        def mock_get_all():
            return entries.copy()
        mock_repo.get_all.side_effect = mock_get_all
        
        # Mock clear method
        def mock_clear():
            entries.clear()
        mock_repo.clear.side_effect = mock_clear
        
        # Mock format_work_log method
        def mock_format_work_log():
            if not entries:
                return "No work log entries"
                
            formatted_entries = []
            for entry in entries:
                formatted_entries.extend([
                    f"## {entry['timestamp']}",
                    "",
                    entry["event"],
                    "",  # Blank line between entries
                ])
                
            return "\n".join(formatted_entries).rstrip()  # Remove trailing newline
        mock_repo.format_work_log.side_effect = mock_format_work_log
        
        # Make the mock context var return our mock repo
        mock_repo_var.get.return_value = mock_repo
        
        yield mock_repo


def test_recursion_limit_in_global_config(mock_dependencies, mock_config_repository):
    """Test that recursion limit is correctly set in global config."""
    import sys
    from unittest.mock import patch

    from ra_aid.__main__ import main

    # Clear the mock repository before each test
    mock_config_repository.update.reset_mock()
    
    # For testing, we need to patch ConfigRepositoryManager.__enter__ to return our mock
    with patch('ra_aid.database.repositories.config_repository.ConfigRepositoryManager.__enter__', return_value=mock_config_repository):
        # Test default recursion limit
        with patch.object(sys, "argv", ["ra-aid", "-m", "test message"]):
            main()
            # Check that the recursion_limit value was included in the update call
            mock_config_repository.update.assert_called()
            # Get the call arguments
            call_args = mock_config_repository.update.call_args_list
            # Find the call that includes recursion_limit
            recursion_limit_found = False
            for args, _ in call_args:
                config_dict = args[0]
                if "recursion_limit" in config_dict and config_dict["recursion_limit"] == DEFAULT_RECURSION_LIMIT:
                    recursion_limit_found = True
                    break
            assert recursion_limit_found, f"recursion_limit not found in update calls: {call_args}"

        # Reset mock to clear call history
        mock_config_repository.update.reset_mock()
        
        # Test custom recursion limit
        with patch.object(sys, "argv", ["ra-aid", "-m", "test message", "--recursion-limit", "50"]):
            main()
            # Check that the recursion_limit value was included in the update call
            mock_config_repository.update.assert_called()
            # Get the call arguments
            call_args = mock_config_repository.update.call_args_list
            # Find the call that includes recursion_limit with value 50
            recursion_limit_found = False
            for args, _ in call_args:
                config_dict = args[0]
                if "recursion_limit" in config_dict and config_dict["recursion_limit"] == 50:
                    recursion_limit_found = True
                    break
            assert recursion_limit_found, f"recursion_limit=50 not found in update calls: {call_args}"


def test_negative_recursion_limit():
    """Test that negative recursion limit raises error."""
    with pytest.raises(SystemExit):
        parse_arguments(["-m", "test message", "--recursion-limit", "-1"])


def test_zero_recursion_limit():
    """Test that zero recursion limit raises error."""
    with pytest.raises(SystemExit):
        parse_arguments(["-m", "test message", "--recursion-limit", "0"])


def test_config_settings(mock_dependencies, mock_config_repository):
    """Test that various settings are correctly applied in global config."""
    import sys
    from unittest.mock import patch

    from ra_aid.__main__ import main
    
    # Clear the mock repository before each test
    mock_config_repository.update.reset_mock()
    
    # For testing, we need to patch ConfigRepositoryManager.__enter__ to return our mock
    with patch('ra_aid.database.repositories.config_repository.ConfigRepositoryManager.__enter__', return_value=mock_config_repository):
        with patch.object(
            sys,
            "argv",
            [
                "ra-aid",
                "-m",
                "test message",
                "--cowboy-mode",
                "--research-only",
                "--provider",
                "anthropic",
                "--model",
                "claude-3-7-sonnet-20250219",
                "--expert-provider",
                "openai",
                "--expert-model",
                "gpt-4",
                "--temperature",
                "0.7",
                "--disable-limit-tokens",
            ],
        ):
            main()
            # Verify config values are set via the update method
            mock_config_repository.update.assert_called()
            # Get the call arguments
            call_args = mock_config_repository.update.call_args_list
            
            # Check for config values in the update calls
            for args, _ in call_args:
                config_dict = args[0]
                if "cowboy_mode" in config_dict:
                    assert config_dict["cowboy_mode"] is True
                if "research_only" in config_dict:
                    assert config_dict["research_only"] is True
                if "limit_tokens" in config_dict:
                    assert config_dict["limit_tokens"] is False
            
            # Check provider and model settings via set method
            mock_config_repository.set.assert_any_call("provider", "anthropic")
            mock_config_repository.set.assert_any_call("model", "claude-3-7-sonnet-20250219")
            mock_config_repository.set.assert_any_call("expert_provider", "openai")
            mock_config_repository.set.assert_any_call("expert_model", "gpt-4")


def test_temperature_validation(mock_dependencies, mock_config_repository):
    """Test that temperature argument is correctly passed to initialize_llm."""
    import sys
    from unittest.mock import patch, ANY

    from ra_aid.__main__ import main

    # For testing, we need to patch ConfigRepositoryManager.__enter__ to return our mock
    with patch('ra_aid.database.repositories.config_repository.ConfigRepositoryManager.__enter__', return_value=mock_config_repository):
        # Test valid temperature (0.7)
        with patch("ra_aid.__main__.initialize_llm", return_value=None) as mock_init_llm:
            # Also patch any calls that would actually use the mocked initialize_llm function
            with patch("ra_aid.__main__.run_research_agent", return_value=None):
                with patch("ra_aid.agents.planning_agent.run_planning_agent", return_value=None):
                    with patch.object(
                        sys, "argv", ["ra-aid", "-m", "test", "--temperature", "0.7"]
                    ):
                        main()
                        # Verify that the temperature was set in the config repository
                        mock_config_repository.set.assert_any_call("temperature", 0.7)

    # Test invalid temperature (2.1)
    with pytest.raises(SystemExit):
        with patch.object(
            sys, "argv", ["ra-aid", "-m", "test", "--temperature", "2.1"]
        ):
            main()


def test_missing_message():
    """Test that missing message argument raises error."""
    # Test chat mode which doesn't require message
    args = parse_arguments(["--chat"])
    assert args.chat is True
    assert args.message is None

    # Test non-chat mode requires message
    args = parse_arguments(["--provider", "openai"])
    assert args.message is None

    # Verify message is captured when provided
    args = parse_arguments(["-m", "test"])
    assert args.message == "test"


def test_research_model_provider_args(mock_dependencies, mock_config_repository):
    """Test that research-specific model/provider args are correctly stored in config."""
    import sys
    from unittest.mock import patch

    from ra_aid.__main__ import main

    # Reset mocks
    mock_config_repository.set.reset_mock()
    
    # For testing, we need to patch ConfigRepositoryManager.__enter__ to return our mock
    with patch('ra_aid.database.repositories.config_repository.ConfigRepositoryManager.__enter__', return_value=mock_config_repository):
        with patch.object(
            sys,
            "argv",
            [
                "ra-aid",
                "-m",
                "test message",
                "--research-provider",
                "anthropic",
                "--research-model",
                "claude-3-haiku-20240307",
                "--planner-provider",
                "openai",
                "--planner-model",
                "gpt-4",
            ],
        ):
            main()
            # Verify the mock repo's set method was called with the expected values
            mock_config_repository.set.assert_any_call("research_provider", "anthropic")
            mock_config_repository.set.assert_any_call("research_model", "claude-3-haiku-20240307")
            mock_config_repository.set.assert_any_call("planner_provider", "openai")
            mock_config_repository.set.assert_any_call("planner_model", "gpt-4")


def test_planner_model_provider_args(mock_dependencies, mock_config_repository):
    """Test that planner provider/model args fall back to main config when not specified."""
    import sys
    from unittest.mock import patch

    from ra_aid.__main__ import main

    # Reset mocks
    mock_config_repository.set.reset_mock()
    
    # For testing, we need to patch ConfigRepositoryManager.__enter__ to return our mock
    with patch('ra_aid.database.repositories.config_repository.ConfigRepositoryManager.__enter__', return_value=mock_config_repository):
        with patch.object(
            sys,
            "argv",
            ["ra-aid", "-m", "test message", "--provider", "openai", "--model", "gpt-4"],
        ):
            main()
            # Verify the mock repo's set method was called with the expected values
            mock_config_repository.set.assert_any_call("planner_provider", "openai")
            mock_config_repository.set.assert_any_call("planner_model", "gpt-4")


def test_use_aider_flag(mock_dependencies, mock_config_repository):
    """Test that use-aider flag is correctly stored in config."""
    import sys
    from unittest.mock import patch

    from ra_aid.__main__ import main
    from ra_aid.tool_configs import MODIFICATION_TOOLS, set_modification_tools

    # Reset mocks
    mock_config_repository.update.reset_mock()
    
    # Reset to default state
    set_modification_tools(False)

    # For testing, we need to patch ConfigRepositoryManager.__enter__ to return our mock
    with patch('ra_aid.database.repositories.config_repository.ConfigRepositoryManager.__enter__', return_value=mock_config_repository):
        # Check default behavior (use_aider=False)
        with patch.object(
            sys,
            "argv",
            ["ra-aid", "-m", "test message"],
        ):
            main()
            # Verify use_aider is set to False in the update call
            mock_config_repository.update.assert_called()
            # Get the call arguments
            call_args = mock_config_repository.update.call_args_list
            # Find the call that includes use_aider
            use_aider_found = False
            for args, _ in call_args:
                config_dict = args[0]
                if "use_aider" in config_dict and config_dict["use_aider"] is False:
                    use_aider_found = True
                    break
            assert use_aider_found, f"use_aider=False not found in update calls: {call_args}"

            # Check that file tools are enabled by default
            tool_names = [tool.name for tool in MODIFICATION_TOOLS]
            assert "file_str_replace" in tool_names
            assert "put_complete_file_contents" in tool_names
            assert "run_programming_task" not in tool_names

        # Reset mocks
        mock_config_repository.update.reset_mock()

        # Check with --use-aider flag
        with patch.object(
            sys,
            "argv",
            ["ra-aid", "-m", "test message", "--use-aider"],
        ):
            main()
            # Verify use_aider is set to True in the update call
            mock_config_repository.update.assert_called()
            # Get the call arguments
            call_args = mock_config_repository.update.call_args_list
            # Find the call that includes use_aider
            use_aider_found = False
            for args, _ in call_args:
                config_dict = args[0]
                if "use_aider" in config_dict and config_dict["use_aider"] is True:
                    use_aider_found = True
                    break
            assert use_aider_found, f"use_aider=True not found in update calls: {call_args}"

            # Check that run_programming_task is enabled
            tool_names = [tool.name for tool in MODIFICATION_TOOLS]
            assert "file_str_replace" not in tool_names
            assert "put_complete_file_contents" not in tool_names
            assert "run_programming_task" in tool_names

    # Reset to default state for other tests
    set_modification_tools(False)