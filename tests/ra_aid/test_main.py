"""Unit tests for __main__.py argument parsing."""

import pytest
from unittest.mock import patch, MagicMock

from ra_aid.__main__ import parse_arguments
from ra_aid.config import DEFAULT_RECURSION_LIMIT
from ra_aid.tools.memory import _global_memory


@pytest.fixture
def mock_dependencies(monkeypatch):
    """Mock all dependencies needed for main()."""
    # Initialize global memory with necessary keys to prevent KeyError
    _global_memory.clear()
    _global_memory["agent_depth"] = 0
    _global_memory["work_log"] = []
    _global_memory["config"] = {}
    
    # Mock dependencies that interact with external systems
    monkeypatch.setattr("ra_aid.__main__.check_dependencies", lambda: None)
    monkeypatch.setattr("ra_aid.__main__.validate_environment", lambda args: (True, [], True, []))
    monkeypatch.setattr("ra_aid.__main__.create_agent", lambda *args, **kwargs: None)
    monkeypatch.setattr("ra_aid.__main__.run_agent_with_retry", lambda *args, **kwargs: None)
    monkeypatch.setattr("ra_aid.__main__.run_research_agent", lambda *args, **kwargs: None)
    monkeypatch.setattr("ra_aid.__main__.run_planning_agent", lambda *args, **kwargs: None)
    
    # Mock LLM initialization
    def mock_config_update(*args, **kwargs):
        config = _global_memory.get("config", {})
        if kwargs.get("temperature"):
            config["temperature"] = kwargs["temperature"]
        _global_memory["config"] = config
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


def test_recursion_limit_in_global_config(mock_dependencies):
    """Test that recursion limit is correctly set in global config."""
    import sys
    from unittest.mock import patch

    from ra_aid.__main__ import main

    _global_memory.clear()

    with patch.object(sys, "argv", ["ra-aid", "-m", "test message"]):
        main()
        assert _global_memory["config"]["recursion_limit"] == DEFAULT_RECURSION_LIMIT

    _global_memory.clear()

    with patch.object(
        sys, "argv", ["ra-aid", "-m", "test message", "--recursion-limit", "50"]
    ):
        main()
        assert _global_memory["config"]["recursion_limit"] == 50


def test_negative_recursion_limit():
    """Test that negative recursion limit raises error."""
    with pytest.raises(SystemExit):
        parse_arguments(["-m", "test message", "--recursion-limit", "-1"])


def test_zero_recursion_limit():
    """Test that zero recursion limit raises error."""
    with pytest.raises(SystemExit):
        parse_arguments(["-m", "test message", "--recursion-limit", "0"])


def test_config_settings(mock_dependencies):
    """Test that various settings are correctly applied in global config."""
    import sys
    from unittest.mock import patch

    from ra_aid.__main__ import main

    _global_memory.clear()

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
        config = _global_memory["config"]
        assert config["cowboy_mode"] is True
        assert config["research_only"] is True
        assert config["provider"] == "anthropic"
        assert config["model"] == "claude-3-7-sonnet-20250219"
        assert config["expert_provider"] == "openai"
        assert config["expert_model"] == "gpt-4"
        assert config["limit_tokens"] is False


def test_temperature_validation(mock_dependencies):
    """Test that temperature argument is correctly passed to initialize_llm."""
    import sys
    from unittest.mock import patch, ANY

    from ra_aid.__main__ import main

    # Reset global memory for clean test
    _global_memory.clear()
    _global_memory["agent_depth"] = 0
    _global_memory["work_log"] = []
    _global_memory["config"] = {}

    # Test valid temperature (0.7)
    with patch("ra_aid.__main__.initialize_llm", return_value=None) as mock_init_llm:
        # Also patch any calls that would actually use the mocked initialize_llm function
        with patch("ra_aid.__main__.run_research_agent", return_value=None):
            with patch("ra_aid.__main__.run_planning_agent", return_value=None):
                with patch.object(
                    sys, "argv", ["ra-aid", "-m", "test", "--temperature", "0.7"]
                ):
                    main()
                    # Check if temperature was stored in config correctly
                    assert _global_memory["config"]["temperature"] == 0.7

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


def test_research_model_provider_args(mock_dependencies):
    """Test that research-specific model/provider args are correctly stored in config."""
    import sys
    from unittest.mock import patch

    from ra_aid.__main__ import main

    _global_memory.clear()

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
        config = _global_memory["config"]
        assert config["research_provider"] == "anthropic"
        assert config["research_model"] == "claude-3-haiku-20240307"
        assert config["planner_provider"] == "openai"
        assert config["planner_model"] == "gpt-4"


def test_planner_model_provider_args(mock_dependencies):
    """Test that planner provider/model args fall back to main config when not specified."""
    import sys
    from unittest.mock import patch

    from ra_aid.__main__ import main

    _global_memory.clear()

    with patch.object(
        sys,
        "argv",
        ["ra-aid", "-m", "test message", "--provider", "openai", "--model", "gpt-4"],
    ):
        main()
        config = _global_memory["config"]
        assert config["planner_provider"] == "openai"
        assert config["planner_model"] == "gpt-4"


def test_use_aider_flag(mock_dependencies):
    """Test that use-aider flag is correctly stored in config."""
    import sys
    from unittest.mock import patch

    from ra_aid.__main__ import main
    from ra_aid.tool_configs import MODIFICATION_TOOLS, set_modification_tools

    _global_memory.clear()

    # Reset to default state
    set_modification_tools(False)

    # Check default behavior (use_aider=False)
    with patch.object(
        sys,
        "argv",
        ["ra-aid", "-m", "test message"],
    ):
        main()
        config = _global_memory["config"]
        assert config.get("use_aider") is False

        # Check that file tools are enabled by default
        tool_names = [tool.name for tool in MODIFICATION_TOOLS]
        assert "file_str_replace" in tool_names
        assert "put_complete_file_contents" in tool_names
        assert "run_programming_task" not in tool_names

    _global_memory.clear()

    # Check with --use-aider flag
    with patch.object(
        sys,
        "argv",
        ["ra-aid", "-m", "test message", "--use-aider"],
    ):
        main()
        config = _global_memory["config"]
        assert config.get("use_aider") is True

        # Check that run_programming_task is enabled
        tool_names = [tool.name for tool in MODIFICATION_TOOLS]
        assert "file_str_replace" not in tool_names
        assert "put_complete_file_contents" not in tool_names
        assert "run_programming_task" in tool_names

    # Reset to default state for other tests
    set_modification_tools(False)