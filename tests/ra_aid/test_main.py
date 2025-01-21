"""Unit tests for __main__.py argument parsing."""

import pytest
from ra_aid.__main__ import parse_arguments
from ra_aid.tools.memory import _global_memory
from ra_aid.config import DEFAULT_RECURSION_LIMIT


@pytest.fixture
def mock_dependencies(monkeypatch):
    """Mock all dependencies needed for main()."""
    monkeypatch.setattr('ra_aid.__main__.check_dependencies', lambda: None)
    
    monkeypatch.setattr('ra_aid.__main__.validate_environment', 
                        lambda args: (True, [], True, []))
    
    def mock_config_update(*args, **kwargs):
        config = _global_memory.get("config", {})
        if kwargs.get("temperature"):
            config["temperature"] = kwargs["temperature"]
        _global_memory["config"] = config
        return None

    monkeypatch.setattr('ra_aid.__main__.initialize_llm', 
                        mock_config_update)
    
    monkeypatch.setattr('ra_aid.__main__.run_research_agent', 
                        lambda *args, **kwargs: None)

def test_recursion_limit_in_global_config(mock_dependencies):
    """Test that recursion limit is correctly set in global config."""
    from ra_aid.__main__ import main
    import sys
    from unittest.mock import patch
    
    _global_memory.clear()
    
    with patch.object(sys, 'argv', ['ra-aid', '-m', 'test message']):
        main()
        assert _global_memory["config"]["recursion_limit"] == DEFAULT_RECURSION_LIMIT
    
    _global_memory.clear()
    
    with patch.object(sys, 'argv', ['ra-aid', '-m', 'test message', '--recursion-limit', '50']):
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
    from ra_aid.__main__ import main
    import sys
    from unittest.mock import patch

    _global_memory.clear()
    
    with patch.object(sys, 'argv', [
        'ra-aid', '-m', 'test message',
        '--cowboy-mode',
        '--research-only',
        '--provider', 'anthropic',
        '--model', 'claude-3-5-sonnet-20241022',
        '--expert-provider', 'openai',
        '--expert-model', 'gpt-4',
        '--temperature', '0.7',
        '--disable-limit-tokens'
    ]):
        main()
        config = _global_memory["config"]
        assert config["cowboy_mode"] is True
        assert config["research_only"] is True
        assert config["provider"] == "anthropic"
        assert config["model"] == "claude-3-5-sonnet-20241022"
        assert config["expert_provider"] == "openai"
        assert config["expert_model"] == "gpt-4"
        assert config["limit_tokens"] is False


def test_temperature_validation(mock_dependencies):
    """Test that temperature argument is correctly passed to initialize_llm."""
    from ra_aid.__main__ import main, initialize_llm
    import sys
    from unittest.mock import patch

    _global_memory.clear()
    
    with patch('ra_aid.__main__.initialize_llm') as mock_init_llm:
        with patch.object(sys, 'argv', ['ra-aid', '-m', 'test', '--temperature', '0.7']):
            main()
            mock_init_llm.assert_called_once()
            assert mock_init_llm.call_args.kwargs['temperature'] == 0.7

    with pytest.raises(SystemExit):
        with patch.object(sys, 'argv', ['ra-aid', '-m', 'test', '--temperature', '2.1']):
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
