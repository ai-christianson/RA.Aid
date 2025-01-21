import pytest
from ra_aid.tools.programmer import parse_aider_flags, run_programming_task

# Test cases for parse_aider_flags function
test_cases = [
    # Test case format: (input_string, expected_output, test_description)
    (
        "yes-always,dark-mode",
        ["--yes-always", "--dark-mode"],
        "basic comma separated flags without dashes"
    ),
    (
        "--yes-always,--dark-mode",
        ["--yes-always", "--dark-mode"],
        "comma separated flags with dashes"
    ),
    (
        "yes-always, dark-mode",
        ["--yes-always", "--dark-mode"],
        "comma separated flags with space"
    ),
    (
        "--yes-always, --dark-mode",
        ["--yes-always", "--dark-mode"],
        "comma separated flags with dashes and space"
    ),
    (
        "",
        [],
        "empty string"
    ),
    (
        "  yes-always  ,  dark-mode  ",
        ["--yes-always", "--dark-mode"],
        "flags with extra whitespace"
    ),
    (
        "--yes-always",
        ["--yes-always"],
        "single flag with dashes"
    ),
    (
        "yes-always",
        ["--yes-always"],
        "single flag without dashes"
    )
]

@pytest.mark.parametrize("input_flags,expected,description", test_cases)
def test_parse_aider_flags(input_flags, expected, description):
    """Table-driven test for parse_aider_flags function."""
    result = parse_aider_flags(input_flags)
    assert result == expected, f"Failed test case: {description}"

def test_aider_config_flag(mocker):
    """Test that aider config flag is properly included in the command when specified."""
    mock_memory = {
        'config': {'aider_config': '/path/to/config.yml'},
        'related_files': {}
    }
    mocker.patch('ra_aid.tools.programmer._global_memory', mock_memory)
    
    # Mock the run_interactive_command to capture the command that would be run
    mock_run = mocker.patch('ra_aid.tools.programmer.run_interactive_command', return_value=(b'', 0))
    
    run_programming_task("test instruction")
    
    args = mock_run.call_args[0][0]  # Get the first positional arg (command list)
    assert '--config' in args
    config_index = args.index('--config')
    assert args[config_index + 1] == '/path/to/config.yml'
