import pytest
from ra_aid.tools.programmer import parse_aider_flags

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
