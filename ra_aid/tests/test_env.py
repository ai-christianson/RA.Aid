"""Unit tests for environment validation."""

from dataclasses import dataclass

import pytest

from ra_aid.env import validate_environment


@dataclass
class MockArgs:
    """Mock arguments for testing."""

    research_only: bool
    provider: str
    expert_provider: str = None


TEST_CASES = [
    pytest.param(
        "research_only_no_model",
        MockArgs(research_only=True, provider="openai"),
        (False, [], False, ["TAVILY_API_KEY environment variable is not set"]),
        {},
        id="research_only_no_model",
    ),
    pytest.param(
        "research_only_with_model",
        MockArgs(research_only=True, provider="openai"),
        (False, [], True, []),
        {"TAVILY_API_KEY": "test_key"},
        id="research_only_with_model",
    ),
]


@pytest.mark.parametrize("test_name,args,expected,env_vars", TEST_CASES)
def test_validate_environment_research_only(
    test_name: str, args: MockArgs, expected: tuple, env_vars: dict, monkeypatch
):
    """Test validate_environment with research_only flag."""
    # Clear any existing environment variables
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    # Set test environment variables
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    result = validate_environment(args)
    assert result == expected, f"Failed test case: {test_name}"
