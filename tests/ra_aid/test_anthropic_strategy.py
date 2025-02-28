import os
from dataclasses import dataclass
from typing import Optional

import pytest

from ra_aid.provider_strategy import AnthropicStrategy


@pytest.fixture
def clean_env():
    """Remove relevant environment variables before each test."""
    # Save existing values
    saved_vars = {}
    for var in [
        "ANTHROPIC_API_KEY",
        "EXPERT_ANTHROPIC_API_KEY",
        "ANTHROPIC_MODEL",
        "EXPERT_ANTHROPIC_MODEL",
    ]:
        saved_vars[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]

    yield

    # Restore saved values
    for var, value in saved_vars.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]


@dataclass
class MockArgs:
    """Mock arguments class for testing."""

    expert_provider: str
    expert_model: Optional[str] = None


def test_anthropic_expert_validation_message(clean_env):
    """Test that validation message refers to base key when neither key exists."""
    strategy = AnthropicStrategy()
    args = MockArgs(expert_provider="anthropic")

    result = strategy.validate(args)

    assert not result.valid
    assert len(result.missing_vars) > 0
    assert "ANTHROPIC_API_KEY environment variable is not set" in result.missing_vars[0]
    assert "EXPERT_ANTHROPIC_API_KEY" not in result.missing_vars[0]
