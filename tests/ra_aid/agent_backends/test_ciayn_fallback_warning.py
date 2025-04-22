import pytest
from unittest.mock import patch, Mock
from langchain_core.messages import HumanMessage

from ra_aid.agent_backends.ciayn_agent import CiaynAgent
from ra_aid.exceptions import ToolExecutionError


def _make_error():
    return ToolExecutionError(
        "dummy failure",
        base_message=HumanMessage("dummy"),
        tool_name="dummy_tool",
    )


def test_no_fallback_warning_when_disabled():
    agent = CiaynAgent(
        Mock(),              # dummy model
        [],                  # no tools
        config={"experimental_fallback_handler": False},
    )
    err = _make_error()

    with patch("ra_aid.console.formatting.print_warning") as mock_warn:
        agent.handle_fallback_response(None, err)
        mock_warn.assert_not_called()


def test_fallback_warning_when_enabled_and_failed():
    agent = CiaynAgent(
        Mock(),
        [],
        config={"experimental_fallback_handler": True},
    )
    err = _make_error()

    with patch("ra_aid.console.formatting.print_warning") as mock_warn:
        agent.handle_fallback_response(None, err)
        mock_warn.assert_called_once()
