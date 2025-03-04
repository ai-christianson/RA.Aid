"""
Tests for the bundled tool calls functionality in the CIAYN agent.
"""
import ast
import pytest
from unittest.mock import MagicMock, patch

from ra_aid.agent_backends.ciayn_agent import CiaynAgent


def test_detect_multiple_tool_calls_single():
    """Test that a single tool call is correctly recognized as a single item."""
    # Setup
    agent = CiaynAgent(
        model=MagicMock(),
        tools=[],
    )
    code = 'ask_expert("What is the meaning of life?")'
    
    # Execute
    result = agent._detect_multiple_tool_calls(code)
    
    # Assert
    assert len(result) == 1
    assert result[0] == code


def test_detect_multiple_tool_calls_bundleable():
    """Test that multiple bundleable tool calls are correctly split."""
    # Setup
    agent = CiaynAgent(
        model=MagicMock(),
        tools=[],
    )
    code = '''emit_expert_context("Important context")
ask_expert("What does this mean?")'''
    
    # Execute
    result = agent._detect_multiple_tool_calls(code)
    
    # Assert
    assert len(result) == 2
    assert "emit_expert_context" in result[0]
    assert "ask_expert" in result[1]


def test_detect_multiple_tool_calls_non_bundleable():
    """Test that multiple tool calls with non-bundleable tools are returned as-is."""
    # Setup
    agent = CiaynAgent(
        model=MagicMock(),
        tools=[],
    )
    # Include one non-bundleable tool
    code = '''emit_expert_context("Important context")
list_directory("path/to/dir")'''
    
    # Execute
    result = agent._detect_multiple_tool_calls(code)
    
    # Assert
    # Should return the original code since list_directory is not bundleable
    assert len(result) == 1
    assert "emit_expert_context" in result[0]
    assert "list_directory" in result[0]


def test_detect_multiple_tool_calls_invalid_syntax():
    """Test that invalid syntax does not break the detection."""
    # Setup
    agent = CiaynAgent(
        model=MagicMock(),
        tools=[],
    )
    code = 'emit_expert_context("Unclosed string'
    
    # Execute
    result = agent._detect_multiple_tool_calls(code)
    
    # Assert
    assert len(result) == 1
    assert result[0] == code


def test_execute_tool_bundled():
    """Test executing a bundled tool call."""
    # Setup mock tools
    mock_emit_expert_context = MagicMock()
    mock_emit_expert_context.__name__ = "emit_expert_context"
    mock_emit_expert_context.return_value = "Context emitted"
    
    mock_ask_expert = MagicMock()
    mock_ask_expert.__name__ = "ask_expert"
    mock_ask_expert.return_value = "Expert answer"
    
    # Create tool mocks with proper function references
    emit_tool = MagicMock()
    emit_tool.func = mock_emit_expert_context
    
    ask_tool = MagicMock()
    ask_tool.func = mock_ask_expert
    
    mock_tools = [emit_tool, ask_tool]
    
    # Mock get_function_info to avoid needing real function inspection
    with patch("ra_aid.tools.reflection.get_function_info", return_value="mock function info"):
        agent = CiaynAgent(
            model=MagicMock(),
            tools=mock_tools,
        )
    
    code = '''emit_expert_context("Important context")
ask_expert("What does this mean?")'''
    
    mock_message = MagicMock()
    mock_message.content = code
    
    # Mock validate_function_call_pattern to pass validation
    with patch("ra_aid.agent_backends.ciayn_agent.validate_function_call_pattern", return_value=False):
        # Execute
        result = agent._execute_tool(mock_message)
    
    # Assert: We now verify that the result contains both tool call results with tagging
    assert "<result-" in result  # Check for result tag start
    assert "</result-" in result  # Check for result tag end
    assert "Context emitted" in result  # Check first result content
    assert "Expert answer" in result  # Check second result content
    # Verify the correct function calls were made with the right parameters
    mock_emit_expert_context.assert_called_once_with("Important context")
    mock_ask_expert.assert_called_once_with("What does this mean?")


def test_execute_tool_bundled_with_validation():
    """Test executing a bundled tool call with validation needed."""
    # Setup mock tools
    mock_emit_key_facts = MagicMock()
    mock_emit_key_facts.__name__ = "emit_key_facts"
    mock_emit_key_facts.return_value = "Facts emitted"
    
    mock_emit_key_snippet = MagicMock()
    mock_emit_key_snippet.__name__ = "emit_key_snippet"
    mock_emit_key_snippet.return_value = "Snippet emitted"
    
    # Create tool mocks with proper function references
    facts_tool = MagicMock()
    facts_tool.func = mock_emit_key_facts
    
    snippet_tool = MagicMock()
    snippet_tool.func = mock_emit_key_snippet
    
    mock_tools = [facts_tool, snippet_tool]
    
    # Mock get_function_info to avoid needing real function inspection
    with patch("ra_aid.tools.reflection.get_function_info", return_value="mock function info"):
        agent = CiaynAgent(
            model=MagicMock(),
            tools=mock_tools,
        )
    
    # Intentionally malformed calls that would require validation
    code = '''emit_key_facts(["Fact 1", "Fact 2",])
emit_key_snippet({"file": "example.py", "start_line": 10, "end_line": 20})'''
    
    mock_message = MagicMock()
    mock_message.content = code
    
    # Mock the validation and extraction
    with patch("ra_aid.agent_backends.ciayn_agent.validate_function_call_pattern") as mock_validate:
        # Setup validation to pass for the second call, fail for the first
        mock_validate.side_effect = [True, False]
        
        # Mock extract_tool_call to return a fixed version of the tool call
        with patch.object(agent, "_extract_tool_call", return_value='emit_key_facts(["Fact 1", "Fact 2"])'):
            # Execute
            result = agent._execute_tool(mock_message)
    
    # Assert: Verify both tool results are included in the tagged output
    assert "<result-" in result  # Check for result tag start
    assert "</result-" in result  # Check for result tag end
    assert "Facts emitted" in result  # Check first result content
    assert "Snippet emitted" in result  # Check second result content
