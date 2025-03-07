"""Tests for CIAYN agent respecting should_exit flag."""

import pytest
from unittest.mock import Mock
from langchain_core.messages import HumanMessage, AIMessage

from ra_aid.agent_backends.ciayn_agent import CiaynAgent
from ra_aid.agent_context import agent_context, mark_should_exit, should_exit
from ra_aid.exceptions import ToolExecutionError


def test_ciayn_agent_stream_respects_should_exit():
    """Test that the CIAYN agent's stream method respects should_exit."""
    # Create mock model and tool
    mock_model = Mock()
    mock_tool = Mock()
    mock_tool.func.__name__ = "mock_tool"
    
    # Configure mock model to return a response
    mock_model.invoke.return_value = AIMessage(content="mock_tool()")
    
    # Create agent
    agent = CiaynAgent(mock_model, [mock_tool])
    
    # Set up test messages
    messages = {"messages": [HumanMessage(content="test")]}
    
    # Test stream exits when should_exit is set
    with agent_context() as ctx:
        # Set should_exit with propagation to all parents
        mark_should_exit(propagation_depth=None)
        
        # Verify should_exit is set
        assert should_exit()
        
        # Call stream - should exit immediately without calling model.invoke
        results = list(agent.stream(messages))
        
        # Verify stream exited without processing (empty results)
        assert len(results) == 0
        
        # Verify model was not called
        mock_model.invoke.assert_not_called()
        
    # Test negative case - stream should continue when should_exit is not set
    with agent_context() as ctx:
        # Verify should_exit is not set
        assert not should_exit()
        
        # Execute stream
        next(agent.stream(messages))
        
        # Verify model was called
        mock_model.invoke.assert_called_once()


def test_ciayn_agent_bundled_tools_respects_should_exit():
    """Test that the CIAYN agent respects should_exit when executing bundled tools."""
    # Create mock model and tools
    mock_model = Mock()
    mock_tool1 = Mock()
    mock_tool1.func.__name__ = "emit_key_facts"
    mock_tool1.func.return_value = "Tool 1 executed"
    mock_tool2 = Mock()
    mock_tool2.func.__name__ = "emit_key_snippet"
    mock_tool2.func.return_value = "Tool 2 executed"
    
    # Configure model to return multiple tool calls
    # Create a response with two bundled tool calls
    mock_model.invoke.return_value = AIMessage(content="""
    emit_key_facts(facts=["fact1", "fact2"])
    emit_key_snippet(snippet_info={"filepath": "test.py", "line_number": 1, "snippet": "code", "description": "test"})
    """)
    
    # Create agent with bundleable tools
    agent = CiaynAgent(mock_model, [mock_tool1, mock_tool2])
    
    # Test executing bundled tools when should_exit is set
    with agent_context() as ctx:
        # Set up test messages
        messages = {"messages": [HumanMessage(content="test")]}
        
        # Set should_exit with propagation to all parents
        mark_should_exit(propagation_depth=None)
        
        # Call stream
        generator = agent.stream(messages)
        # Stream should exit without processing due to should_exit in the stream method
        result = list(generator)
        
        # Model should not have been called
        mock_model.invoke.assert_not_called()
        
        # Tools should not have been executed
        mock_tool1.func.assert_not_called()
        mock_tool2.func.assert_not_called()


def test_ciayn_agent_single_tool_respects_should_exit():
    """Test that the CIAYN agent respects should_exit when executing a single tool."""
    # Create mock model and tool
    mock_model = Mock()
    mock_tool = Mock()
    mock_tool.func.__name__ = "test_tool"
    mock_tool.func.return_value = "Tool executed"
    
    # Configure model to return a single tool call
    mock_model.invoke.return_value = AIMessage(content="test_tool()")
    
    # Create agent
    agent = CiaynAgent(mock_model, [mock_tool])
    
    # Set up a context that manipulates should_exit during execution
    class TestContext:
        def __init__(self):
            self.should_exit_flag = False
            
        def set_exit_flag(self):
            self.should_exit_flag = True
            return "Exit flag set"
        
        def get_exit_flag(self):
            return self.should_exit_flag
    
    test_context = TestContext()
    
    # Replace the actual tool execution with one that sets should_exit
    def execute_and_exit(*args, **kwargs):
        # Set should_exit flag
        with agent_context() as ctx:
            mark_should_exit(propagation_depth=None)
        return "Tool executed"
    
    # Override the mock tool to set should_exit
    mock_tool.func.side_effect = execute_and_exit
    
    # Execute tool call that sets should_exit
    with agent_context() as ctx:
        # Verify should_exit is initially False
        assert not should_exit()
        
        # Set up test messages
        messages = {"messages": [HumanMessage(content="test")]}
        
        # Call stream
        generator = agent.stream(messages)
        # Get first response (which will execute the tool and set should_exit)
        next(generator, None)
        
        # Verify model was called
        mock_model.invoke.assert_called_once()
        
        # Verify tool was executed
        mock_tool.func.assert_called_once()
        
        # Verify should_exit was set
        assert should_exit()
        
        # Get next response (should be empty because stream should exit)
        results = list(generator)
        assert len(results) == 0
        
        # Verify model wasn't called again
        assert mock_model.invoke.call_count == 1


def test_ciayn_agent_execute_tool_respects_should_exit():
    """Test that _execute_tool respects should_exit."""
    # Create mock model and tool
    mock_model = Mock()
    mock_tool = Mock()
    mock_tool.func.__name__ = "test_tool"
    
    # Create agent
    agent = CiaynAgent(mock_model, [mock_tool])
    
    # Test _execute_tool exits when should_exit is set
    with agent_context() as ctx:
        # Set should_exit with propagation to all parents
        mark_should_exit(propagation_depth=None)
        
        # Call _execute_tool
        message = HumanMessage(content="test_tool()")
        result = agent._execute_tool(message)
        
        # Verify early exit message
        assert "agent should exit flag is set" in result
        
        # Verify tool was not executed
        mock_tool.func.assert_not_called()