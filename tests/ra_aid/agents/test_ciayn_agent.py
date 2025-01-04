"""Tests for CiaynAgent."""

import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from ra_aid.exceptions import ToolExecutionError
from ra_aid.agents.ciayn_agent import CiaynAgent, ChunkMessage

@pytest.fixture
def mock_model():
    """Create a mock LLM model."""
    model = Mock()
    model.invoke.return_value = AIMessage(content="output_markdown_message(\"Test message\")")
    return model

@pytest.fixture
def test_tools():
    """Create test tools for the agent."""
    @tool
    def test_tool_1(arg: str) -> str:
        """A test tool that returns its argument."""
        return arg

    @tool
    def test_tool_2(arg1: str, arg2: str) -> str:
        """A test tool that combines its arguments."""
        return f"{arg1} {arg2}"

    return [test_tool_1, test_tool_2]

@pytest.fixture
def agent(mock_model, test_tools):
    """Create a CiaynAgent instance."""
    return CiaynAgent(mock_model, test_tools)

def test_agent_initialization(agent, test_tools):
    """Test that agent is initialized with correct tools."""
    assert len(agent.tools) == len(test_tools)
    assert all(hasattr(tool.func, '__name__') for tool in agent.tools)

def test_agent_tool_execution_success(agent):
    """Test successful tool execution."""
    code = 'test_tool_1("hello")'
    result = agent._execute_tool(code)
    assert result == "hello"

def test_agent_tool_execution_error(agent):
    """Test tool execution error handling."""
    code = 'invalid_tool("test")'
    with pytest.raises(ToolExecutionError) as exc_info:
        agent._execute_tool(code)
    assert "Error executing tool call" in str(exc_info.value)

def test_agent_tool_execution_validation(agent):
    """Test tool execution input validation."""
    # Test invalid code format
    with pytest.raises(ToolExecutionError) as exc_info:
        agent._execute_tool("print('hello')")  # Not a tool call
    assert "Invalid tool call format" in str(exc_info.value)

    # Test missing tool
    with pytest.raises(ToolExecutionError) as exc_info:
        agent._execute_tool("missing_tool()")
    assert "Invalid tool call format" in str(exc_info.value)

def test_agent_stream_success(agent):
    """Test successful streaming of agent responses."""
    messages = {"messages": [HumanMessage(content="Test query")]}
    chunks = list(agent.stream(messages))
    
    # Should yield empty dict on success
    assert len(chunks) == 1
    assert chunks[0] == {}

def test_agent_stream_error(agent):
    """Test error handling in stream method."""
    # Mock a tool execution error
    agent.model.invoke.return_value = AIMessage(content="invalid_tool()")
    
    messages = {"messages": [HumanMessage(content="Test query")]}
    chunks = list(agent.stream(messages))
    
    # Should yield error chunk
    assert len(chunks) == 1
    assert "tools" in chunks[0]
    assert chunks[0]["tools"]["messages"][0].status == "error"

def test_agent_chat_history_management(agent):
    """Test chat history management."""
    messages = {"messages": [HumanMessage(content="Test query")]}
    
    # Run multiple iterations
    list(agent.stream(messages))
    list(agent.stream(messages))
    
    # Verify system message is included
    system_messages = [
        msg for msg in agent.model.invoke.call_args[0][0]
        if isinstance(msg, SystemMessage)
    ]
    assert len(system_messages) == 1
    assert "Execute efficiently" in system_messages[0].content

def test_agent_token_management(agent):
    """Test token management in chat history."""
    # Create agent with small token limit
    agent = CiaynAgent(mock_model, test_tools, max_tokens=100)
    
    # Add many messages
    messages = {
        "messages": [
            HumanMessage(content="A" * 1000),  # Long message
            HumanMessage(content="B" * 1000),  # Long message
        ]
    }
    
    # Stream should work without errors
    list(agent.stream(messages))
    
    # Verify history was trimmed
    assert len(agent.model.invoke.call_args[0][0]) < 5  # Small number of messages

def test_agent_error_chunk_format(agent):
    """Test error chunk format."""
    error_chunk = agent._create_error_chunk("Test error")
    
    assert "tools" in error_chunk
    assert isinstance(error_chunk["tools"]["messages"][0], ChunkMessage)
    assert error_chunk["tools"]["messages"][0].status == "error"
    assert error_chunk["tools"]["messages"][0].content == "Test error"

def test_agent_webui_integration(agent):
    """Test integration with WebUI configuration and streaming."""
    # Mock WebUI config
    config = {
        "provider": "openai",
        "model": "gpt-4",
        "web_research_enabled": True,
        "hil": True
    }
    
    messages = {
        "messages": [HumanMessage(content="Test query")],
        "config": config
    }
    
    # Test streaming with WebUI config
    chunks = list(agent.stream(messages))
    assert len(chunks) >= 1
    
    # Verify config was properly applied
    call_args = agent.model.invoke.call_args[0][0]
    assert any(isinstance(msg, SystemMessage) for msg in call_args)

def test_agent_tool_streaming(agent):
    """Test streaming of tool execution results."""
    # Mock a tool execution response
    agent.model.invoke.return_value = AIMessage(content='test_tool_1("hello")')
    
    messages = {"messages": [HumanMessage(content="Test query")]}
    chunks = list(agent.stream(messages))
    
    # Should yield tool execution result
    assert len(chunks) >= 1
    if "tools" in chunks[0]:
        assert chunks[0]["tools"]["messages"][0].status == "success"
        assert "hello" in chunks[0]["tools"]["messages"][0].content

def test_agent_concurrent_tool_execution(agent):
    """Test handling of concurrent tool executions."""
    import asyncio
    import threading
    
    def run_agent():
        messages = {"messages": [HumanMessage(content="Test query")]}
        return list(agent.stream(messages))
    
    # Run multiple agent instances concurrently
    threads = []
    results = []
    
    for _ in range(3):
        thread = threading.Thread(target=lambda: results.append(run_agent()))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    # Verify all executions completed successfully
    assert len(results) == 3
    for chunks in results:
        assert len(chunks) >= 1

def test_agent_specific_scenarios(agent):
    """Test specific edge cases and scenarios."""
    # Test empty message handling
    empty_messages = {"messages": []}
    empty_chunks = list(agent.stream(empty_messages))
    assert len(empty_chunks) == 1
    
    # Test system message override
    system_messages = {
        "messages": [
            SystemMessage(content="Custom system message"),
            HumanMessage(content="Test query")
        ]
    }
    system_chunks = list(agent.stream(system_messages))
    assert len(system_chunks) >= 1
    
    # Test multiple tool calls in one response
    agent.model.invoke.return_value = AIMessage(
        content='test_tool_1("hello")\ntest_tool_2("world", "!")'
    )
    multi_chunks = list(agent.stream({"messages": [HumanMessage(content="Test query")]}))
    assert len(multi_chunks) >= 1

def test_agent_error_recovery(agent):
    """Test agent's ability to recover from errors."""
    # First call fails, second succeeds
    agent.model.invoke.side_effect = [
        AIMessage(content='invalid_tool()'),
        AIMessage(content='test_tool_1("recovery")')
    ]
    
    messages = {"messages": [HumanMessage(content="Test query")]}
    chunks = list(agent.stream(messages))
    
    # Should have both error and recovery chunks
    assert len(chunks) >= 2
    assert any(
        chunk.get("tools", {}).get("messages", [{}])[0].get("status") == "error"
        for chunk in chunks
    )
    assert any(
        chunk.get("tools", {}).get("messages", [{}])[0].get("status") == "success"
        for chunk in chunks
    )

def test_agent_memory_management(agent):
    """Test agent's memory management during streaming."""
    # Create a sequence of messages that exceeds token limit
    long_messages = {
        "messages": [
            HumanMessage(content="A" * 1000),
            AIMessage(content="B" * 1000),
            HumanMessage(content="C" * 1000),
            AIMessage(content="D" * 1000)
        ]
    }
    
    # Run multiple iterations to test memory cleanup
    for _ in range(3):
        chunks = list(agent.stream(long_messages))
        assert len(chunks) >= 1
    
    # Verify memory was properly managed
    assert len(agent.model.invoke.call_args[0][0]) < len(long_messages["messages"])
