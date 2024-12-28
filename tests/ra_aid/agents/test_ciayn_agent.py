import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, AIMessage
from ra_aid.agents.ciayn_agent import CiaynAgent

@pytest.fixture
def mock_model():
    """Create a mock language model."""
    model = Mock()
    model.invoke = Mock()
    return model

@pytest.fixture
def agent(mock_model):
    """Create a CiaynAgent instance with mock model."""
    tools = []  # Empty tools list for testing trimming functionality
    return CiaynAgent(mock_model, tools, max_history_messages=3)

def test_trim_chat_history_preserves_initial_messages(agent):
    """Test that initial messages are preserved during trimming."""
    initial_messages = [
        HumanMessage(content="Initial 1"),
        AIMessage(content="Initial 2")
    ]
    chat_history = [
        HumanMessage(content="Chat 1"),
        AIMessage(content="Chat 2"),
        HumanMessage(content="Chat 3"),
        AIMessage(content="Chat 4")
    ]
    
    result = agent._trim_chat_history(initial_messages, chat_history)
    
    # Verify initial messages are preserved
    assert result[:2] == initial_messages
    # Verify only last 3 chat messages are kept (due to max_history_messages=3)
    assert len(result[2:]) == 3
    assert result[2:] == chat_history[-3:]

def test_trim_chat_history_under_limit(agent):
    """Test trimming when chat history is under the maximum limit."""
    initial_messages = [HumanMessage(content="Initial")]
    chat_history = [
        HumanMessage(content="Chat 1"),
        AIMessage(content="Chat 2")
    ]
    
    result = agent._trim_chat_history(initial_messages, chat_history)
    
    # Verify no trimming occurred
    assert len(result) == 3
    assert result == initial_messages + chat_history

def test_trim_chat_history_over_limit(agent):
    """Test trimming when chat history exceeds the maximum limit."""
    initial_messages = [HumanMessage(content="Initial")]
    chat_history = [
        HumanMessage(content="Chat 1"),
        AIMessage(content="Chat 2"),
        HumanMessage(content="Chat 3"),
        AIMessage(content="Chat 4"),
        HumanMessage(content="Chat 5")
    ]
    
    result = agent._trim_chat_history(initial_messages, chat_history)
    
    # Verify correct trimming
    assert len(result) == 4  # initial + max_history_messages
    assert result[0] == initial_messages[0]  # Initial message preserved
    assert result[1:] == chat_history[-3:]  # Last 3 chat messages kept

def test_trim_chat_history_empty_initial(agent):
    """Test trimming with empty initial messages."""
    initial_messages = []
    chat_history = [
        HumanMessage(content="Chat 1"),
        AIMessage(content="Chat 2"),
        HumanMessage(content="Chat 3"),
        AIMessage(content="Chat 4")
    ]
    
    result = agent._trim_chat_history(initial_messages, chat_history)
    
    # Verify only last 3 messages are kept
    assert len(result) == 3
    assert result == chat_history[-3:]

def test_trim_chat_history_empty_chat(agent):
    """Test trimming with empty chat history."""
    initial_messages = [
        HumanMessage(content="Initial 1"),
        AIMessage(content="Initial 2")
    ]
    chat_history = []
    
    result = agent._trim_chat_history(initial_messages, chat_history)
    
    # Verify initial messages are preserved and no trimming occurred
    assert result == initial_messages
    assert len(result) == 2
