import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(autouse=True)
def clear_session_state():
    """Clear session state before each test"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]

def test_initialize_session_state():
    """Test session state initialization"""
    # Import here to avoid circular imports
    from webui.app import initialize_session_state
    
    # Initialize session state
    initialize_session_state()
    
    # Check if messages are initialized
    assert 'messages' in st.session_state
    assert len(st.session_state.messages) == 1
    assert st.session_state.messages[0]['role'] == 'assistant'
    
    # Check if models are initialized
    assert 'models' in st.session_state
    assert 'anthropic' in st.session_state.models
    assert 'openai' in st.session_state.models

def test_handle_task_shell_command():
    """Test handling shell commands"""
    # Import here to avoid circular imports
    from webui.app import handle_task, initialize_session_state
    
    # Mock configuration
    config = {
        "provider": "anthropic",
        "model": "claude-2.1",
        "research_only": True,
        "cowboy_mode": True,
        "hil": False,
        "web_research_enabled": False
    }
    
    # Initialize session state
    initialize_session_state()
    
    # Test shell command
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(
            stdout="test output",
            stderr="",
            returncode=0
        )
        
        handle_task("run pwd", config)
        
        # Check if command was executed
        mock_run.assert_called_once_with(
            "pwd",
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Check if output was added to messages
        assert len(st.session_state.messages) == 3  # Welcome message + user command + response
        assert "test output" in st.session_state.messages[-1]['content']

def test_handle_task_llm():
    """Test handling LLM tasks"""
    # Import here to avoid circular imports
    from webui.app import handle_task, initialize_session_state
    
    # Mock configuration
    config = {
        "provider": "anthropic",
        "model": "claude-2.1",
        "research_only": True,
        "cowboy_mode": True,
        "hil": False,
        "web_research_enabled": False
    }
    
    # Initialize session state
    initialize_session_state()
    
    # Mock litellm response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="Research Notes:\n- Note 1\n- Note 2\n\nKey Facts:\nFact 1: Value 1\nFact 2: Value 2"
            )
        )
    ]
    
    # Test LLM task
    with patch('webui.app.completion', return_value=mock_response) as mock_completion:
        handle_task("What is Python?", config)
        
        # Verify litellm was called correctly
        mock_completion.assert_called_once_with(
            model="claude-2.1",
            messages=[{"role": "user", "content": "What is Python?"}],
            temperature=0.7,
            max_tokens=2000
        )
        
        # Check if response was added to messages
        assert len(st.session_state.messages) == 3  # Welcome message + user query + response
        assert "Research Notes" in st.session_state.messages[-1]['content']
        assert "Key Facts" in st.session_state.messages[-1]['content'] 