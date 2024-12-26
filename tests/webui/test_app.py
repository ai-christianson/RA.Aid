import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
from queue import Queue
from types import SimpleNamespace

# Import the app functions
from webui.app import initialize_session_state, process_messages, render_messages, run_app

# Mock the components
mock_research = MagicMock()
mock_research.return_value = {"success": True, "research_notes": ["Test note"], "key_facts": {}}

mock_planning = MagicMock()
mock_planning.return_value = {"success": True, "plan": "Test plan", "tasks": ["Task 1"]}

mock_implementation = MagicMock()
mock_implementation.return_value = {"success": True, "implemented_tasks": ["Task 1"], "skipped_tasks": [], "failed_tasks": []}

@pytest.fixture
def mock_streamlit():
    """Fixture to mock streamlit components."""
    with patch('streamlit.title') as mock_title, \
         patch('streamlit.text_input') as mock_input, \
         patch('streamlit.button') as mock_button, \
         patch('streamlit.success') as mock_success, \
         patch('streamlit.error') as mock_error:
        yield {
            'title': mock_title,
            'text_input': mock_input,
            'button': mock_button,
            'success': mock_success,
            'error': mock_error
        }

def test_initialize_session_state():
    """Test session state initialization."""
    session_state = SimpleNamespace()
    with patch('streamlit.session_state', session_state):
        initialize_session_state()
        
        assert hasattr(session_state, 'messages')
        assert hasattr(session_state, 'task_submitted')
        assert hasattr(session_state, 'connected')
        assert isinstance(session_state.messages, list)
        assert session_state.task_submitted == False
        assert session_state.connected == False

def test_process_messages_success():
    """Test successful message processing."""
    message_queue = Queue()
    test_message = {"type": "test", "content": "Test message"}
    message_queue.put(test_message)
    
    session_state = SimpleNamespace(messages=[])
    with patch('streamlit.session_state', session_state):
        process_messages(message_queue)
        assert len(session_state.messages) == 1
        assert session_state.messages[0] == test_message

def test_process_messages_empty_queue():
    """Test message processing with empty queue."""
    message_queue = Queue()
    session_state = SimpleNamespace(messages=[])
    with patch('streamlit.session_state', session_state):
        process_messages(message_queue)
        assert len(session_state.messages) == 0

def test_render_messages():
    """Test message rendering."""
    messages = [
        {"type": "info", "content": "Info message"},
        {"type": "error", "content": "Error message"},
        {"type": "success", "content": "Success message"},
        {"type": "default", "content": "Default message"}
    ]
    
    session_state = SimpleNamespace(messages=messages)
    with patch('streamlit.session_state', session_state):
        with patch('streamlit.info') as mock_info, \
             patch('streamlit.error') as mock_error, \
             patch('streamlit.success') as mock_success, \
             patch('streamlit.write') as mock_write:
            render_messages()
            
            mock_info.assert_called_once_with("Info message")
            mock_error.assert_called_once_with("Error message")
            mock_success.assert_called_once_with("Success message")
            mock_write.assert_called_once_with("Default message")

def test_run_app_success(mock_streamlit):
    """Test successful app execution flow."""
    mock_streamlit['text_input'].return_value = "test task"
    mock_streamlit['button'].return_value = True
    
    session_state = SimpleNamespace(
        messages=[],
        task_submitted=False,
        connected=False
    )
    with patch('streamlit.session_state', session_state):
        with patch('webui.app.research_component', mock_research), \
             patch('webui.app.planning_component', mock_planning), \
             patch('webui.app.implementation_component', mock_implementation):
            run_app()
            
            mock_streamlit['title'].assert_called_once()
            mock_streamlit['text_input'].assert_called_once()
            mock_streamlit['button'].assert_called_once()
            assert mock_streamlit['success'].call_count == 3
            assert session_state.task_submitted == True

def test_run_app_no_task(mock_streamlit):
    """Test app execution with no task."""
    mock_streamlit['text_input'].return_value = ""
    mock_streamlit['button'].return_value = True
    
    session_state = SimpleNamespace(
        messages=[],
        task_submitted=False,
        connected=False
    )
    with patch('streamlit.session_state', session_state):
        run_app()
        mock_streamlit['error'].assert_called_once_with("Please enter a task or query.")

def test_run_app_research_failure(mock_streamlit):
    """Test app execution with research failure."""
    mock_streamlit['text_input'].return_value = "test task"
    mock_streamlit['button'].return_value = True
    mock_research.return_value = {"success": False}
    
    session_state = SimpleNamespace(
        messages=[],
        task_submitted=False,
        connected=False
    )
    with patch('streamlit.session_state', session_state):
        with patch('webui.app.research_component', mock_research):
            run_app()
            
            mock_streamlit['error'].assert_called_once_with("Research phase failed")
            assert session_state.task_submitted == True

def test_run_app_planning_failure(mock_streamlit):
    """Test app execution with planning failure."""
    mock_streamlit['text_input'].return_value = "test task"
    mock_streamlit['button'].return_value = True
    mock_research.return_value = {"success": True}
    mock_planning.return_value = {"success": False}
    
    session_state = SimpleNamespace(
        messages=[],
        task_submitted=False,
        connected=False
    )
    with patch('streamlit.session_state', session_state):
        with patch('webui.app.research_component', mock_research), \
             patch('webui.app.planning_component', mock_planning):
            run_app()
            
            assert mock_streamlit['success'].call_count == 1
            mock_streamlit['error'].assert_called_once_with("Planning phase failed")
            assert session_state.task_submitted == True

def test_run_app_implementation_failure(mock_streamlit):
    """Test app execution with implementation failure."""
    mock_streamlit['text_input'].return_value = "test task"
    mock_streamlit['button'].return_value = True
    mock_research.return_value = {"success": True}
    mock_planning.return_value = {"success": True}
    mock_implementation.return_value = {"success": False}
    
    session_state = SimpleNamespace(
        messages=[],
        task_submitted=False,
        connected=False
    )
    with patch('streamlit.session_state', session_state):
        with patch('webui.app.research_component', mock_research), \
             patch('webui.app.planning_component', mock_planning), \
             patch('webui.app.implementation_component', mock_implementation):
            run_app()
            
            assert mock_streamlit['success'].call_count == 2
            mock_streamlit['error'].assert_called_once_with("Implementation phase failed")
            assert session_state.task_submitted == True 