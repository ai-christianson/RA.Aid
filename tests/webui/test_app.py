import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import streamlit as st
from queue import Queue, Empty

# Mock the components instead of importing them
with patch('components.memory._global_memory', {}) as mock_global_memory:
    from webui.app import (
        initialize_session_state, 
        process_message_queue, 
        render_messages, 
        main,
        load_available_models,
        render_environment_status,
        send_task,
        websocket_thread,
        handle_output
    )

# Mock the components
mock_research = MagicMock()
mock_research.return_value = {"success": True, "research_notes": ["Test note"], "key_facts": {}}

mock_planning = MagicMock()
mock_planning.return_value = {"success": True, "plan": "Test plan", "tasks": ["Task 1"]}

mock_implementation = MagicMock()
mock_implementation.return_value = {"success": True, "implemented_tasks": ["Task 1"]}

@pytest.fixture
def mock_memory():
    """Mock the global memory dictionary."""
    with patch('components.memory._global_memory', {}) as mock_mem:
        yield mock_mem

@pytest.fixture(autouse=True)
def reset_mock_global_memory():
    """Reset mock_global_memory before each test."""
    mock_global_memory['config'] = {
        'provider': 'anthropic',
        'model': 'claude-3',
        'research_only': False,
        'cowboy_mode': False,
        'hil': False,
        'web_research_enabled': False
    }
    mock_global_memory['research_notes'] = []
    mock_global_memory['plans'] = []
    mock_global_memory['tasks'] = {}
    mock_global_memory['key_facts'] = {}
    mock_global_memory['key_snippets'] = {}
    mock_global_memory['related_files'] = {}
    yield

@pytest.fixture(autouse=True)
def mock_imports():
    """Mock all external imports that might cause issues."""
    mock_research.reset_mock()  # Reset mock before each test
    mock_planning.reset_mock()
    mock_implementation.reset_mock()
    
    with patch('components.memory._global_memory', mock_global_memory), \
         patch('components.research.research_component', mock_research), \
         patch('components.planning.planning_component', mock_planning), \
         patch('components.implementation.implementation_component', mock_implementation), \
         patch('webui.app.research_component', mock_research), \
         patch('webui.app.planning_component', mock_planning), \
         patch('webui.app.implementation_component', mock_implementation), \
         patch('webui.app.message_queue', Queue()):
        yield

@pytest.fixture
def mock_session_state():
    """Fixture to mock streamlit.session_state as a dictionary."""
    with patch.dict('streamlit.session_state', {}, clear=True) as mock_state:
        yield mock_state

@pytest.fixture
def mock_streamlit():
    """Fixture to mock streamlit components."""
    with patch('streamlit.title') as mock_title, \
         patch('streamlit.text_input') as mock_input, \
         patch('streamlit.button') as mock_button, \
         patch('streamlit.success') as mock_success, \
         patch('streamlit.error') as mock_error, \
         patch('streamlit.sidebar.checkbox') as mock_checkbox, \
         patch('streamlit.sidebar.radio') as mock_radio, \
         patch('streamlit.sidebar.selectbox') as mock_selectbox, \
         patch('streamlit.text_area') as mock_text_area, \
         patch('streamlit.spinner') as mock_spinner, \
         patch('streamlit.write') as mock_write, \
         patch('streamlit.set_page_config') as mock_page_config:
        yield {
            'title': mock_title,
            'text_input': mock_input,
            'button': mock_button,
            'success': mock_success,
            'error': mock_error,
            'checkbox': mock_checkbox,
            'radio': mock_radio,
            'selectbox': mock_selectbox,
            'text_area': mock_text_area,
            'spinner': mock_spinner,
            'write': mock_write,
            'page_config': mock_page_config
        }

@pytest.fixture
def mock_env_vars():
    """Mock environment variables."""
    with patch.dict('os.environ', {
        'ANTHROPIC_API_KEY': 'test_key',
        'OPENAI_API_KEY': 'test_key',
        'TAVILY_API_KEY': 'test_key'
    }):
        yield

def test_initialize_session_state(mock_session_state):
    """Test session state initialization."""
    initialize_session_state()
    
    assert 'messages' in st.session_state
    assert 'connected' in st.session_state
    assert 'models' in st.session_state
    assert 'websocket_thread_started' in st.session_state
    assert isinstance(st.session_state.messages, list)
    assert st.session_state.connected == False
    assert st.session_state.websocket_thread_started == False
    assert isinstance(st.session_state.models, dict)

def test_process_message_queue(mock_session_state):
    """Test message queue processing."""
    st.session_state.messages = []
    with patch('webui.app.message_queue') as mock_queue:
        test_message = {"type": "test", "content": "Test message"}
        mock_queue.get_nowait.side_effect = [test_message, Empty()]
        
        process_message_queue()
        assert len(st.session_state.messages) == 1
        assert st.session_state.messages[0] == test_message

def test_process_message_queue_error(mock_session_state):
    """Test error message processing."""
    st.session_state.messages = []
    with patch('webui.app.message_queue') as mock_queue:
        error_message = {"type": "error", "content": "Error message"}
        mock_queue.get_nowait.side_effect = [error_message, Empty()]
        
        process_message_queue()
        assert len(st.session_state.messages) == 1
        assert st.session_state.messages[0] == error_message

def test_process_message_queue_empty(mock_session_state):
    """Test processing an empty message queue."""
    st.session_state.messages = []
    process_message_queue()
    assert len(st.session_state.messages) == 0

def test_render_messages(mock_session_state):
    """Test message rendering."""
    messages = [
        {"type": "error", "content": "Error message"},
        {"content": "Regular message"}
    ]
    
    st.session_state.messages = messages
    with patch('streamlit.error') as mock_error, \
         patch('streamlit.write') as mock_write:
        render_messages()
        
        mock_error.assert_called_once_with("Error message")
        mock_write.assert_called_once_with("Regular message")

def test_send_task_success(mock_streamlit, mock_session_state):
    """Test successful task sending."""
    st.session_state.connected = True
    with patch('webui.socket_interface.SocketInterface.send_task', new_callable=AsyncMock) as mock_send:
        mock_send.return_value = True
        send_task("test task", {"test": "config"})
        mock_streamlit['success'].assert_called_once()

def test_send_task_not_connected(mock_streamlit, mock_session_state):
    """Test task sending when not connected."""
    st.session_state.connected = False
    send_task("test task", {"test": "config"})
    mock_streamlit['error'].assert_called_once_with("Not connected to server")

def test_send_task_failure(mock_streamlit, mock_session_state):
    """Test task sending when it fails."""
    st.session_state.connected = True
    with patch('webui.socket_interface.SocketInterface.send_task', new_callable=AsyncMock) as mock_send:
        mock_send.side_effect = Exception("Test error")
        send_task("test task", {"test": "config"})
        mock_streamlit['error'].assert_called_once_with("Failed to send task: Test error")

def test_main_success(mock_streamlit, mock_env_vars, mock_session_state, mock_memory):
    """Test successful main flow."""
    # Set up initial state
    mock_streamlit['text_area'].return_value = "test task"
    mock_streamlit['radio'].return_value = "Full Development"
    mock_streamlit['selectbox'].return_value = "anthropic"
    mock_streamlit['checkbox'].side_effect = [True, True, True]  # cowboy_mode, hil_mode, web_research
    mock_streamlit['text_input'].return_value = "test_key"
    
    st.session_state.messages = []
    st.session_state.connected = True
    st.session_state.models = {'anthropic': ['claude-3'], 'openai': ['gpt-4']}
    st.session_state.websocket_thread_started = True
    
    # Create a new mock for research component to ensure clean state
    research_mock = MagicMock()
    research_mock.return_value = {"success": True, "research_notes": ["Test note"], "key_facts": {}}
    
    with patch('webui.app.research_component', research_mock), \
         patch('webui.app.planning_component', mock_planning), \
         patch('webui.app.implementation_component', mock_implementation), \
         patch('webui.app.load_environment_status') as mock_env_status:
        
        mock_env_status.return_value = {
            'anthropic': True,
            'openai': True,
            'web_research_enabled': True
        }
        
        # First call to set up the UI
        mock_streamlit['button'].return_value = False
        main()
        
        # Second call to simulate clicking the Start button
        mock_streamlit['button'].return_value = True
        main()
        
        # Verify configuration was set correctly
        assert mock_memory['config']['cowboy_mode'] == True
        assert mock_memory['config']['hil'] == True
        assert mock_memory['config']['web_research_enabled'] == True
        
        # Verify components were called
        research_mock.assert_called_once()
        mock_planning.assert_called_once()
        mock_implementation.assert_called_once()
        mock_streamlit['spinner'].assert_called()

def test_main_no_task(mock_streamlit, mock_env_vars, mock_session_state):
    """Test main flow with no task."""
    mock_streamlit['text_area'].return_value = ""
    mock_streamlit['button'].return_value = True
    
    st.session_state.messages = []
    st.session_state.connected = True
    st.session_state.models = {'anthropic': ['claude-3'], 'openai': ['gpt-4']}
    st.session_state.websocket_thread_started = True
    
    main()
    mock_streamlit['error'].assert_called_once_with("Please enter a valid task or query.")

def test_main_research_only(mock_streamlit, mock_env_vars, mock_session_state, mock_memory):
    """Test main flow in research only mode."""
    # Set up initial state
    mock_streamlit['text_area'].return_value = "test task"
    mock_streamlit['radio'].return_value = "Research Only"
    mock_streamlit['selectbox'].return_value = "anthropic"
    mock_streamlit['checkbox'].side_effect = [True, True, True]  # cowboy_mode, hil_mode, web_research
    mock_streamlit['text_input'].return_value = "test_key"
    
    st.session_state.messages = []
    st.session_state.connected = True
    st.session_state.models = {'anthropic': ['claude-3'], 'openai': ['gpt-4']}
    st.session_state.websocket_thread_started = True
    
    # Create a new mock for research component to ensure clean state
    research_mock = MagicMock()
    research_mock.return_value = {"success": True, "research_notes": ["Test note"], "key_facts": {}}
    
    with patch('webui.app.research_component', research_mock), \
         patch('webui.app.load_environment_status') as mock_env_status:
        
        mock_env_status.return_value = {
            'anthropic': True,
            'openai': True,
            'web_research_enabled': True
        }
        
        # First call to set up the UI
        mock_streamlit['button'].return_value = False
        main()
        
        # Second call to simulate clicking the Start button
        mock_streamlit['button'].return_value = True
        main()
        
        # Verify only research was called once
        research_mock.assert_called_once()
        mock_planning.assert_not_called()
        mock_implementation.assert_not_called()
        
        # Verify research only mode was set
        assert mock_memory['config']['research_only'] == True

def test_websocket_thread_success(mock_session_state):
    """Test successful websocket thread execution."""
    with patch('webui.socket_interface.SocketInterface.connect_server', new_callable=AsyncMock) as mock_connect, \
         patch('webui.socket_interface.SocketInterface.setup_handlers', new_callable=AsyncMock) as mock_setup, \
         patch('webui.socket_interface.SocketInterface.register_handler') as mock_register:
        
        mock_connect.return_value = True
        websocket_thread()
        
        assert st.session_state.connected == True
        mock_connect.assert_called_once()
        mock_setup.assert_called_once()
        mock_register.assert_called_once_with("message", handle_output)

def test_websocket_thread_failure(mock_session_state):
    """Test websocket thread failure."""
    with patch('webui.socket_interface.SocketInterface.connect_server', new_callable=AsyncMock) as mock_connect:
        mock_connect.side_effect = Exception("Connection failed")
        websocket_thread()
        
        assert st.session_state.connected == False

def test_handle_output():
    """Test message handling."""
    with patch('webui.app.message_queue') as mock_queue:
        test_message = {"type": "test", "content": "Test message"}
        handle_output(test_message)
        mock_queue.put.assert_called_once_with(test_message) 