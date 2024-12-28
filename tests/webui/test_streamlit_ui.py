import pytest
import streamlit as st
from unittest.mock import MagicMock, patch, AsyncMock
from queue import Queue, Empty
from webui.app import (
    render_environment_status, render_messages, process_message_queue, 
    filter_models, main, send_task, initialize_session_state,
    message_queue
)
from components.memory import _global_memory

@pytest.fixture
def mock_streamlit():
    """Mock Streamlit components"""
    mock_empty = MagicMock()
    mock_empty.text = MagicMock()
    
    mock_spinner = MagicMock()
    mock_spinner.__enter__ = MagicMock()
    mock_spinner.__exit__ = MagicMock()
    
    mock_components = {
        'write': MagicMock(),
        'error': MagicMock(),
        'success': MagicMock(),
        'markdown': MagicMock(),
        'empty': MagicMock(return_value=mock_empty),
        'sidebar': MagicMock(),
        'button': MagicMock(),
        'text_input': MagicMock(),
        'text_area': MagicMock(),
        'selectbox': MagicMock(),
        'radio': MagicMock(),
        'checkbox': MagicMock(),
        'code': MagicMock(),
        'subheader': MagicMock(),
        'caption': MagicMock(),
        'spinner': MagicMock(return_value=mock_spinner)
    }
    
    # Mock sidebar components
    mock_components['sidebar'].subheader = MagicMock()
    mock_components['sidebar'].caption = MagicMock()
    
    with patch.multiple('streamlit', **mock_components):
        yield mock_components

@pytest.fixture
def mock_message_queue():
    """Mock message queue with proper queue functionality"""
    class MockQueue:
        def __init__(self):
            self.items = []
        
        def put(self, item):
            self.items.append(item)
        
        def get_nowait(self):
            if not self.items:
                raise Empty()
            return self.items.pop(0)
    
    queue = MockQueue()
    with patch('webui.app.message_queue', queue):
        yield queue

@pytest.fixture
def mock_session_state():
    """Mock Streamlit session state"""
    if not hasattr(st, 'session_state'):
        st.session_state = {}
    st.session_state.messages = []
    st.session_state.message_queue = []
    yield st.session_state

@pytest.fixture
def mock_queue():
    """Mock message queue"""
    return Queue()

@pytest.fixture
def mock_components():
    """Mock RA.Aid components"""
    mocks = {
        'research_component': MagicMock(return_value={
            'success': True,
            'research_notes': ['Found relevant information'],
            'key_facts': {'fact1': 'value1'}
        }),
        'planning_component': MagicMock(return_value={
            'success': True,
            'plan': 'Implementation plan',
            'tasks': ['Task 1', 'Task 2']
        }),
        'implementation_component': MagicMock(return_value={
            'success': True,
            'implemented_tasks': ['Task 1', 'Task 2']
        })
    }
    with patch.multiple('webui.app', **mocks):
        yield mocks

@pytest.fixture
def mock_memory():
    """Mock global memory"""
    _global_memory.clear()
    return _global_memory

def test_render_environment_status(mock_streamlit, monkeypatch):
    """Test environment status rendering"""
    # Mock environment status
    env_status = {
        'openai': True,
        'anthropic': False,
        'tavily': True
    }
    
    with patch('webui.app.load_environment_status', return_value=env_status):
        render_environment_status()
        
        # Verify sidebar components were called
        mock_streamlit['sidebar'].subheader.assert_called_once_with("Environment Status")
        mock_streamlit['sidebar'].caption.assert_called_once()
        
        # Verify status text format
        caption_text = mock_streamlit['sidebar'].caption.call_args[0][0]
        assert "API Status:" in caption_text
        assert "openai: ✓" in caption_text
        assert "anthropic: ×" in caption_text
        assert "tavily: ✓" in caption_text

def test_render_messages_empty(mock_streamlit, mock_session_state):
    """Test rendering empty message list"""
    st.session_state.messages = []
    render_messages()
    mock_streamlit['write'].assert_not_called()
    mock_streamlit['error'].assert_not_called()

def test_render_messages_with_content(mock_streamlit, mock_session_state):
    """Test rendering messages with content"""
    st.session_state.messages = [
        {'type': 'standard', 'content': 'Test message'},
        {'type': 'error', 'content': 'Error message'},
        {'content': 'No type message'}
    ]
    
    render_messages()
    
    # Verify each message was rendered
    mock_streamlit['write'].assert_any_call('Test message')
    mock_streamlit['error'].assert_called_once_with('Error message')
    mock_streamlit['write'].assert_any_call('No type message')

def test_process_message_queue_empty(mock_queue, mock_session_state):
    """Test processing empty message queue"""
    st.session_state.messages = []
    process_message_queue()
    assert len(st.session_state.messages) == 0

def test_process_message_queue_with_messages(mock_queue, mock_session_state):
    """Test processing message queue with content"""
    # Mock the global message queue
    with patch('webui.app.message_queue', mock_queue):
        # Add messages to queue
        messages = [
            {'type': 'standard', 'content': 'Test message'},
            {'type': 'error', 'content': 'Error message'},
            'Invalid message'  # Should be ignored
        ]
        
        for msg in messages:
            mock_queue.put(msg)
        
        st.session_state.messages = []
        process_message_queue()
        
        # Verify messages were processed
        assert len(st.session_state.messages) == 2  # Invalid message should be ignored
        assert st.session_state.messages[0]['content'] == 'Test message'
        assert st.session_state.messages[1]['type'] == 'error'

def test_filter_models_empty():
    """Test model filtering with empty input"""
    models = []
    assert filter_models(models, '') == []
    assert filter_models(models, 'test') == []

def test_filter_models_by_provider():
    """Test model filtering by provider"""
    models = [
        'openai/gpt-4',
        'anthropic/claude-3',
        'openai/gpt-3.5-turbo'
    ]
    
    filtered = filter_models(models, 'openai')
    assert len(filtered) == 2
    assert all('openai' in model for model in filtered)

def test_filter_models_by_name():
    """Test model filtering by model name"""
    models = [
        'openai/gpt-4',
        'anthropic/claude-3',
        'openai/gpt-3.5-turbo'
    ]
    
    filtered = filter_models(models, 'gpt')
    assert len(filtered) == 2
    assert all('gpt' in model for model in filtered)

def test_filter_models_case_insensitive():
    """Test case-insensitive model filtering"""
    models = [
        'openai/GPT-4',
        'anthropic/Claude-3',
        'openai/gpt-3.5-turbo'
    ]
    
    filtered = filter_models(models, 'gpt')
    assert len(filtered) == 2
    assert all('GPT' in model.upper() for model in filtered)

def test_filter_models_no_match():
    """Test model filtering with no matches"""
    models = [
        'openai/gpt-4',
        'anthropic/claude-3',
        'openai/gpt-3.5-turbo'
    ]
    
    filtered = filter_models(models, 'nonexistent')
    assert len(filtered) == 0 

def test_task_execution_research_only(mock_streamlit, mock_session_state, mock_components, mock_memory):
    """Test task execution in research-only mode"""
    # Set up session state
    initialize_session_state()
    st.session_state.connected = True
    st.session_state.models = {'openai': ['gpt-4', 'gpt-3.5-turbo']}  # Using valid OpenAI models
    st.session_state.websocket_thread_started = True
    
    # Mock user inputs
    mock_streamlit['text_area'].return_value = "Research task"
    mock_streamlit['radio'].return_value = "Research Only"
    mock_streamlit['selectbox'].side_effect = ["openai", "gpt-4"]  # Using valid provider and model
    mock_streamlit['checkbox'].side_effect = [True, True, True]  # cowboy_mode, hil, web_research
    mock_streamlit['button'].return_value = True
    
    # Run main function
    main()
    
    # Verify research component was called
    mock_components['research_component'].assert_called_once()
    mock_components['planning_component'].assert_not_called()
    mock_components['implementation_component'].assert_not_called()
    
    # Verify UI updates
    mock_streamlit['spinner'].assert_any_call("Conducting Research...")
    mock_streamlit['success'].assert_called()
    
    # Verify memory updates
    assert mock_memory['config']['research_only'] is True
    assert 'research_notes' in mock_memory
    assert mock_memory['research_notes'] == ['Found relevant information']

def test_task_execution_full_development(mock_streamlit, mock_session_state, mock_components, mock_memory):
    """Test task execution in full development mode"""
    # Set up session state
    initialize_session_state()
    st.session_state.connected = True
    st.session_state.models = {'openai': ['gpt-4', 'gpt-3.5-turbo']}  # Using valid OpenAI models
    st.session_state.websocket_thread_started = True
    
    # Mock user inputs
    mock_streamlit['text_area'].return_value = "Development task"
    mock_streamlit['radio'].return_value = "Full Development"
    mock_streamlit['selectbox'].side_effect = ["openai", "gpt-4"]  # Using valid provider and model
    mock_streamlit['checkbox'].side_effect = [True, True, True]  # cowboy_mode, hil, web_research
    mock_streamlit['button'].return_value = True
    
    # Run main function
    main()
    
    # Verify all components were called in order
    mock_components['research_component'].assert_called_once()
    mock_components['planning_component'].assert_called_once()
    mock_components['implementation_component'].assert_called_once()
    
    # Verify UI updates for each phase
    mock_streamlit['spinner'].assert_any_call("Conducting Research...")
    mock_streamlit['spinner'].assert_any_call("Planning Implementation...")
    mock_streamlit['spinner'].assert_any_call("Implementing Changes...")
    
    # Verify memory updates
    assert mock_memory['config']['research_only'] is False
    assert 'research_notes' in mock_memory
    assert 'plans' in mock_memory
    assert 'tasks' in mock_memory

def test_task_execution_research_failure(mock_streamlit, mock_session_state, mock_components, mock_memory):
    """Test task execution when research fails"""
    # Set up session state
    initialize_session_state()
    st.session_state.connected = True
    st.session_state.models = {'openai': ['gpt-4', 'gpt-3.5-turbo']}  # Using valid OpenAI models
    st.session_state.websocket_thread_started = True
    
    # Mock research failure
    mock_components['research_component'].return_value = {
        'success': False,
        'error': 'Research failed'
    }
    
    # Mock user inputs
    mock_streamlit['text_area'].return_value = "Failed task"
    mock_streamlit['radio'].return_value = "Full Development"
    mock_streamlit['selectbox'].side_effect = ["openai", "gpt-4"]  # Using valid provider and model
    mock_streamlit['checkbox'].side_effect = [True, True, True]
    mock_streamlit['button'].return_value = True
    
    # Run main function
    main()
    
    # Verify only research was attempted
    mock_components['research_component'].assert_called_once()
    mock_components['planning_component'].assert_not_called()
    mock_components['implementation_component'].assert_not_called()
    
    # Verify error was shown
    mock_streamlit['error'].assert_called()
    
    # Verify memory state
    assert 'error' in mock_memory
    assert mock_memory.get('research_notes') == []

def test_task_execution_progress_updates(mock_streamlit, mock_session_state, mock_components, mock_memory):
    """Test progress updates during task execution"""
    # Set up session state
    initialize_session_state()
    st.session_state.connected = True
    st.session_state.models = {'openai': ['gpt-4', 'gpt-3.5-turbo']}  # Using valid OpenAI models
    st.session_state.websocket_thread_started = True
    
    # Mock progress updates in components
    def research_with_progress(*args, **kwargs):
        st.session_state.execution_stage = "research"
        return {'success': True, 'research_notes': ['Research complete']}
    
    def planning_with_progress(*args, **kwargs):
        st.session_state.execution_stage = "planning"
        return {'success': True, 'plan': 'Plan complete', 'tasks': ['Task 1']}
    
    def implementation_with_progress(*args, **kwargs):
        st.session_state.execution_stage = "implementation"
        return {'success': True, 'implemented_tasks': ['Task 1']}
    
    mock_components['research_component'].side_effect = research_with_progress
    mock_components['planning_component'].side_effect = planning_with_progress
    mock_components['implementation_component'].side_effect = implementation_with_progress
    
    # Mock user inputs
    mock_streamlit['text_area'].return_value = "Progress task"
    mock_streamlit['radio'].return_value = "Full Development"
    mock_streamlit['selectbox'].side_effect = ["openai", "gpt-4"]  # Using valid provider and model
    mock_streamlit['checkbox'].side_effect = [True, True, True]
    mock_streamlit['button'].return_value = True
    
    # Run main function
    main()
    
    # Verify execution stages were updated
    assert hasattr(st.session_state, 'execution_stage')
    assert st.session_state.execution_stage == "implementation"
    
    # Verify progress indicators
    mock_streamlit['spinner'].assert_any_call("Conducting Research...")
    mock_streamlit['spinner'].assert_any_call("Planning Implementation...")
    mock_streamlit['spinner'].assert_any_call("Implementing Changes...")
    
    # Verify results were stored
    assert st.session_state.research_results is not None
    assert st.session_state.planning_results is not None

def test_research_message_display(mock_streamlit, mock_session_state, mock_message_queue):
    """Test that research messages are properly displayed in the UI"""
    # Initialize session state
    initialize_session_state()
    st.session_state.messages = []
    
    # Add some research messages to the queue
    research_messages = [
        {"type": "status", "content": "🔍 Starting Research Phase..."},
        {"type": "research", "content": "Analyzing codebase structure..."},
        {"type": "research", "content": "Found relevant files: main.py, utils.py"},
        {"type": "research", "content": "Examining dependencies..."},
        {"type": "status", "content": "✅ Research phase complete"}
    ]
    
    for msg in research_messages:
        mock_message_queue.put(msg)
    
    # Process the message queue
    process_message_queue()
    
    # Verify messages were added to session state
    assert len(st.session_state.messages) == len(research_messages)
    
    # Verify messages were displayed
    for msg in research_messages:
        if msg["type"] == "status":
            mock_streamlit['write'].assert_any_call(msg["content"])
        else:
            mock_streamlit['markdown'].assert_any_call(msg["content"])

def test_research_progress_updates(mock_streamlit, mock_memory):
    """Test that research progress updates are shown correctly"""
    # Initialize session state
    initialize_session_state()
    st.session_state.messages = []
    
    # Create a mock empty container that persists across messages
    mock_empty = MagicMock()
    mock_empty.text = MagicMock()
    mock_streamlit['empty'].return_value = mock_empty
    
    # Simulate a research progress sequence
    progress_messages = [
        {"type": "status", "content": "🔍 Starting Research Phase..."},
        {"type": "progress", "content": "Searching files: 25%"},
        {"type": "progress", "content": "Searching files: 50%"},
        {"type": "progress", "content": "Searching files: 75%"},
        {"type": "progress", "content": "Searching files: 100%"},
        {"type": "status", "content": "✅ Research complete"}
    ]
    
    # Add messages to session state and process them
    for msg in progress_messages:
        st.session_state.messages.append(msg)
        render_messages()  # Call render_messages to display the messages
        
        # Verify message was displayed correctly
        if msg["type"] == "progress":
            mock_empty.text.assert_called_with(msg["content"])
        else:
            mock_streamlit['write'].assert_called_with(msg["content"])

def test_research_error_display(mock_streamlit, mock_session_state, mock_message_queue):
    """Test that research errors are properly displayed"""
    # Add error message
    error_message = {
        "type": "error",
        "content": "❌ Research failed: Could not access repository"
    }
    
    mock_message_queue.put(error_message)
    process_message_queue()
    
    # Verify error was displayed
    mock_streamlit['error'].assert_called_with(error_message["content"])

def test_research_output_formatting(mock_streamlit, mock_session_state, mock_message_queue):
    """Test that research output is properly formatted"""
    # Add formatted research output
    research_output = {
        "type": "research",
        "content": """
### Research Notes
- Found important file: main.py
- Code contains helper functions
- Documentation needs updating

### Key Facts
- Language: Python 3.8
- Framework: FastAPI
- Test Coverage: 80%
"""
    }
    
    mock_message_queue.put(research_output)
    process_message_queue()
    
    # Verify formatted output was displayed
    mock_streamlit['markdown'].assert_called_with(research_output["content"])

def test_process_message_queue(mock_streamlit, mock_session_state, mock_message_queue):
    """Test message queue processing"""
    # Add test messages
    messages = [
        {"type": "status", "content": "Test status"},
        {"type": "research", "content": "Test research"},
        {"type": "error", "content": "Test error"}
    ]
    
    for msg in messages:
        mock_message_queue.put(msg)
    
    # Process messages
    process_message_queue()
    
    # Verify messages were added to session state
    assert len(st.session_state.messages) == len(messages)
    assert all(msg in st.session_state.messages for msg in messages)
    
    # Verify messages were displayed correctly
    mock_streamlit['write'].assert_any_call("Test status")
    mock_streamlit['markdown'].assert_any_call("Test research")
    mock_streamlit['error'].assert_any_call("Test error")

def test_render_messages(mock_streamlit, mock_session_state):
    """Test message rendering"""
    # Create a mock empty container that persists
    mock_empty = MagicMock()
    mock_empty.text = MagicMock()
    mock_streamlit['empty'].return_value = mock_empty
    
    # Add test messages
    st.session_state.messages = [
        {"type": "status", "content": "Test status"},
        {"type": "research", "content": "Test research"},
        {"type": "error", "content": "Test error"},
        {"type": "progress", "content": "Test progress"},
        {"type": "success", "content": "Test success"}
    ]
    
    # Render messages
    render_messages()
    
    # Verify each message type was rendered correctly
    mock_streamlit['write'].assert_any_call("Test status")
    mock_streamlit['markdown'].assert_any_call("Test research")
    mock_streamlit['error'].assert_any_call("Test error")
    mock_empty.text.assert_any_call("Test progress")
    mock_streamlit['success'].assert_any_call("Test success") 