import pytest
import streamlit as st
from unittest.mock import MagicMock, patch, Mock
from webui.app import main, initialize_session_state, get_configured_providers, load_available_models, filter_models, handle_conversation, WebUIConfig
from components.memory import _global_memory

@pytest.fixture
def mock_streamlit():
    """Mock Streamlit components"""
    mocks = {
        'text_area': MagicMock(),
        'button': MagicMock(),
        'radio': MagicMock(),
        'selectbox': MagicMock(),
        'checkbox': MagicMock(),
        'text_input': MagicMock(),
        'sidebar': MagicMock(),
        'spinner': MagicMock(),
        'success': MagicMock(),
        'error': MagicMock(),
        'write': MagicMock(),
        'caption': MagicMock(),
        'subheader': MagicMock(),
        'header': MagicMock(),
        'title': MagicMock(),
        'set_page_config': MagicMock()
    }
    
    with patch.multiple('streamlit', **mocks):
        yield mocks

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up environment variables for testing"""
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key')
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    monkeypatch.setenv('TAVILY_API_KEY', 'test-key')

@pytest.fixture
def mock_session_state():
    """Mock Streamlit session state"""
    if not hasattr(st, 'session_state'):
        setattr(st, 'session_state', type('SessionState', (), {}))
    return st.session_state

@pytest.fixture
def mock_memory():
    """Mock global memory"""
    _global_memory.clear()
    return _global_memory

@pytest.fixture
def mock_config():
    """Create a mock WebUI configuration."""
    config = Mock(spec=WebUIConfig)
    config.temperature = 0.7
    config.max_tokens = 2000
    return config

def test_initialize_session_state(mock_session_state):
    """Test session state initialization"""
    initialize_session_state()
    assert hasattr(st.session_state, 'messages')
    assert hasattr(st.session_state, 'connected')
    assert hasattr(st.session_state, 'models')
    assert hasattr(st.session_state, 'websocket_thread_started')

def test_get_configured_providers(mock_env_vars):
    """Test provider configuration retrieval"""
    providers = get_configured_providers()
    assert 'anthropic' in providers
    assert 'openai' in providers
    assert providers['anthropic']['env_key'] == 'ANTHROPIC_API_KEY'
    assert providers['openai']['env_key'] == 'OPENAI_API_KEY'

def test_load_available_models(mock_env_vars):
    """Test model loading"""
    models = load_available_models()
    assert 'anthropic' in models
    assert 'openai' in models
    assert len(models['anthropic']) > 0
    assert len(models['openai']) > 0

def test_filter_models():
    """Test model filtering"""
    models = [
        'anthropic/claude-3',
        'openai/gpt-4',
        'openai/gpt-3.5-turbo'
    ]
    
    # Test filtering by provider
    filtered = filter_models(models, 'openai')
    assert len(filtered) == 2
    assert all('openai' in model for model in filtered)
    
    # Test filtering by model name
    filtered = filter_models(models, 'claude')
    assert len(filtered) == 1
    assert 'claude' in filtered[0]
    
    # Test empty query
    filtered = filter_models(models, '')
    assert len(filtered) == len(models)

def test_main_success(mock_streamlit, mock_env_vars, mock_session_state, mock_memory):
    """Test successful main flow"""
    # Set up initial state
    mock_streamlit['text_area'].return_value = "test task"
    mock_streamlit['radio'].return_value = "Full Development"
    mock_streamlit['selectbox'].side_effect = ["anthropic", "claude-3"] * 5  # Provide enough values for multiple calls
    mock_streamlit['checkbox'].side_effect = [True, True, True] * 5  # Provide enough values for multiple calls
    mock_streamlit['text_input'].return_value = "test_key"

    st.session_state.messages = []
    st.session_state.connected = True
    st.session_state.models = {'anthropic': ['claude-3'], 'openai': ['gpt-4']}
    st.session_state.websocket_thread_started = True
    st.session_state.execution_stage = None  # Reset execution stage
    st.session_state.research_results = None  # Reset research results
    st.session_state.planning_results = None  # Reset planning results

    # Initialize mock memory
    mock_memory.clear()
    mock_memory.update({
        'config': {
            'provider': 'anthropic',
            'model': 'claude-3',
            'research_only': False,
            'cowboy_mode': True,
            'hil': True,
            'web_research_enabled': True
        },
        'related_files': {},
        'implementation_requested': False,
        'research_notes': ["Test note"],
        'plans': [],
        'tasks': {},
        'key_facts': {},
        'key_snippets': {}
    })

    # Create mocks for components
    research_mock = MagicMock()
    research_mock.return_value = {"success": True, "research_notes": ["Test note"], "key_facts": {}}

    planning_mock = MagicMock()
    planning_mock.return_value = {"success": True, "plan": "Test plan", "tasks": ["Task 1"]}

    implementation_mock = MagicMock()
    implementation_mock.return_value = {"success": True, "implemented_tasks": ["Task 1"]}

    with patch('webui.app.research_component', research_mock), \
         patch('webui.app.planning_component', planning_mock), \
         patch('webui.app.implementation_component', implementation_mock), \
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
        assert mock_memory['config']['research_only'] == False

        # Verify components were called
        research_mock.assert_called_once()
        planning_mock.assert_called_once()
        implementation_mock.assert_called_once()
        mock_streamlit['spinner'].assert_called()

def test_main_research_only(mock_streamlit, mock_env_vars, mock_session_state, mock_memory):
    """Test research-only mode"""
    # Set up initial state
    mock_streamlit['text_area'].return_value = "test task"
    mock_streamlit['radio'].return_value = "Research Only"
    mock_streamlit['selectbox'].side_effect = ["anthropic", "claude-3"] * 5  # Provide enough values for multiple calls
    mock_streamlit['checkbox'].side_effect = [True, True, True] * 5  # Provide enough values for multiple calls
    mock_streamlit['text_input'].return_value = "test_key"

    st.session_state.messages = []
    st.session_state.connected = True
    st.session_state.models = {'anthropic': ['claude-3']}
    st.session_state.websocket_thread_started = True

    # Initialize mock memory
    mock_memory.clear()
    mock_memory.update({
        'config': {
            'provider': 'anthropic',
            'model': 'claude-3',
            'research_only': True,
            'cowboy_mode': True,
            'hil': True,
            'web_research_enabled': True
        },
        'related_files': {},
        'implementation_requested': False,
        'research_notes': [],
        'plans': [],
        'tasks': {},
        'key_facts': {},
        'key_snippets': {}
    })

    # Create mocks for components
    research_mock = MagicMock()
    research_mock.return_value = {"success": True, "research_notes": ["Test note"], "key_facts": {}}

    planning_mock = MagicMock()
    implementation_mock = MagicMock()

    with patch('webui.app.research_component', research_mock), \
         patch('webui.app.planning_component', planning_mock), \
         patch('webui.app.implementation_component', implementation_mock), \
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

        # Verify only research was called
        research_mock.assert_called_once()
        planning_mock.assert_not_called()
        implementation_mock.assert_not_called()

def test_main_no_task(mock_streamlit, mock_env_vars, mock_session_state):
    """Test main flow with no task"""
    mock_streamlit['text_area'].return_value = ""
    mock_streamlit['button'].return_value = True
    
    st.session_state.messages = []
    st.session_state.connected = True
    st.session_state.models = {'anthropic': ['claude-3'], 'openai': ['gpt-4']}
    st.session_state.websocket_thread_started = True
    
    main()
    mock_streamlit['error'].assert_called_once_with("Please enter a valid task or query.")

def test_main_no_providers(mock_streamlit, mock_session_state):
    """Test main flow with no configured providers"""
    st.session_state.models = {}
    st.session_state.websocket_thread_started = True
    
    main()
    mock_streamlit['error'].assert_called_once_with("No API providers configured. Please check your .env file.") 

@pytest.mark.asyncio
async def test_handle_conversation_with_provider_prefix(mock_config):
    """Test conversation handling with provider/model format."""
    mock_config.model = "anthropic/claude-3"
    mock_config.provider = "anthropic"
    
    with patch('webui.app.create_model') as mock_create_model, \
         patch('webui.app.run_conversation_agent') as mock_run_agent:
        mock_run_agent.return_value = "Test response"
        
        result = await handle_conversation("Test task", mock_config)
        
        assert result == "Test response"
        mock_create_model.assert_called_once_with(
            provider="anthropic",
            model_name="claude-3",
            temperature=0.7,
            max_tokens=2000
        )

@pytest.mark.asyncio
async def test_handle_conversation_without_provider_prefix(mock_config):
    """Test conversation handling with model name only."""
    mock_config.model = "claude-3"
    mock_config.provider = "anthropic"
    
    with patch('webui.app.create_model') as mock_create_model, \
         patch('webui.app.run_conversation_agent') as mock_run_agent:
        mock_run_agent.return_value = "Test response"
        
        result = await handle_conversation("Test task", mock_config)
        
        assert result == "Test response"
        mock_create_model.assert_called_once_with(
            provider="anthropic",
            model_name="claude-3",
            temperature=0.7,
            max_tokens=2000
        )

@pytest.mark.asyncio
async def test_handle_conversation_error_handling(mock_config):
    """Test error handling in conversation handler."""
    mock_config.model = "claude-3"
    mock_config.provider = "anthropic"
    
    with patch('webui.app.create_model') as mock_create_model, \
         patch('webui.app.logger') as mock_logger:
        mock_create_model.side_effect = Exception("Test error")
        
        with pytest.raises(Exception) as exc_info:
            await handle_conversation("Test task", mock_config)
        
        assert "LLM Error: Test error" in str(exc_info.value)
        mock_logger.error.assert_called_once() 