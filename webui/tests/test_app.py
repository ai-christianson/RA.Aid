import pytest
from unittest.mock import Mock, AsyncMock, patch
import streamlit as st
from webui.app import initialize_session_state, handle_task
from webui.config import WebUIConfig

@pytest.fixture
def mock_config():
    with patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test_key',
        'ANTHROPIC_API_KEY': 'test_key',
        'OPENROUTER_API_KEY': 'test_key'
    }):
        return WebUIConfig(
            provider="openai",
            model="openai/gpt-4",
            cowboy_mode=True
        )

@pytest.fixture(autouse=True)
def setup_streamlit():
    """Setup Streamlit session state"""
    # Mock Streamlit session state
    if not hasattr(st, "session_state"):
        st.session_state = {}
    
    # Clear existing state
    st.session_state.clear()
    
    yield
    
    # Clean up
    st.session_state.clear()

def test_initialize_session_state():
    """Test session state initialization"""
    # Mock models
    with patch("webui.app.load_available_models") as mock_load_models:
        mock_load_models.return_value = {
            "openai": ["gpt-4"],
            "anthropic": ["claude-2.1"],
            "openrouter": ["openai/gpt-4"]
        }
        
        initialize_session_state()
        
        assert "messages" in st.session_state
        assert len(st.session_state.messages) == 1
        assert st.session_state.messages[0]["role"] == "assistant"
        assert "Welcome" in st.session_state.messages[0]["content"]
        assert "models" in st.session_state

@pytest.mark.asyncio
async def test_handle_task_shell_command(mock_config):
    """Test shell command handling"""
    # Initialize session state
    initialize_session_state()
    
    # Clear messages after initialization
    st.session_state.messages = []
    
    with patch("subprocess.run") as mock_run, \
         patch("webui.app.determine_interaction_type", return_value="shell"), \
         patch("webui.app.load_available_models", return_value={}):
        mock_run.return_value.stdout = "command output"
        
        await handle_task("run pwd", mock_config)
        
        mock_run.assert_called_once_with(
            "pwd",
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        
        assert len(st.session_state.messages) == 2
        assert st.session_state.messages[0]["role"] == "user"
        assert st.session_state.messages[0]["content"] == "run pwd"
        assert st.session_state.messages[1]["role"] == "assistant"
        assert st.session_state.messages[1]["content"] == "command output"

@pytest.mark.asyncio
async def test_handle_task_llm(mock_config):
    """Test LLM task handling"""
    # Initialize session state
    initialize_session_state()
    
    # Clear messages after initialization
    st.session_state.messages = []
    
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="LLM response"))]
    
    print(f"Mock config model: {mock_config.model}")  # Debug print
    
    with patch("webui.app.completion", return_value=mock_response) as mock_completion, \
         patch("webui.app.determine_interaction_type", return_value="conversation"), \
         patch("webui.app.load_available_models", return_value={}):
        
        print("Before handle_task call")  # Debug print
        await handle_task("Hello, how are you?", mock_config)
        print("After handle_task call")  # Debug print
        
        print(f"Mock completion call count: {mock_completion.call_count}")  # Debug print
        if mock_completion.call_args_list:
            print(f"Mock completion call args: {mock_completion.call_args_list[0]}")  # Debug print
        
        mock_completion.assert_called_once_with(
            model=mock_config.model,  # The model already includes the provider prefix
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            temperature=0.7,
            max_tokens=2000
        )
        
        assert len(st.session_state.messages) == 2
        assert st.session_state.messages[0]["role"] == "user"
        assert st.session_state.messages[0]["content"] == "Hello, how are you?"
        assert st.session_state.messages[1]["role"] == "assistant"
        assert st.session_state.messages[1]["content"] == "LLM response" 