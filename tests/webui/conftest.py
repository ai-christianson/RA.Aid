import pytest
from unittest.mock import MagicMock, patch
import streamlit as st
from webui.config import WebUIConfig
import socketio

@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing"""
    env_vars = {
        'ANTHROPIC_API_KEY': 'test_anthropic_key',
        'OPENAI_API_KEY': 'test_openai_key',
        'OPENROUTER_API_KEY': 'test_openrouter_key',
        'TAVILY_API_KEY': 'test_tavily_key'
    }
    with patch.dict('os.environ', env_vars):
        yield env_vars

@pytest.fixture
def mock_streamlit():
    """Mock Streamlit for testing"""
    mock_st = MagicMock()
    mock_st.session_state = MagicMock()
    return mock_st

@pytest.fixture
def web_config():
    """Create test WebUI configuration"""
    return WebUIConfig(
        provider='anthropic',
        model='claude-3-opus',
        expert_provider='openai',
        expert_model='gpt-4',
        cowboy_mode=False,
        hil=True,
        web_research_enabled=True
    )

@pytest.fixture
def mock_socketio():
    """Mock SocketIO client for testing"""
    mock_sio = MagicMock(spec=socketio.AsyncClient)
    return mock_sio 