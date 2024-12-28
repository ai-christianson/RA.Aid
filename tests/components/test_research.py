import pytest
from unittest.mock import MagicMock, patch
import streamlit as st
from components.research import research_component
from components.memory import _global_memory

@pytest.fixture
def mock_streamlit():
    """Mock Streamlit components"""
    with patch('streamlit.write') as mock_write, \
         patch('streamlit.markdown') as mock_markdown, \
         patch('streamlit.code') as mock_code, \
         patch('streamlit.error') as mock_error:
        yield {
            'write': mock_write,
            'markdown': mock_markdown,
            'code': mock_code,
            'error': mock_error
        }

@pytest.fixture
def mock_llm():
    """Mock LLM initialization"""
    with patch('components.research.initialize_llm') as mock:
        mock.return_value = MagicMock()  # Return a mock model
        yield mock

@pytest.fixture
def mock_research_agent():
    """Mock research agent"""
    with patch('components.research.run_research_agent') as mock:
        yield mock

@pytest.fixture
def mock_memory():
    """Mock global memory"""
    _global_memory.clear()
    return _global_memory

def test_research_success(mock_streamlit, mock_llm, mock_research_agent, mock_memory):
    """Test successful research execution"""
    # Mock research agent response
    mock_research_agent.return_value = (
        "Research Notes:\n"
        "- Found important file: main.py\n"
        "- Code contains helper functions\n"
        "- Documentation needs updating\n"
        "\n"
        "Key Facts:\n"
        "- Language: Python 3.8\n"
        "- Framework: FastAPI\n"
        "- Test Coverage: 80%"
    )
    
    # Test configuration
    config = {
        "provider": "test-provider",
        "model": "test-model",
        "research_only": True,
        "hil": False,
        "web_research_enabled": True
    }
    
    # Run research component
    results = research_component("Test task", config)
    
    # Verify success
    assert results["success"] is True
    
    # Verify research notes were parsed
    assert len(results["research_notes"]) == 3
    assert "Found important file: main.py" in results["research_notes"]
    
    # Verify key facts were parsed
    assert results["key_facts"]["Language"] == "Python 3.8"
    assert results["key_facts"]["Framework"] == "FastAPI"
    assert results["key_facts"]["Test Coverage"] == "80%"
    
    # Verify memory updates
    assert mock_memory["research_notes"] == results["research_notes"]
    assert mock_memory["key_facts"] == results["key_facts"]
    
    # Verify UI updates
    mock_streamlit['write'].assert_called_with("🔍 Starting Research Phase...")
    mock_streamlit['markdown'].assert_any_call("### Research Notes")
    mock_streamlit['markdown'].assert_any_call("### Key Facts")

def test_research_no_results(mock_streamlit, mock_llm, mock_research_agent, mock_memory):
    """Test handling of no research results"""
    # Mock research agent returning None
    mock_research_agent.return_value = None
    
    config = {
        "provider": "test-provider",
        "model": "test-model",
        "research_only": True,
        "hil": False
    }
    
    # Run research component
    results = research_component("Test task", config)
    
    # Verify failure
    assert results["success"] is False
    assert "error" in results
    assert "Research agent returned no results" in results["error"]
    
    # Verify error was shown
    mock_streamlit['error'].assert_called()

def test_research_invalid_config(mock_streamlit, mock_llm, mock_research_agent, mock_memory):
    """Test handling of invalid configuration"""
    # Missing required fields
    config = {
        "provider": "test-provider"
    }
    
    # Run research component
    results = research_component("Test task", config)
    
    # Verify failure
    assert results["success"] is False
    assert "error" in results
    assert "Missing required configuration field" in results["error"]
    
    # Verify error was shown
    mock_streamlit['error'].assert_called()

def test_research_empty_sections(mock_streamlit, mock_llm, mock_research_agent, mock_memory):
    """Test handling of empty research sections"""
    # Mock research agent returning empty sections
    mock_research_agent.return_value = """
Research Notes:

Key Facts:
"""
    
    config = {
        "provider": "test-provider",
        "model": "test-model",
        "research_only": True,
        "hil": False
    }
    
    # Run research component
    results = research_component("Test task", config)
    
    # Verify success but empty results
    assert results["success"] is True
    assert len(results["research_notes"]) == 0
    assert len(results["key_facts"]) == 0
    
    # Verify memory updates
    assert mock_memory["research_notes"] == []
    assert mock_memory["key_facts"] == {}

def test_research_malformed_key_facts(mock_streamlit, mock_llm, mock_research_agent, mock_memory):
    """Test handling of malformed key facts"""
    # Mock research agent returning malformed key facts
    mock_research_agent.return_value = (
        "Research Notes:\n"
        "- Valid note\n"
        "\n"
        "Key Facts:\n"
        "- Invalid fact without colon\n"
        "- Key with multiple: colons: here"
    )
    
    config = {
        "provider": "test-provider",
        "model": "test-model",
        "research_only": True,
        "hil": False
    }
    
    # Run research component
    results = research_component("Test task", config)
    
    # Verify success
    assert results["success"] is True
    
    # Verify valid note was parsed
    assert len(results["research_notes"]) == 1
    assert "Valid note" in results["research_notes"]
    
    # Verify key facts handling
    assert "Key with multiple" in results["key_facts"]
    assert results["key_facts"]["Key with multiple"] == "colons: here"

def test_research_with_related_files(mock_streamlit, mock_llm, mock_research_agent, mock_memory):
    """Test handling of related files"""
    # Set up mock related files
    mock_memory['related_files'] = {
        'main.py': 'print("Hello")',
        'test.py': 'def test_main():'
    }
    
    mock_research_agent.return_value = """
Research Notes:
- Found code files
"""
    
    config = {
        "provider": "test-provider",
        "model": "test-model",
        "research_only": True,
        "hil": False
    }
    
    # Run research component
    results = research_component("Test task", config)
    
    # Verify success
    assert results["success"] is True
    
    # Verify related files were preserved
    assert results["related_files"] == mock_memory['related_files']
    
    # Verify UI updates
    mock_streamlit['markdown'].assert_any_call("### Related Files")
    mock_streamlit['code'].assert_called()

def test_research_exception_handling(mock_streamlit, mock_llm, mock_research_agent, mock_memory):
    """Test handling of exceptions during research"""
    # Mock research agent raising an exception
    mock_research_agent.side_effect = Exception("Test error")
    
    config = {
        "provider": "test-provider",
        "model": "test-model",
        "research_only": True,
        "hil": False
    }
    
    # Run research component
    results = research_component("Test task", config)
    
    # Verify failure
    assert results["success"] is False
    assert "error" in results
    assert "Test error" in results["error"]
    
    # Verify error was shown
    mock_streamlit['error'].assert_called() 