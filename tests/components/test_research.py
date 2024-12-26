import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
from components.research import research_component

@pytest.fixture
def mock_initialize_llm():
    with patch('components.research.initialize_llm') as mock:
        yield mock

@pytest.fixture
def mock_run_research_agent():
    with patch('components.research.run_research_agent') as mock:
        mock.return_value = {
            'key_facts': ['fact1', 'fact2'],
            'key_snippets': ['snippet1', 'snippet2'],
            'success': True
        }
        yield mock

def test_research_success(mock_initialize_llm, mock_run_research_agent):
    task = "test task"
    config = {"provider": "test-provider", "model": "test-model", "research_only": False, "hil": False, "web_research_enabled": False}

    with patch('streamlit.write') as mock_write, \
         patch('streamlit.header') as mock_header:
        result = research_component(task, config)

        assert result['success']
        assert len(result['key_facts']) == 2
        assert len(result['key_snippets']) == 2
        mock_initialize_llm.assert_called_once_with(config["provider"], config["model"])
        mock_run_research_agent.assert_called_once_with(
            task,
            mock_initialize_llm.return_value,
            expert_enabled=True,
            research_only=config["research_only"],
            hil=config["hil"],
            web_research_enabled=config["web_research_enabled"],
            config=config
        )

def test_research_failure(mock_initialize_llm, mock_run_research_agent):
    mock_run_research_agent.return_value = {
        'error': 'Research failed',
        'success': False
    }

    task = "test task"
    config = {"provider": "test-provider", "model": "test-model", "research_only": False, "hil": False, "web_research_enabled": False}

    with patch('streamlit.write') as mock_write, \
         patch('streamlit.header') as mock_header:
        result = research_component(task, config)

        assert not result['success']
        assert 'Research failed' in result['error']
        mock_initialize_llm.assert_called_once_with(config["provider"], config["model"])
        mock_run_research_agent.assert_called_once_with(
            task,
            mock_initialize_llm.return_value,
            expert_enabled=True,
            research_only=config["research_only"],
            hil=config["hil"],
            web_research_enabled=config["web_research_enabled"],
            config=config
        )

def test_research_display(mock_initialize_llm, mock_run_research_agent):
    task = "test task"
    config = {"provider": "test-provider", "model": "test-model", "research_only": False, "hil": False, "web_research_enabled": False}

    with patch('streamlit.write') as mock_write, \
         patch('streamlit.header') as mock_header:
        result = research_component(task, config)

        assert result['success']
        mock_write.assert_called()
        assert mock_write.call_count >= 1  # Facts + snippets
        mock_run_research_agent.assert_called_once_with(
            task,
            mock_initialize_llm.return_value,
            expert_enabled=True,
            research_only=config["research_only"],
            hil=config["hil"],
            web_research_enabled=config["web_research_enabled"],
            config=config
        )

def test_research_config_handling(mock_initialize_llm, mock_run_research_agent):
    task = "test task"
    config = {
        "provider": "test-provider",
        "model": "test-model",
        "max_tokens": 2000,
        "research_only": False,
        "hil": False,
        "web_research_enabled": False
    }

    with patch('streamlit.write') as mock_write, \
         patch('streamlit.header') as mock_header:
        result = research_component(task, config)

        assert result['success']
        mock_initialize_llm.assert_called_once_with(config["provider"], config["model"])
        mock_run_research_agent.assert_called_once_with(
            task,
            mock_initialize_llm.return_value,
            expert_enabled=True,
            research_only=config["research_only"],
            hil=config["hil"],
            web_research_enabled=config["web_research_enabled"],
            config=config
        ) 