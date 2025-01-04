import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
from components.planning import planning_component

@pytest.fixture
def mock_initialize_llm():
    with patch('components.planning.initialize_llm') as mock:
        yield mock

@pytest.fixture
def mock_run_planning_agent():
    with patch('components.planning.run_planning_agent') as mock:
        mock.return_value = {
            'plans': ['plan1', 'plan2'],
            'tasks': ['task1', 'task2'],
            'success': True
        }
        yield mock

def test_planning_success(mock_initialize_llm, mock_run_planning_agent):
    task = "test task"
    config = {"provider": "test-provider", "model": "test-model", "hil": False}

    with patch('streamlit.write') as mock_write, \
         patch('streamlit.header') as mock_header:
        result = planning_component(task, config)

        assert result['success']
        assert len(result['plans']) == 2
        assert len(result['tasks']) == 2
        mock_initialize_llm.assert_called_once_with(config["provider"], config["model"])
        mock_run_planning_agent.assert_called_once_with(
            task,
            mock_initialize_llm.return_value,
            expert_enabled=True,
            hil=config["hil"],
            config=config
        )

def test_planning_failure(mock_initialize_llm, mock_run_planning_agent):
    mock_run_planning_agent.return_value = {
        'error': 'Planning failed',
        'success': False
    }

    task = "test task"
    config = {"provider": "test-provider", "model": "test-model", "hil": False}

    with patch('streamlit.write') as mock_write, \
         patch('streamlit.header') as mock_header:
        result = planning_component(task, config)

        assert not result['success']
        assert 'Planning failed' in result['error']
        mock_initialize_llm.assert_called_once_with(config["provider"], config["model"])
        mock_run_planning_agent.assert_called_once_with(
            task,
            mock_initialize_llm.return_value,
            expert_enabled=True,
            hil=config["hil"],
            config=config
        )

def test_planning_display(mock_initialize_llm, mock_run_planning_agent):
    task = "test task"
    config = {"provider": "test-provider", "model": "test-model", "hil": False}

    with patch('streamlit.write') as mock_write, \
         patch('streamlit.header') as mock_header:
        result = planning_component(task, config)

        assert result['success']
        mock_write.assert_called()
        assert mock_write.call_count >= 1  # Plans + tasks
        mock_run_planning_agent.assert_called_once_with(
            task,
            mock_initialize_llm.return_value,
            expert_enabled=True,
            hil=config["hil"],
            config=config
        )

def test_planning_config_handling(mock_initialize_llm, mock_run_planning_agent):
    task = "test task"
    config = {
        "provider": "test-provider",
        "model": "test-model",
        "max_tokens": 2000,
        "hil": False
    }

    with patch('streamlit.write') as mock_write, \
         patch('streamlit.header') as mock_header:
        result = planning_component(task, config)

        assert result['success']
        mock_initialize_llm.assert_called_once_with(config["provider"], config["model"])
        mock_run_planning_agent.assert_called_once_with(
            task,
            mock_initialize_llm.return_value,
            expert_enabled=True,
            hil=config["hil"],
            config=config
        ) 