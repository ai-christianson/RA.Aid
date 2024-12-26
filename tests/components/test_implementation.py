import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
from components.implementation import implementation_component

@pytest.fixture
def mock_initialize_llm():
    with patch('components.implementation.initialize_llm') as mock:
        yield mock

@pytest.fixture
def mock_run_task_implementation_agent():
    with patch('components.implementation.run_task_implementation_agent') as mock:
        mock.return_value = {
            'success': True
        }
        yield mock

def test_implementation_success(mock_initialize_llm, mock_run_task_implementation_agent):
    task = "test task"
    research_results = {'key_facts': ['fact1']}
    planning_results = {'tasks': ['task1', 'task2']}
    config = {"provider": "test-provider", "model": "test-model"}

    with patch('streamlit.write') as mock_write, \
         patch('streamlit.header') as mock_header:
        result = implementation_component(task, research_results, planning_results, config)

        assert result['success']
        assert len(result['implemented_tasks']) == 2
        mock_initialize_llm.assert_called_once_with(config["provider"], config["model"])
        
        # Verify that run_task_implementation_agent was called for each task
        assert mock_run_task_implementation_agent.call_count == 2
        
        # Verify first task call
        mock_run_task_implementation_agent.assert_any_call(
            base_task=task,
            tasks=planning_results["tasks"],
            task=planning_results["tasks"][0],
            plan=planning_results.get("plan", ""),
            related_files=research_results.get("related_files", []),
            model=mock_initialize_llm.return_value,
            expert_enabled=True,
            config=config
        )
        
        # Verify second task call
        mock_run_task_implementation_agent.assert_any_call(
            base_task=task,
            tasks=planning_results["tasks"],
            task=planning_results["tasks"][1],
            plan=planning_results.get("plan", ""),
            related_files=research_results.get("related_files", []),
            model=mock_initialize_llm.return_value,
            expert_enabled=True,
            config=config
        )

def test_implementation_failure(mock_initialize_llm, mock_run_task_implementation_agent):
    mock_run_task_implementation_agent.return_value = {
        'error': 'Implementation failed',
        'success': False
    }

    task = "test task"
    research_results = {'key_facts': ['fact1']}
    planning_results = {'tasks': ['task1', 'task2']}
    config = {"provider": "test-provider", "model": "test-model"}

    with patch('streamlit.write') as mock_write, \
         patch('streamlit.header') as mock_header:
        result = implementation_component(task, research_results, planning_results, config)

        assert not result['success']
        assert 'Implementation failed' in result['error']
        mock_initialize_llm.assert_called_once_with(config["provider"], config["model"])
        
        # Verify that run_task_implementation_agent was called only once (stops on first failure)
        mock_run_task_implementation_agent.assert_called_once_with(
            base_task=task,
            tasks=planning_results["tasks"],
            task=planning_results["tasks"][0],
            plan=planning_results.get("plan", ""),
            related_files=research_results.get("related_files", []),
            model=mock_initialize_llm.return_value,
            expert_enabled=True,
            config=config
        )

def test_implementation_display(mock_initialize_llm, mock_run_task_implementation_agent):
    task = "test task"
    research_results = {'key_facts': ['fact1']}
    planning_results = {'tasks': ['task1', 'task2']}
    config = {"provider": "test-provider", "model": "test-model"}

    with patch('streamlit.write') as mock_write, \
         patch('streamlit.header') as mock_header:
        result = implementation_component(task, research_results, planning_results, config)

        assert result['success']
        mock_write.assert_called()
        assert mock_write.call_count >= 1  # Implemented tasks
        
        # Verify that run_task_implementation_agent was called for each task
        assert mock_run_task_implementation_agent.call_count == 2
        
        # Verify first task call
        mock_run_task_implementation_agent.assert_any_call(
            base_task=task,
            tasks=planning_results["tasks"],
            task=planning_results["tasks"][0],
            plan=planning_results.get("plan", ""),
            related_files=research_results.get("related_files", []),
            model=mock_initialize_llm.return_value,
            expert_enabled=True,
            config=config
        )
        
        # Verify second task call
        mock_run_task_implementation_agent.assert_any_call(
            base_task=task,
            tasks=planning_results["tasks"],
            task=planning_results["tasks"][1],
            plan=planning_results.get("plan", ""),
            related_files=research_results.get("related_files", []),
            model=mock_initialize_llm.return_value,
            expert_enabled=True,
            config=config
        )

def test_implementation_config_handling(mock_initialize_llm, mock_run_task_implementation_agent):
    task = "test task"
    research_results = {'key_facts': ['fact1']}
    planning_results = {'tasks': ['task1', 'task2']}
    config = {
        "provider": "test-provider",
        "model": "test-model",
        "max_tokens": 2000
    }

    with patch('streamlit.write') as mock_write, \
         patch('streamlit.header') as mock_header:
        result = implementation_component(task, research_results, planning_results, config)

        assert result['success']
        mock_initialize_llm.assert_called_once_with(config["provider"], config["model"])
        
        # Verify that run_task_implementation_agent was called for each task
        assert mock_run_task_implementation_agent.call_count == 2
        
        # Verify first task call
        mock_run_task_implementation_agent.assert_any_call(
            base_task=task,
            tasks=planning_results["tasks"],
            task=planning_results["tasks"][0],
            plan=planning_results.get("plan", ""),
            related_files=research_results.get("related_files", []),
            model=mock_initialize_llm.return_value,
            expert_enabled=True,
            config=config
        )
        
        # Verify second task call
        mock_run_task_implementation_agent.assert_any_call(
            base_task=task,
            tasks=planning_results["tasks"],
            task=planning_results["tasks"][1],
            plan=planning_results.get("plan", ""),
            related_files=research_results.get("related_files", []),
            model=mock_initialize_llm.return_value,
            expert_enabled=True,
            config=config
        ) 