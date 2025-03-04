import pytest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from ra_aid.agent_backends.ciayn_agent import CiaynAgent
from ra_aid.tools import fuzzy_find_project_files
from ra_aid.exceptions import ToolExecutionError
from ra_aid.file_listing import FileListerError


def test_fuzzy_find_project_files_none_args_execution():
    """Test that the CiaynAgent can correctly execute fuzzy_find_project_files 
    with None arguments as seen in the failing case."""
    
    # Create a mock agent with the fuzzy_find_project_files tool
    mock_model = MagicMock()
    agent = CiaynAgent(
        model=mock_model,
        tools=[fuzzy_find_project_files],
        max_history_messages=5
    )
    
    # This is the exact function call from the error message
    function_call = 'fuzzy_find_project_files(search_term="nonexistent_term", repo_path=".", threshold=60, max_results=10, include_paths=None, exclude_patterns=None)'
    
    # Mock the response from the LLM
    mock_response = AIMessage(content=function_call)
    
    # Patch process.extract to return empty results for any search
    with patch('ra_aid.tools.fuzzy_find.process.extract', return_value=[]):
        result = agent._execute_tool(mock_response)
        assert result == []


def test_error_handling_with_nonexistent_path():
    """Test that we handle errors gracefully with nonexistent paths."""
    
    # Create a mock agent with the fuzzy_find_project_files tool
    mock_model = MagicMock()
    agent = CiaynAgent(
        model=mock_model,
        tools=[fuzzy_find_project_files],
        max_history_messages=5
    )
    
    function_call = 'fuzzy_find_project_files(search_term="test", repo_path="/nonexistent/path", threshold=60, max_results=10)'
    
    # Mock the response from the LLM
    mock_response = AIMessage(content=function_call)
    
    # Patch get_all_project_files to raise a FileListerError
    with patch('ra_aid.file_listing.get_all_project_files', side_effect=FileListerError("Directory not found")):
        # The function should now return an empty list and log the error rather than raising an exception
        result = agent._execute_tool(mock_response)
        assert result == []


def test_fallback_not_needed_for_fuzzy_find():
    """Test that fallback handling is not needed for fuzzy_find_project_files 
    since it now handles errors gracefully."""
    
    # Create a mock agent with the fuzzy_find_project_files tool
    mock_model = MagicMock()
    
    # Create a predefined response for the model.invoke
    function_call = 'fuzzy_find_project_files(search_term="bullet physics", repo_path="/nonexistent/path", threshold=60, max_results=10, include_paths=None, exclude_patterns=None)'
    mock_model.invoke.return_value = AIMessage(content=function_call)
    
    # Create the agent with fallback enabled
    agent = CiaynAgent(
        model=mock_model,
        tools=[fuzzy_find_project_files],
        max_history_messages=5,
        config={"experimental_fallback_handler": True}
    )
    
    # Mock the fallback handler methods
    agent.fallback_handler.handle_failure = MagicMock()
    agent.handle_fallback_response = MagicMock()
    
    # Patch get_all_project_files to raise a FileListerError
    with patch('ra_aid.file_listing.get_all_project_files', side_effect=FileListerError("Directory not found")):
        # Call _execute_tool directly, it should not raise an exception
        result = agent._execute_tool(mock_model.invoke.return_value)
        
        # Verify the result is an empty list
        assert result == []
        
        # Verify that fallback_handler was not called since no exception was raised
        agent.fallback_handler.handle_failure.assert_not_called()
