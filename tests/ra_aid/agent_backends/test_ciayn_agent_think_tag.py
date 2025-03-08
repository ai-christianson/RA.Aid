import pytest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ra_aid.agent_backends.ciayn_agent import CiaynAgent


@pytest.fixture
def mock_get_model():
    """Mock the get_model function to avoid database connection issues."""
    with patch("ra_aid.agent_backends.ciayn_agent.get_model") as mock:
        mock.return_value = MagicMock()
        yield mock


def test_stream_supports_think_tag(mock_get_model):
    """Test that CiaynAgent.stream extracts think tags when the model supports them."""
    # Setup mock model
    mock_model = MagicMock()
    mock_response = AIMessage(content="<think>These are my thoughts</think>Actual response")
    mock_model.invoke.return_value = mock_response
    
    # Setup agent with config that supports think tags and show_thoughts
    config = {
        "provider": "openai-compatible",
        "model": "qwen-qwq-32b",
        "show_thoughts": True
    }
    agent = CiaynAgent(mock_model, [], config=config)
    
    # Mock print_warning and print_error to avoid unwanted console output
    with patch("ra_aid.agent_backends.ciayn_agent.print_warning"), \
         patch("ra_aid.agent_backends.ciayn_agent.print_error"):
        
        # We're not patching console.print to verify it's called with the panel
        # Mock _execute_tool to avoid actually executing tools
        with patch.object(agent, "_execute_tool") as mock_execute:
            mock_execute.return_value = "Tool result"
            
            # For console.print, we want to verify it's called, but not actually print anything
            with patch("rich.console.Console.print") as mock_console_print:
                # Call stream method
                next(agent.stream({"messages": []}, {}))
                
                # Verify console.print was called
                mock_console_print.assert_called()
                
                # Check if the response content was updated to remove the think tag
                assert "Actual response" in mock_execute.call_args[0][0].content
                assert "<think>" not in mock_execute.call_args[0][0].content


def test_stream_no_think_tag_support(mock_get_model):
    """Test that CiaynAgent.stream doesn't extract think tags when not supported."""
    # Setup mock model
    mock_model = MagicMock()
    mock_response = AIMessage(content="<think>These are my thoughts</think>Actual response")
    mock_model.invoke.return_value = mock_response
    
    # Setup agent with config that doesn't support think tags
    config = {
        "provider": "openai",
        "model": "gpt-4"
    }
    agent = CiaynAgent(mock_model, [], config=config)
    
    # Mock print_warning and print_error to avoid unwanted console output
    with patch("ra_aid.agent_backends.ciayn_agent.print_warning"), \
         patch("ra_aid.agent_backends.ciayn_agent.print_error"):
        
        # Mock _execute_tool to avoid actually executing tools
        with patch.object(agent, "_execute_tool") as mock_execute:
            mock_execute.return_value = "Tool result"
            
            # For console.print, we want to patch it to verify Panel with title="💭 Thoughts" is not used
            with patch("ra_aid.agent_backends.ciayn_agent.Panel") as mock_panel:
                # Call stream method
                next(agent.stream({"messages": []}, {}))
                
                # Verify panel was not created with '💭 Thoughts' title
                thoughts_panel_call = None
                for call in mock_panel.call_args_list:
                    args, kwargs = call
                    if kwargs.get("title") == "💭 Thoughts":
                        thoughts_panel_call = call
                        break
                
                assert thoughts_panel_call is None, "A panel with title '💭 Thoughts' was created but should not have been"
                
                # Check that the response content was not modified
                assert "<think>These are my thoughts</think>Actual response" in mock_execute.call_args[0][0].content


def test_stream_with_no_think_tags(mock_get_model):
    """Test that CiaynAgent.stream works properly when no think tags are present."""
    # Setup mock model
    mock_model = MagicMock()
    mock_response = AIMessage(content="Actual response without tags")
    mock_model.invoke.return_value = mock_response
    
    # Setup agent with config that supports think tags and show_thoughts
    config = {
        "provider": "openai-compatible",
        "model": "qwen-qwq-32b",
        "show_thoughts": True
    }
    agent = CiaynAgent(mock_model, [], config=config)
    
    # Mock print_warning and print_error to avoid unwanted console output
    with patch("ra_aid.agent_backends.ciayn_agent.print_warning"), \
         patch("ra_aid.agent_backends.ciayn_agent.print_error"):
        
        # Mock _execute_tool to avoid actually executing tools
        with patch.object(agent, "_execute_tool") as mock_execute:
            mock_execute.return_value = "Tool result"
            
            # For console.print, we want to verify it's not called with a thoughts panel
            with patch("ra_aid.agent_backends.ciayn_agent.Panel") as mock_panel:
                # Call stream method
                next(agent.stream({"messages": []}, {}))
                
                # Verify panel was not created with '💭 Thoughts' title
                thoughts_panel_call = None
                for call in mock_panel.call_args_list:
                    args, kwargs = call
                    if kwargs.get("title") == "💭 Thoughts":
                        thoughts_panel_call = call
                        break
                
                assert thoughts_panel_call is None, "A panel with title '💭 Thoughts' was created but should not have been"
                
                # Check that the response content was not modified
                assert "Actual response without tags" in mock_execute.call_args[0][0].content