import pytest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ra_aid.agent_backends.ciayn_agent import CiaynAgent


@pytest.fixture(autouse=True)
def mock_trajectory_repository():
    """Mock the trajectory repository to avoid database connection issues."""
    with patch("ra_aid.callbacks.default_callback_handler.get_trajectory_repository") as mock:
        mock_repo = MagicMock()
        mock.return_value = mock_repo
        yield mock_repo


@pytest.fixture(autouse=True)
def mock_session_repository():
    """Mock the session repository to avoid database connection issues."""
    with patch("ra_aid.callbacks.default_callback_handler.get_session_repository") as mock:
        mock_repo = MagicMock()
        mock_repo.get_current_session_record.return_value = MagicMock(get_id=lambda: 1)
        mock.return_value = mock_repo
        yield mock_repo


def test_stream_supports_think_tag():
    """Test that CiaynAgent.stream extracts think tags when the model supports them."""
    # Setup mock model
    mock_model = MagicMock()
    mock_response = AIMessage(content="<think>These are my thoughts</think>Actual response")
    mock_model.invoke.return_value = mock_response

    # Setup agent with config that supports think tags and show_thoughts
    config = {
        "provider": "openai-compatible",
        "model": "qwen-qwq-32b", # This model has supports_think_tag: True in models_params.py
        "show_thoughts": True
    }
    agent = CiaynAgent(mock_model, [], config=config)

    # Mock print_warning and print_error to avoid unwanted console output
    with patch("ra_aid.console.formatting.print_warning"), \
         patch("ra_aid.console.formatting.print_error"):

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
                # Check the first argument of the first call to mock_execute, which should be the AIMessage prepared for the tool
                assert mock_execute.call_args[0][0].content == "Actual response"
                assert "<think>" not in mock_execute.call_args[0][0].content


def test_stream_implicit_think_tag_support():
    """Test that CiaynAgent.stream extracts think tags implicitly and displays them if show_thoughts is globally True."""
    # Setup mock model
    mock_model = MagicMock()
    mock_response = AIMessage(content="<think>These are my thoughts</think>Actual response")
    mock_model.invoke.return_value = mock_response

    # Setup agent with config where supports_think_tag is absent (implicitly None)
    # And show_thoughts is absent in the agent config (will cause lookup)
    config = {
        "provider": "openai",
        "model": "gpt-4" # This model has no supports_think_tag in models_params.py
    }
    agent = CiaynAgent(mock_model, [], config=config)

    # Mock print_warning and print_error to avoid unwanted console output
    with patch("ra_aid.console.formatting.print_warning"), \
         patch("ra_aid.console.formatting.print_error"):

        # Mock _execute_tool to avoid actually executing tools
        with patch.object(agent, "_execute_tool") as mock_execute:
            mock_execute.return_value = "Tool result"

            # Mock the config repository lookup within process_thinking_content
            # to simulate show_thoughts being globally enabled.
            # Patch where it's imported: ra_aid.database.repositories.config_repository
            with patch("ra_aid.database.repositories.config_repository.get_config_repository") as mock_get_repo, \
                 patch("rich.console.Console.print") as mock_console_print:

                # Configure the mock repository and its get method
                mock_repo_instance = MagicMock()
                # Simulate get("show_thoughts", False) returning True
                mock_repo_instance.get.side_effect = lambda key, default: True if key == "show_thoughts" else default
                mock_get_repo.return_value = mock_repo_instance

                # Call stream method
                next(agent.stream({"messages": []}, {}))

                # Verify console.print was called (because show_thoughts was mocked to True)
                mock_console_print.assert_called()

                # Check that the response content WAS modified due to implicit detection
                assert mock_execute.call_args[0][0].content == "Actual response"


def test_stream_explicitly_disabled_think_tag_support(): # Added in previous step
    """Test that CiaynAgent.stream does NOT extract think tags when explicitly disabled."""
    # Setup mock model
    mock_model = MagicMock()
    mock_response = AIMessage(content="<think>These are my thoughts</think>Actual response")
    mock_model.invoke.return_value = mock_response

    # Setup agent with config where supports_think_tag is explicitly False
    # Use a model that *does* have the tag in models_params.py but override it
    config = {
        "provider": "openai-compatible",
        "model": "qwen-qwq-32b", # This model normally supports it
        "supports_think_tag": False, # Explicitly disable
        # No need to set show_thoughts, as extraction shouldn't happen anyway
    }
    agent = CiaynAgent(mock_model, [], config=config)

    # Mock print_warning and print_error to avoid unwanted console output
    with patch("ra_aid.console.formatting.print_warning"), \
         patch("ra_aid.console.formatting.print_error"):

        # Mock _execute_tool to avoid actually executing tools
        with patch.object(agent, "_execute_tool") as mock_execute:
            mock_execute.return_value = "Tool result"

            # Patch Panel to ensure the thoughts panel is not created
            # Also patch Console.print to avoid noise if panel somehow gets created
            with patch("rich.panel.Panel") as mock_panel, \
                 patch("rich.console.Console.print"):
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

                # Check that the response content was NOT modified because it was explicitly disabled
                assert mock_execute.call_args[0][0].content == "<think>These are my thoughts</think>Actual response" # Correct assertion for this case


def test_stream_with_no_think_tags():
    """Test that CiaynAgent.stream works properly when no think tags are present."""
    # Setup mock model
    mock_model = MagicMock()
    mock_response = AIMessage(content="Actual response without tags")
    mock_model.invoke.return_value = mock_response

    # Setup agent with config that supports think tags and show_thoughts
    config = {
        "provider": "openai-compatible",
        "model": "qwen-qwq-32b", # This model has supports_think_tag: True
        "show_thoughts": True
    }
    agent = CiaynAgent(mock_model, [], config=config)

    # Mock print_warning and print_error to avoid unwanted console output
    with patch("ra_aid.console.formatting.print_warning"), \
         patch("ra_aid.console.formatting.print_error"):

        # Mock _execute_tool to avoid actually executing tools
        with patch.object(agent, "_execute_tool") as mock_execute:
            mock_execute.return_value = "Tool result"

            # For console.print, we want to verify it's not called with a thoughts panel
            with patch("rich.panel.Panel") as mock_panel:
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
                assert mock_execute.call_args[0][0].content == "Actual response without tags"
