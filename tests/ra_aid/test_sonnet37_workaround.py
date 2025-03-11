import pytest
from unittest.mock import MagicMock, patch

from ra_aid.agent_utils import run_agent_with_retry
from ra_aid.agent_context import reset_completion_flags


# Create a mock APIError class for testing
class MockAPIError(Exception):
    """Mock version of Anthropic's APIError for testing."""
    pass


class TestSonnet37Workaround:
    """Test suite for the automatic Claude 3.7 Sonnet thinking block error workaround."""

    def test_automatic_workaround_applied(self):
        """Test that the workaround is automatically applied when the specific error occurs."""
        # Mock dependencies
        mock_agent = MagicMock()
        
        # Create a mock error that simulates the thinking block error
        thinking_error = MockAPIError("400 Bad Request: messages.1.content.0.type: Expected thinking or redacted_thinking, but found text")
        
        # Set up the run_agent_stream to first raise the error, then succeed
        mock_run_stream = MagicMock()
        mock_run_stream.side_effect = [
            thinking_error,  # First call raises error
            None,  # Second call succeeds
        ]
        
        # Mock config repository
        mock_config = {
            "provider": "anthropic",
            "model": "claude-3-7-sonnet-20250219",
        }
        
        with patch("ra_aid.agent_utils.APIError", MockAPIError):
            with patch("ra_aid.agent_utils.get_config_repository") as mock_get_config:
                # Create a mock repository that returns our test config
                mock_repo = MagicMock()
                mock_repo.get_all.return_value = mock_config
                mock_repo.get.side_effect = lambda key, default=None: mock_config.get(key, default)
                mock_repo.set = MagicMock()
                mock_get_config.return_value = mock_repo
                
                # Mock other dependencies to prevent actual execution
                with patch("ra_aid.agent_utils._run_agent_stream", side_effect=mock_run_stream.side_effect):
                    with patch("ra_aid.agent_utils._execute_test_command_wrapper") as mock_test_cmd:
                        # Mock the test command wrapper to return a tuple indicating success
                        mock_test_cmd.return_value = (True, "", False, 0)  # (should_break, prompt, auto_test, test_attempts)
                        
                        # Run the function
                        result = run_agent_with_retry(mock_agent, "Test prompt")
                        
                        # Verify the workaround was applied
                        mock_repo.set.assert_any_call("disable_thinking", True)
                        
                        # The result might be None since we're mocking _run_agent_stream
                        # Just verify that the workaround was applied
                        assert mock_repo.set.call_count > 0

    def test_skip_sonnet37_workaround(self):
        """Test that the workaround is not applied when skip_sonnet37_workaround is True."""
        # Mock dependencies
        mock_agent = MagicMock()
        
        # Create a mock error that simulates the thinking block error
        thinking_error = MockAPIError("400 Bad Request: messages.1.content.0.type: Expected thinking or redacted_thinking, but found text")
        
        # Set up the run_agent_stream to raise the error
        mock_run_stream = MagicMock()
        mock_run_stream.side_effect = thinking_error
        
        # Mock config repository with skip_sonnet37_workaround=True
        mock_config = {
            "provider": "anthropic",
            "model": "claude-3-7-sonnet-20250219",
            "skip_sonnet37_workaround": True,
        }
        
        with patch("ra_aid.agent_utils.APIError", MockAPIError):
            with patch("ra_aid.agent_utils.get_config_repository") as mock_get_config:
                # Create a mock repository that returns our test config
                mock_repo = MagicMock()
                mock_repo.get_all.return_value = mock_config
                mock_repo.get.side_effect = lambda key, default=None: mock_config.get(key, default)
                mock_get_config.return_value = mock_repo
                
                # Mock agent_context.mark_agent_crashed to verify it's called
                with patch("ra_aid.agent_context.mark_agent_crashed") as mock_mark_crashed:
                    # Mock other dependencies to prevent actual execution
                    with patch("ra_aid.agent_utils._run_agent_stream", side_effect=mock_run_stream.side_effect):
                        
                        # Run the function - should crash with unretryable error
                        result = run_agent_with_retry(mock_agent, "Test prompt")
                        
                        # Verify the agent was marked as crashed
                        mock_mark_crashed.assert_called_once()
                        
                        # Verify the function returned a crash message
                        assert "Agent has crashed" in result
                        assert "Unretryable API error" in result

    def test_non_thinking_error_not_handled(self):
        """Test that other 400 errors are not handled by the workaround."""
        # Mock dependencies
        mock_agent = MagicMock()
        
        # Create a mock error that simulates a different 400 error
        other_error = MockAPIError("400 Bad Request: Some other error message")
        
        # Set up the run_agent_stream to raise the error
        mock_run_stream = MagicMock()
        mock_run_stream.side_effect = other_error
        
        # Mock config repository
        mock_config = {
            "provider": "anthropic",
            "model": "claude-3-7-sonnet-20250219",
        }
        
        with patch("ra_aid.agent_utils.APIError", MockAPIError):
            with patch("ra_aid.agent_utils.get_config_repository") as mock_get_config:
                # Create a mock repository that returns our test config
                mock_repo = MagicMock()
                mock_repo.get_all.return_value = mock_config
                mock_repo.get.side_effect = lambda key, default=None: mock_config.get(key, default)
                mock_get_config.return_value = mock_repo
                
                # Mock agent_context.mark_agent_crashed to verify it's called
                with patch("ra_aid.agent_context.mark_agent_crashed") as mock_mark_crashed:
                    # Mock other dependencies to prevent actual execution
                    with patch("ra_aid.agent_utils._run_agent_stream", side_effect=mock_run_stream.side_effect):
                        
                        # Run the function - should crash with unretryable error
                        result = run_agent_with_retry(mock_agent, "Test prompt")
                        
                        # Verify the agent was marked as crashed
                        mock_mark_crashed.assert_called_once()
                        
                        # Verify the function returned a crash message
                        assert "Agent has crashed" in result
                        assert "Unretryable API error" in result
