import pytest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from ra_aid.agent_utils import run_agent_with_retry


class TestThinkingIntegration:
    """Test suite for the integration of thinking block functionality in run_agent_with_retry."""

    @pytest.mark.parametrize(
        "test_id, config, model_name, should_ensure_thinking, description",
        [
            # Test case 1: Claude 3.7 model with thinking enabled should ensure thinking blocks
            (
                "claude_thinking_enabled",
                {
                    "provider": "anthropic",
                    "model": "claude-3-7-sonnet-20250219",
                },
                "claude-3-7-sonnet-20250219",
                True,
                "Claude 3.7 should ensure thinking blocks",
            ),
            # Test case 2: Claude 3.7 model with thinking disabled should not ensure thinking blocks
            (
                "claude_thinking_disabled",
                {
                    "provider": "anthropic",
                    "model": "claude-3-7-sonnet-20250219",
                    "disable_thinking": True,
                },
                "claude-3-7-sonnet-20250219",
                False,
                "Claude 3.7 with thinking disabled should not ensure thinking blocks",
            ),
            # Test case 3: Non-Claude 3.7 model should not ensure thinking blocks
            (
                "non_claude_model",
                {
                    "provider": "openai",
                    "model": "gpt-4",
                },
                "gpt-4",
                False,
                "Non-Claude model should not ensure thinking blocks",
            ),
        ],
    )
    def test_run_agent_with_retry_thinking_integration(
        self, test_id, config, model_name, should_ensure_thinking, description
    ):
        """Test that run_agent_with_retry correctly integrates thinking block functionality."""
        # Mock the necessary dependencies
        with patch("ra_aid.agent_utils.get_config_repository") as mock_get_config:
            with patch("ra_aid.agent_utils._ensure_thinking_block") as mock_ensure_thinking:
                with patch("ra_aid.agent_utils._run_agent_stream") as mock_run_agent_stream:
                    with patch("ra_aid.agent_utils._setup_interrupt_handling") as mock_setup:
                        with patch("ra_aid.agent_utils._restore_interrupt_handling"):
                            with patch("ra_aid.agent_utils.agent_context"):
                                with patch("ra_aid.agent_utils.InterruptibleSection"):
                                    with patch("ra_aid.agent_utils.is_anthropic_claude") as mock_is_anthropic:
                                        with patch("ra_aid.agent_utils.models_params") as mock_models_params:
                                            # Set up the mocks
                                            mock_get_config.return_value.get_all.return_value = config
                                            mock_get_config.return_value.get.return_value = False
                                            mock_setup.return_value = None
                                            mock_run_agent_stream.return_value = True
                                            
                                            # Set up is_anthropic_claude to return True for Claude models
                                            mock_is_anthropic.return_value = "claude" in model_name.lower()
                                            
                                            # Set up models_params to return supports_thinking=True for Claude 3.7 models
                                            if "claude-3-7" in model_name:
                                                mock_models_params.get.return_value = {
                                                    model_name: {"supports_thinking": True}
                                                }
                                            else:
                                                mock_models_params.get.return_value = {}
                                            
                                            # Create a mock agent
                                            mock_agent = MagicMock()
                                            
                                            # Call the function
                                            run_agent_with_retry(mock_agent, "Test prompt")
                                            
                                            # Check if _ensure_thinking_block was called correctly
                                            if should_ensure_thinking:
                                                mock_ensure_thinking.assert_called()
                                            else:
                                                mock_ensure_thinking.assert_not_called()
