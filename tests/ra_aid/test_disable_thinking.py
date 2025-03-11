import pytest
from unittest.mock import MagicMock, patch

from ra_aid.llm import create_llm_client


class TestDisableThinking:
    """Test suite for the disable_thinking configuration option."""

    @pytest.mark.parametrize(
        "test_id, config, model_config, expected_thinking_param, description",
        [
            # Test case 1: Claude 3.7 model with thinking enabled (default)
            (
                "claude_thinking_enabled",
                {
                    "provider": "anthropic",
                    "model": "claude-3-7-sonnet-20250219",
                },
                {"supports_thinking": True},
                {"thinking": {"type": "enabled", "budget_tokens": 12000}},
                "Claude 3.7 should have thinking enabled by default",
            ),
            # Test case 2: Claude 3.7 model with thinking explicitly disabled
            (
                "claude_thinking_disabled",
                {
                    "provider": "anthropic",
                    "model": "claude-3-7-sonnet-20250219",
                    "disable_thinking": True,
                },
                {"supports_thinking": True},
                {},
                "Claude 3.7 with disable_thinking=True should not have thinking param",
            ),
            # Test case 3: Non-thinking model should not have thinking param
            (
                "non_thinking_model",
                {
                    "provider": "anthropic",
                    "model": "claude-3-5-sonnet-20240620",
                },
                {"supports_thinking": False},
                {},
                "Non-thinking model should not have thinking param",
            ),
            # Test case 4: Non-Claude model should not have thinking param
            (
                "non_claude_model",
                {
                    "provider": "openai",
                    "model": "gpt-4",
                },
                {},
                {},
                "Non-Claude model should not have thinking param",
            ),
        ],
    )
    def test_disable_thinking_option(
        self, test_id, config, model_config, expected_thinking_param, description
    ):
        """Test that the disable_thinking option correctly controls thinking mode."""
        # Mock the necessary dependencies
        with patch("ra_aid.llm.ChatAnthropic") as mock_anthropic:
            with patch("ra_aid.llm.ChatOpenAI") as mock_openai:
                with patch("ra_aid.llm.models_params") as mock_models_params:
                    # Set up the mock to return the specified model_config
                    mock_models_params.get.return_value = {config["model"]: model_config}
                    
                    # Set up the mock for get_provider_config
                    with patch("ra_aid.llm.get_provider_config") as mock_get_provider_config:
                        # Include disable_thinking in the provider config if it's in the test config
                        provider_config = {
                            "api_key": "test-key",
                            "base_url": None,
                        }
                        if "disable_thinking" in config:
                            provider_config["disable_thinking"] = config["disable_thinking"]
                        
                        mock_get_provider_config.return_value = provider_config
                        
                        # Call the function without passing disable_thinking directly
                        create_llm_client(
                            config["provider"],
                            config["model"],
                            temperature=None,
                            is_expert=False
                        )
                        
                        # Check if the correct parameters were passed
                        if config["provider"] == "anthropic":
                            # Get the kwargs passed to ChatAnthropic
                            _, kwargs = mock_anthropic.call_args
                            
                            # Check if thinking param was included or not
                            if expected_thinking_param:
                                assert "thinking" in kwargs, f"Test {test_id} failed: {description}"
                                assert kwargs["thinking"] == expected_thinking_param["thinking"], \
                                    f"Test {test_id} failed: Thinking param doesn't match expected value"
                            else:
                                assert "thinking" not in kwargs, \
                                    f"Test {test_id} failed: Thinking param should not be present"
