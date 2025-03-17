"""Unit tests for temperature handling in LLM initialization."""

import unittest
from unittest.mock import patch, MagicMock

from ra_aid.llm import get_model_default_temperature, create_llm_client


class TestTemperatureHandling(unittest.TestCase):
    """Tests for temperature handling in the LLM module."""

    # Define a helper function to create mock models_params
    def _mock_models_params(self):
        return {
            "anthropic": {
                "claude3": {
                    "default_temperature": 0.8,
                    "supports_temperature": True,
                },
                "claude-3-7-sonnet-20250219": {
                    "default_temperature": 1.0,
                    "supports_temperature": True,
                    "supports_thinking": True,
                },
            },
            "openai": {
                "gpt-4": {
                    "default_temperature": 0.6,
                    "supports_temperature": True,
                },
            },
        }

    def test_get_model_default_temperature(self):
        """Test that get_model_default_temperature returns the correct default temperature."""
        # Mock the models_params and DEFAULT_TEMPERATURE
        mock_models_params = self._mock_models_params()
        
        with patch("ra_aid.models_params.models_params", mock_models_params), \
             patch("ra_aid.models_params.DEFAULT_TEMPERATURE", 0.7):
            
            # Test with explicitly defined default_temperature
            self.assertEqual(get_model_default_temperature("anthropic", "claude3"), 0.8)
            self.assertEqual(get_model_default_temperature("anthropic", "claude-3-7-sonnet-20250219"), 1.0)
            self.assertEqual(get_model_default_temperature("openai", "gpt-4"), 0.6)
            
            # Test with undefined default_temperature (should return DEFAULT_TEMPERATURE)
            self.assertEqual(get_model_default_temperature("anthropic", "nonexistent-model"), 0.7)
            self.assertEqual(get_model_default_temperature("nonexistent-provider", "any-model"), 0.7)

    def test_create_llm_client_temperature_handling(self):
        """Test that create_llm_client handles temperature settings correctly."""
        # Mock the models_params
        mock_models_params = self._mock_models_params()
        
        # Setup mocks
        with patch("ra_aid.llm.models_params", mock_models_params), \
             patch("ra_aid.models_params.models_params", mock_models_params), \
             patch("ra_aid.models_params.DEFAULT_TEMPERATURE", 0.7), \
             patch("ra_aid.llm.cpm") as mock_cpm, \
             patch("ra_aid.llm.get_provider_config") as mock_get_provider_config, \
             patch("ra_aid.llm.ChatAnthropic") as mock_chat_anthropic, \
             patch("ra_aid.llm.LLM_REQUEST_TIMEOUT", 180), \
             patch("ra_aid.llm.LLM_MAX_RETRIES", 5):
            
            mock_get_provider_config.return_value = {"api_key": "fake-key"}
            mock_chat_anthropic.return_value = MagicMock()
            
            # Test 1: When temperature is explicitly provided
            create_llm_client("anthropic", "claude3", temperature=0.5)
            mock_chat_anthropic.assert_called_with(
                api_key="fake-key",
                model_name="claude3",
                timeout=180,
                max_retries=5,
                temperature=0.5,
            )
            mock_cpm.assert_not_called()
            mock_chat_anthropic.reset_mock()
            mock_cpm.reset_mock()
            
            # Test 2: When temperature is None, should use model's default_temperature
            create_llm_client("anthropic", "claude3", temperature=None)
            mock_chat_anthropic.assert_called_with(
                api_key="fake-key",
                model_name="claude3",
                timeout=180,
                max_retries=5,
                temperature=0.8,  # Should use claude3's default_temperature
            )
            mock_cpm.assert_called_once()
            mock_chat_anthropic.reset_mock()
            mock_cpm.reset_mock()
            
            # Test 3: Claude 3.7 with thinking enabled
            create_llm_client("anthropic", "claude-3-7-sonnet-20250219", temperature=None)
            # Should apply both temperature and thinking kwargs
            mock_chat_anthropic.assert_called_with(
                api_key="fake-key",
                model_name="claude-3-7-sonnet-20250219",
                timeout=180,
                max_retries=5,
                temperature=1.0,  # Should use claude-3-7's default_temperature (1.0)
                thinking={"type": "enabled", "budget_tokens": 12000},
                max_tokens=64000,
            )
        
    def test_create_llm_client_without_default_temperature(self):
        """Test handling when model doesn't have default_temperature in models_params."""
        # Create a mock models_params without default_temperature
        mock_models_params = {
            "anthropic": {
                "claude3": {
                    "supports_temperature": True,
                },
            },
        }
        
        # Setup mocks
        with patch("ra_aid.llm.models_params", mock_models_params), \
             patch("ra_aid.models_params.models_params", mock_models_params), \
             patch("ra_aid.models_params.DEFAULT_TEMPERATURE", 0.7), \
             patch("ra_aid.llm.cpm") as mock_cpm, \
             patch("ra_aid.llm.get_provider_config") as mock_get_provider_config, \
             patch("ra_aid.llm.ChatAnthropic") as mock_chat_anthropic, \
             patch("ra_aid.llm.LLM_REQUEST_TIMEOUT", 180), \
             patch("ra_aid.llm.LLM_MAX_RETRIES", 5):
            
            mock_get_provider_config.return_value = {"api_key": "fake-key"}
            mock_chat_anthropic.return_value = MagicMock()
            
            # Test: Should use DEFAULT_TEMPERATURE when model doesn't have default_temperature
            create_llm_client("anthropic", "claude3", temperature=None)
            mock_chat_anthropic.assert_called_with(
                api_key="fake-key",
                model_name="claude3",
                timeout=180,
                max_retries=5,
                temperature=0.7,  # Should use DEFAULT_TEMPERATURE
            )
            mock_cpm.assert_called_once()


if __name__ == "__main__":
    unittest.main()
