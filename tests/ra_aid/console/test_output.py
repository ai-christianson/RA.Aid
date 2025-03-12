"""Tests for the output module."""

from unittest.mock import patch, MagicMock

import pytest

from ra_aid.callbacks.anthropic_callback_handler import AnthropicCallbackHandler
from ra_aid.console.output import get_cost_subtitle
from ra_aid.database.repositories.config_repository import get_config_repository


class TestGetCostSubtitle:
    """Tests for the get_cost_subtitle function."""

    @patch("ra_aid.console.output.get_config_repository")
    def test_no_cost_subtitle_when_show_cost_false(self, mock_get_config_repo):
        # Setup
        mock_config_repo = MagicMock()
        mock_config_repo.get.return_value = False
        mock_get_config_repo.return_value = mock_config_repo
        
        cost_cb = AnthropicCallbackHandler("claude-3-opus")
        cost_cb.total_cost = 0.123456
        cost_cb.prompt_tokens = 100
        cost_cb.completion_tokens = 50
        cost_cb.total_tokens = 150
        
        # Test
        result = get_cost_subtitle(cost_cb)
        
        # Verify
        assert result is None
        mock_config_repo.get.assert_called_once_with("show_cost", False)

    @patch("ra_aid.console.output.get_config_repository")
    def test_no_cost_subtitle_when_no_callback(self, mock_get_config_repo):
        # Setup
        mock_config_repo = MagicMock()
        mock_config_repo.get.return_value = True
        mock_get_config_repo.return_value = mock_config_repo
        
        # Test
        result = get_cost_subtitle(None)
        
        # Verify
        assert result is None

    @patch("ra_aid.console.output.get_config_repository")
    def test_displays_detailed_cost_subtitle(self, mock_get_config_repo):
        # Setup
        mock_config_repo = MagicMock()
        mock_config_repo.get.return_value = True
        mock_get_config_repo.return_value = mock_config_repo
        
        cost_cb = AnthropicCallbackHandler("claude-3-opus")
        cost_cb.total_cost = 0.123456
        cost_cb.prompt_tokens = 100
        cost_cb.completion_tokens = 50
        cost_cb.total_tokens = 150
        
        # Test
        result = get_cost_subtitle(cost_cb)
        
        # Verify
        assert "Cost: $0.123456" in result
        assert "Input tokens: 100" in result
        assert "Output tokens: 50" in result
        assert "Total tokens: 150" in result