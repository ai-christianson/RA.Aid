"""Tests for the output module."""

from unittest.mock import patch, MagicMock

import pytest

from ra_aid.callbacks.default_callback_handler import DefaultCallbackHandler
from ra_aid.console.output import get_cost_subtitle
from ra_aid.database.repositories.config_repository import get_config_repository


class TestGetCostSubtitle:
    """Tests for the get_cost_subtitle function."""

    @patch("ra_aid.console.output.get_config_repository")
    @patch("ra_aid.callbacks.default_callback_handler.DefaultCallbackHandler")
    def test_no_cost_subtitle_when_show_cost_false(self, mock_callback_class, mock_get_config_repo):
        # Setup
        mock_config_repo = MagicMock()
        mock_config_repo.get.return_value = False
        mock_get_config_repo.return_value = mock_config_repo
        
        mock_callback = MagicMock()
        mock_callback.total_cost = 0.123456
        mock_callback.total_tokens = 150
        mock_callback_class._instances = {DefaultCallbackHandler: mock_callback}
        
        # Test
        result = get_cost_subtitle()
        
        # Verify
        assert result is None
        mock_config_repo.get.assert_called_once_with("show_cost", False)

    @patch("ra_aid.console.output.get_config_repository")
    @patch("ra_aid.callbacks.default_callback_handler.DefaultCallbackHandler")
    def test_no_cost_subtitle_when_no_callback(self, mock_callback_class, mock_get_config_repo):
        # Setup
        mock_config_repo = MagicMock()
        mock_config_repo.get.return_value = True
        mock_get_config_repo.return_value = mock_config_repo
        
        # No callback instance
        mock_callback_class._instances = {}
        
        # Test
        result = get_cost_subtitle()
        
        # Verify
        assert result is None

    @patch("ra_aid.console.output.get_config_repository")
    def test_displays_detailed_cost_subtitle(self, mock_get_config_repo):
        # Setup
        mock_config_repo = MagicMock()
        mock_config_repo.get.return_value = True
        mock_get_config_repo.return_value = mock_config_repo
        
        # Create a real instance for the _instances dictionary
        mock_callback = MagicMock()
        mock_callback.total_cost = 0.123456
        mock_callback.total_tokens = 150
        mock_callback.session_totals = {"cost": 0.123456, "tokens": 150}
        
        # Directly patch the _instances class attribute
        original_instances = getattr(DefaultCallbackHandler, '_instances', {})
        DefaultCallbackHandler._instances = {DefaultCallbackHandler: mock_callback}
        
        try:
            # Test
            result = get_cost_subtitle()
            
            # Verify
            assert result is not None
            assert "Cost: $0.12" in result
            assert "Tokens: 150" in result
        finally:
            # Restore original _instances to avoid affecting other tests
            DefaultCallbackHandler._instances = original_instances
