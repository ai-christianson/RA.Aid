import pytest
from unittest.mock import MagicMock, patch
from ra_aid.text.processing import process_thinking_content


class TestProcessThinkingContent:
    def test_unsupported_model(self):
        """Test when the model doesn't support thinking."""
        content = "This is a test response"
        result, thinking = process_thinking_content(content, supports_think_tag=False, supports_thinking=False)
        assert result == content
        assert thinking is None

    def test_string_with_think_tag(self):
        """Test extraction of think tags from string content."""
        content = "<think>This is thinking content</think>This is the actual response"
        result, thinking = process_thinking_content(
            content, 
            supports_think_tag=True, 
            show_thoughts=False,
            logger=MagicMock()
        )
        assert result == "This is the actual response"
        assert thinking == "This is thinking content"

    def test_string_without_think_tag(self):
        """Test handling of string content without think tags."""
        content = "This is a response without thinking"
        logger = MagicMock()
        result, thinking = process_thinking_content(
            content, 
            supports_think_tag=True, 
            show_thoughts=False,
            logger=logger
        )
        assert result == content
        assert thinking is None
        logger.debug.assert_any_call("Checking for think tags in response")
        logger.debug.assert_any_call("No think tag content found in response")

    def test_structured_thinking(self):
        """Test handling of structured thinking content (list format)."""
        content = [
            {"type": "thinking", "text": "First thinking step"},
            {"type": "thinking", "text": "Second thinking step"},
            {"text": "Actual response"}
        ]
        logger = MagicMock()
        result, thinking = process_thinking_content(
            content, 
            supports_thinking=True, 
            show_thoughts=False,
            logger=logger
        )
        assert result == [{"text": "Actual response"}]
        assert thinking == "First thinking step\n\nSecond thinking step"
        # Check that debug was called with a string starting with "Found structured thinking content"
        debug_calls = [call[0][0] for call in logger.debug.call_args_list]
        assert any(call.startswith("Found structured thinking content") for call in debug_calls)

    def test_mixed_content_types(self):
        """Test with a mixed list of different content types."""
        content = [
            {"type": "thinking", "text": "Thinking"},
            "Plain string",
            {"other": "data"}
        ]
        result, thinking = process_thinking_content(
            content, 
            supports_thinking=True, 
            show_thoughts=False
        )
        assert result == ["Plain string", {"other": "data"}]
        assert thinking == "Thinking"

    def test_config_lookup(self):
        """Test it looks up show_thoughts from config when not provided."""
        content = "<think>Thinking</think>Response"
        
        # Mock the imported modules
        with patch("ra_aid.database.repositories.config_repository.get_config_repository") as mock_get_config:
            with patch("rich.panel.Panel") as mock_panel:
                with patch("rich.markdown.Markdown") as mock_markdown:
                    with patch("rich.console.Console") as mock_console:
                        # Setup mocks
                        mock_repo = MagicMock()
                        mock_repo.get.return_value = True
                        mock_get_config.return_value = mock_repo
                        mock_console_instance = MagicMock()
                        mock_console.return_value = mock_console_instance
                        
                        # Call the function
                        result, thinking = process_thinking_content(
                            content, 
                            supports_think_tag=True
                        )
                        
                        # Verify results
                        mock_repo.get.assert_called_once_with("show_thoughts", False)
                        mock_console_instance.print.assert_called_once()
                        mock_panel.assert_called_once()
                        mock_markdown.assert_called_once()
                        assert result == "Response"
                        assert thinking == "Thinking"

    def test_panel_styling(self):
        """Test custom panel title and style are applied."""
        content = "<think>Custom thinking</think>Response"
        
        # Mock the imported modules
        with patch("rich.panel.Panel") as mock_panel:
            with patch("rich.markdown.Markdown"):
                with patch("rich.console.Console") as mock_console:
                    # Setup mock
                    mock_console_instance = MagicMock()
                    mock_console.return_value = mock_console_instance
                    
                    # Call the function
                    process_thinking_content(
                        content, 
                        supports_think_tag=True,
                        show_thoughts=True,
                        panel_title="Custom Title",
                        panel_style="red"
                    )
                    
                    # Check that Panel was called with the right kwargs
                    _, kwargs = mock_panel.call_args
                    assert kwargs["title"] == "Custom Title"
                    assert kwargs["border_style"] == "red"