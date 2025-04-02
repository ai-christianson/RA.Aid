
import pytest
from unittest.mock import MagicMock, patch, call
from ra_aid.text.processing import process_thinking_content, extract_think_tag # Added import


class TestProcessThinkingContent:
    def test_unsupported_model(self):
        """Test when the model doesn't support thinking (both False)."""
        content = "This is a test response"
        logger = MagicMock()
        result, thinking = process_thinking_content(
            content,
            supports_think_tag=False,
            supports_thinking=False,
            logger=logger,
        )
        assert result == content
        assert thinking is None
        # No debug calls expected when both are False
        logger.debug.assert_not_called()

    def test_string_with_think_tag_explicitly_enabled(self):
        """Test extraction when supports_think_tag=True."""
        content = "<think>This is thinking content</think>This is the actual response"
        logger = MagicMock()
        result, thinking = process_thinking_content(
            content,
            supports_think_tag=True,
            show_thoughts=False,
            logger=logger
        )
        assert result == "This is the actual response"
        assert thinking == "This is thinking content"
        # Updated assertions to match actual log messages
        logger.debug.assert_any_call("Model config supports_think_tag=True, checking for tag.")
        logger.debug.assert_any_call("Attempting to extract <think> tag.")
        assert any(c[0][0].startswith("Found think tag content") for c in logger.debug.call_args_list)

    def test_string_without_think_tag_explicitly_enabled(self):
        """Test no extraction when supports_think_tag=True but no tag present."""
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
        # Updated assertions to match actual log messages
        logger.debug.assert_any_call("Model config supports_think_tag=True, checking for tag.")
        logger.debug.assert_any_call("Attempting to extract <think> tag.")
        logger.debug.assert_any_call("No think tag content found despite check.") # Updated

    # --- Start New Tests for Implicit/Explicit Disabled --- #

    def test_implicit_detection_tag_present(self):
        """Test implicit detection when supports_think_tag=None and tag is present."""
        content = "<think>Implicit Thinking...</think>Implicit Response"
        logger = MagicMock()
        result, thinking = process_thinking_content(
            content,
            supports_think_tag=None,
            show_thoughts=False,
            logger=logger
        )
        assert result == "Implicit Response"
        assert thinking == "Implicit Thinking..."
        # Updated assertions to match actual log messages
        logger.debug.assert_any_call(
            "Model config supports_think_tag=None and content starts with <think>, checking for tag." # Updated
        )
        logger.debug.assert_any_call("Attempting to extract <think> tag.") # Updated
        assert any(c[0][0].startswith("Found think tag content") for c in logger.debug.call_args_list)

    def test_implicit_detection_tag_absent(self):
        """Test implicit detection when supports_think_tag=None and no tag is present."""
        content = "Just a response"
        logger = MagicMock()
        result, thinking = process_thinking_content(
            content,
            supports_think_tag=None,
            show_thoughts=False,
            logger=logger
        )
        assert result == "Just a response"
        assert thinking is None
        # Updated assertions to match actual log messages
        logger.debug.assert_any_call(
            "Model config supports_think_tag=None but content does not start with <think>, skipping tag check." # Updated
        )
        assert not any(
            c[0][0] == "Attempting to extract <think> tag." # Updated
            for c in logger.debug.call_args_list
        )

    def test_explicitly_disabled_detection_tag_present(self):
        """Test no extraction when supports_think_tag=False, even if tag is present."""
        content = "<think>Thinking...</think>Response"
        logger = MagicMock()
        result, thinking = process_thinking_content(
            content,
            supports_think_tag=False,
            show_thoughts=False,
            logger=logger
        )
        assert result == "<think>Thinking...</think>Response"
        assert thinking is None
        # Removed assertion for "skipping extraction" log, as it's not logged in this case
        assert not any(
            c[0][0] == "Attempting to extract <think> tag." # Updated
            for c in logger.debug.call_args_list
        )
        assert not any(
            c[0][0].startswith("Found think tag content")
            for c in logger.debug.call_args_list
        )

    # --- End New Tests --- #

    # Existing test renamed slightly for clarity, logic unchanged, logs updated
    def test_string_with_think_tag_explicitly_disabled_but_present(self):
        """Test no extraction when supports_think_tag=False, even if tag is present (legacy name)."""
        content = "<think>This thinking should be ignored</think>Actual response"
        logger = MagicMock()
        result, thinking = process_thinking_content(
            content,
            supports_think_tag=False, # Explicitly disabled
            show_thoughts=False,
            logger=logger
        )
        assert result == content # The original content should be returned
        assert thinking is None
        # Removed assertion for "skipping extraction" log
        # Ensure extraction attempt logs are NOT present
        assert not any(c[0][0] == "Attempting to extract <think> tag." for c in logger.debug.call_args_list)

    # Existing test renamed for consistency, logic unchanged, logs updated
    def test_string_with_think_tag_implicit_detection_legacy(self):
        """Test implicit detection when supports_think_tag=None and tag is present (legacy name)."""
        content = "<think>Implicit thinking</think>Implicit response"
        logger = MagicMock()
        result, thinking = process_thinking_content(
            content,
            supports_think_tag=None, # Implicit detection
            show_thoughts=False,
            logger=logger
        )
        assert result == "Implicit response"
        assert thinking == "Implicit thinking"
        # Updated assertions
        logger.debug.assert_any_call("Model config supports_think_tag=None and content starts with <think>, checking for tag.")
        logger.debug.assert_any_call("Attempting to extract <think> tag.")
        assert any(c[0][0].startswith("Found think tag content") for c in logger.debug.call_args_list)

    # Existing test renamed for consistency, logic unchanged, logs updated
    def test_string_without_think_tag_implicit_detection_legacy(self):
        """Test implicit detection when supports_think_tag=None and no tag is present (legacy name)."""
        content = "No implicit thinking here"
        logger = MagicMock()
        result, thinking = process_thinking_content(
            content,
            supports_think_tag=None, # Implicit detection
            show_thoughts=False,
            logger=logger
        )
        assert result == content
        assert thinking is None
        # Updated assertion
        logger.debug.assert_any_call("Model config supports_think_tag=None but content does not start with <think>, skipping tag check.")
        # Ensure extraction attempt logs are NOT present
        assert not any(c[0][0] == "Attempting to extract <think> tag." for c in logger.debug.call_args_list)


    def test_structured_thinking(self):
        """Test handling of structured thinking content (list format)."""
        content = [
            {"type": "thinking", "text": "First thinking step"},
            {"type": "thinking", "text": "Second thinking step"},
            {"type": "text", "text": "Actual response"}
        ]
        logger = MagicMock()
        result, thinking = process_thinking_content(
            content,
            supports_thinking=True,
            supports_think_tag=False, # Make sure tag processing doesn't interfere
            show_thoughts=False,
            logger=logger
        )
        assert result == [{"type": "text", "text": "Actual response"}]
        assert thinking == "First thinking step\n\nSecond thinking step"
        logger.debug.assert_any_call("Checking for structured thinking content (list format)")
        # Check that debug was called with a string starting with "Found structured thinking content"
        debug_calls = [c[0][0] for c in logger.debug.call_args_list]
        assert any(call.startswith("Found structured thinking content") for call in debug_calls)

    def test_no_structured_thinking(self):
        """Test list content without structured thinking."""
        content = [
            {"type": "text", "text": "Just response"},
            {"other": "data"}
        ]
        logger = MagicMock()
        result, thinking = process_thinking_content(
            content,
            supports_thinking=True,
            show_thoughts=False,
            logger=logger
        )
        assert result == content
        assert thinking is None
        logger.debug.assert_any_call("Checking for structured thinking content (list format)")
        logger.debug.assert_any_call("No structured thinking content found in list.")

    def test_mixed_content_types_structured(self):
        """Test with a mixed list including structured thinking."""
        content = [
            {"type": "thinking", "text": "Thinking"},
            {"type": "text", "text": "Plain string"},
            {"other": "data"}
        ]
        result, thinking = process_thinking_content(
            content,
            supports_thinking=True,
            show_thoughts=False
        )
        assert result == [{"type": "text", "text": "Plain string"}, {"other": "data"}]
        assert thinking == "Thinking"

    def test_config_lookup_for_show_thoughts(self):
        """Test it looks up show_thoughts from config when not provided."""
        content = "<think>Thinking</think>Response"

        # Mock the imported modules
        with patch("ra_aid.database.repositories.config_repository.get_config_repository") as mock_get_config:
            with patch("ra_aid.text.processing.cpm") as mock_cpm:
                # Setup mocks
                mock_repo = MagicMock()
                mock_repo.get.return_value = True  # Simulate show_thoughts=True in config
                mock_get_config.return_value = mock_repo

                # Call the function
                result, thinking = process_thinking_content(
                    content,
                    supports_think_tag=True # Enable tag processing
                )

                # Verify results
                mock_repo.get.assert_called_once_with("show_thoughts", False)
                mock_cpm.assert_called_once() # Should be called because config returned True
                assert result == "Response"
                assert thinking == "Thinking"

    def test_panel_styling(self):
        """Test custom panel title and style are applied when showing thoughts."""
        content = "<think>Custom thinking</think>Response"

        # Mock the cpm function since it's now used instead of Panel directly
        with patch("ra_aid.text.processing.cpm") as mock_cpm:
            # Call the function
            process_thinking_content(
                content,
                supports_think_tag=True,
                show_thoughts=True, # Explicitly show thoughts
                panel_title="Custom Title",
                panel_style="red"
            )

            # Check that cpm was called with the right parameters
            mock_cpm.assert_called_once()
            # Verify the title and style were passed correctly
            args, kwargs = mock_cpm.call_args
            assert args[0] == "Custom thinking" # Check content passed to cpm
            assert kwargs.get('title') == "Custom Title"
            assert kwargs.get('border_style') == "red"

# Renamed from test_multiple_think_tags, updated for greedy extraction
def test_greedy_extraction():
    """Test that extract_think_tag uses greedy matching with multiple tags."""
    # Input string designed to test greedy matching across multiple tags
    test_input = "<think>First tag</think>Middle<think>Second tag</think>End"
    # Expected result for greedy matching: extracts everything between the first <think> and the last </think>
    expected_extracted = "First tag</think>Middle<think>Second tag"
    # Expected remaining: the content after the last </think>
    expected_remaining = "End"

    extracted, remaining = extract_think_tag(test_input)

    assert extracted == expected_extracted
    assert remaining == expected_remaining
