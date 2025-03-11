import pytest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from ra_aid.agent_utils import _ensure_thinking_block


class TestEnsureThinkingBlock:
    """Test suite for the _ensure_thinking_block function."""

    @pytest.mark.parametrize(
        "test_id, messages, config, expected_changes, description",
        [
            # Test case 1: Non-Claude 3.7 model should not modify messages
            (
                "non_claude_model",
                [
                    HumanMessage(content="Hello"),
                    AIMessage(content=[{"type": "text", "text": "Hi there"}]),
                ],
                {"provider": "openai", "model": "gpt-4"},
                False,
                "Non-Claude 3.7 model should not modify messages",
            ),
            # Test case 2: Claude 3.7 model with thinking disabled should not modify messages
            (
                "claude_thinking_disabled",
                [
                    HumanMessage(content="Hello"),
                    AIMessage(content=[{"type": "text", "text": "Hi there"}]),
                ],
                {
                    "provider": "anthropic",
                    "model": "claude-3-7-sonnet-20250219",
                    "disable_thinking": True,
                },
                False,
                "Claude 3.7 with thinking disabled should not modify messages",
            ),
            # Test case 3: Claude 3.7 model with thinking enabled but no AI messages should not modify messages
            (
                "claude_no_ai_messages",
                [
                    HumanMessage(content="Hello"),
                    SystemMessage(content="System message"),
                ],
                {"provider": "anthropic", "model": "claude-3-7-sonnet-20250219"},
                False,
                "Claude 3.7 with no AI messages should not modify messages",
            ),
            # Test case 4: Claude 3.7 model with thinking enabled and AI message with text content should log warning
            (
                "claude_ai_text_content",
                [
                    HumanMessage(content="Hello"),
                    AIMessage(content="Text response"),
                ],
                {"provider": "anthropic", "model": "claude-3-7-sonnet-20250219"},
                False,
                "Claude 3.7 with AI message with text content should log warning",
            ),
            # Test case 5: Claude 3.7 model with thinking enabled and AI message with list content but no thinking block
            (
                "claude_ai_no_thinking_block",
                [
                    HumanMessage(content="Hello"),
                    AIMessage(content=[{"type": "text", "text": "Hi there"}]),
                ],
                {"provider": "anthropic", "model": "claude-3-7-sonnet-20250219"},
                True,
                "Claude 3.7 with AI message without thinking block should add one",
            ),
            # Test case 6: Claude 3.7 model with thinking enabled and AI message with list content with thinking block
            (
                "claude_ai_with_thinking_block",
                [
                    HumanMessage(content="Hello"),
                    AIMessage(
                        content=[
                            {"type": "thinking", "thinking": "Let me think..."},
                            {"type": "text", "text": "Hi there"},
                        ]
                    ),
                ],
                {"provider": "anthropic", "model": "claude-3-7-sonnet-20250219"},
                False,
                "Claude 3.7 with AI message with thinking block should not modify it",
            ),
            # Test case 7: Claude 3.7 model with thinking enabled and multiple AI messages
            (
                "claude_multiple_ai_messages",
                [
                    HumanMessage(content="Hello"),
                    AIMessage(content=[{"type": "text", "text": "First response"}]),
                    HumanMessage(content="Follow-up"),
                    AIMessage(content=[{"type": "text", "text": "Second response"}]),
                ],
                {"provider": "anthropic", "model": "claude-3-7-sonnet-20250219"},
                True,
                "Claude 3.7 with multiple AI messages should add thinking blocks to all",
            ),
        ],
    )
    def test_ensure_thinking_block(
        self, test_id, messages, config, expected_changes, description
    ):
        """Test the _ensure_thinking_block function with various inputs."""
        # Mock the logger
        with patch("ra_aid.agent_utils.logger") as mock_logger:
            # Mock the models_params dictionary
            with patch("ra_aid.agent_utils.models_params") as mock_models_params:
                # Set up the mock to return supports_thinking=True for Claude 3.7 models
                if "claude-3-7" in config.get("model", ""):
                    mock_models_params.get.return_value = {
                        config["model"]: {"supports_thinking": True}
                    }
                else:
                    mock_models_params.get.return_value = {}

                # Make a copy of the original messages for comparison
                original_messages = [
                    AIMessage(content=msg.content) if isinstance(msg, AIMessage) else msg
                    for msg in messages
                ]

                # Call the function
                result = _ensure_thinking_block(messages, config)

                # Check if the result is different from the input when expected
                if expected_changes:
                    assert result != original_messages, f"Test {test_id} failed: {description}"
                    
                    # Check that AI messages have thinking blocks
                    for msg in result:
                        if hasattr(msg, "type") and msg.type == "ai":
                            if isinstance(msg.content, list) and len(msg.content) > 0:
                                assert msg.content[0].get("type") == "thinking" or msg.content[0].get("type") == "redacted_thinking", \
                                    f"Test {test_id} failed: AI message should have thinking block"
                else:
                    # Check that the messages were not modified
                    assert result == messages, f"Test {test_id} failed: {description}"

                # Check for warning logs for text content
                if "claude_ai_text_content" == test_id:
                    mock_logger.warning.assert_called_once()
