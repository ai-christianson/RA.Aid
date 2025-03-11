import unittest
from unittest.mock import MagicMock, patch

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.prebuilt.chat_agent_executor import AgentState

from ra_aid.anthropic_token_limiter import (
    create_token_counter_wrapper,
    estimate_messages_tokens,
    get_model_token_limit,
    state_modifier,
)


class TestAnthropicTokenLimiter(unittest.TestCase):
    def setUp(self):
        from ra_aid.config import DEFAULT_MODEL
        
        self.mock_model = MagicMock(spec=ChatAnthropic)
        self.mock_model.model = DEFAULT_MODEL
        
        # Sample messages for testing
        self.system_message = SystemMessage(content="You are a helpful assistant.")
        self.human_message = HumanMessage(content="Hello, can you help me with a task?")
        self.long_message = HumanMessage(content="A" * 1000)  # Long message to test trimming
        
        # Create more messages for testing
        self.extra_messages = [
            HumanMessage(content=f"Extra message {i}") for i in range(5)
        ]
        
        # Mock state for testing state_modifier with many messages
        self.state = AgentState(
            messages=[self.system_message, self.human_message, self.long_message] + self.extra_messages,
            next=None,
        )

    @patch("ra_aid.anthropic_token_limiter.token_counter")
    def test_create_token_counter_wrapper(self, mock_token_counter):
        from ra_aid.config import DEFAULT_MODEL
        
        # Setup mock return values
        mock_token_counter.return_value = 50
        
        # Create the wrapper
        wrapper = create_token_counter_wrapper(DEFAULT_MODEL)
        
        # Test with BaseMessage objects
        result = wrapper([self.human_message])
        self.assertEqual(result, 50)
        
        # Test with empty list
        result = wrapper([])
        self.assertEqual(result, 0)
        
        # Verify the mock was called with the right parameters
        mock_token_counter.assert_called_with(messages=unittest.mock.ANY, model=DEFAULT_MODEL)

    @patch("ra_aid.anthropic_token_limiter.CiaynAgent._estimate_tokens")
    def test_estimate_messages_tokens(self, mock_estimate_tokens):
        # Setup mock to return different values for different messages
        mock_estimate_tokens.side_effect = lambda msg: 10 if isinstance(msg, SystemMessage) else 20
        
        # Test with multiple messages
        messages = [self.system_message, self.human_message]
        result = estimate_messages_tokens(messages)
        
        # Should be sum of individual token counts (10 + 20)
        self.assertEqual(result, 30)
        
        # Test with empty list
        result = estimate_messages_tokens([])
        self.assertEqual(result, 0)

    @patch("ra_aid.anthropic_token_limiter.create_token_counter_wrapper")
    @patch("ra_aid.anthropic_token_limiter.print_messages_compact")
    def test_state_modifier(self, mock_print, mock_create_wrapper):
        # Setup a proper token counter function that returns integers
        # This function needs to return values that will cause trim_messages to keep only the first message
        def token_counter(msgs):
            # For a single message, return a small token count
            if len(msgs) == 1:
                return 10
            # For two messages (first + one more), return a value under our limit
            elif len(msgs) == 2:
                return 30  # This is under our 40 token remaining budget (50-10)
            # For three messages, return a value just under our limit
            elif len(msgs) == 3:
                return 40  # This is exactly at our 40 token remaining budget (50-10)
            # For four messages, return a value just at our limit
            elif len(msgs) == 4:
                return 40  # This is exactly at our 40 token remaining budget (50-10)
            # For five messages, return a value that exceeds our 40 token budget
            elif len(msgs) == 5:
                return 60  # This exceeds our 40 token budget, forcing only 4 more messages
            # For more messages, return a value over our limit
            else:
                return 100  # This exceeds our limit
        
        # Don't use side_effect here, directly return the function
        mock_create_wrapper.return_value = token_counter
        
        # Call state_modifier with a max token limit of 50
        result = state_modifier(self.state, self.mock_model, max_input_tokens=50)
        
        # Should keep first message and some of the others (up to 5 total)
        self.assertEqual(len(result), 5)  # First message plus four more
        self.assertEqual(result[0], self.system_message)  # First message is preserved
        
        # Verify the wrapper was created with the right model
        mock_create_wrapper.assert_called_with(self.mock_model.model)
        
        # Verify print_messages_compact was called
        mock_print.assert_called_once()

    @patch("ra_aid.anthropic_token_limiter.get_config_repository")
    @patch("litellm.get_model_info")
    def test_get_model_token_limit_from_litellm(self, mock_get_model_info, mock_get_config_repo):
        from ra_aid.config import DEFAULT_MODEL
        
        # Setup mocks
        mock_config = {"provider": "anthropic", "model": DEFAULT_MODEL}
        mock_get_config_repo.return_value.get_all.return_value = mock_config
        
        # Mock litellm's get_model_info to return a token limit
        mock_get_model_info.return_value = {"max_input_tokens": 100000}
        
        # Test getting token limit
        result = get_model_token_limit(mock_config)
        self.assertEqual(result, 100000)
        
        # Verify get_model_info was called with the right model
        mock_get_model_info.assert_called_with(f"anthropic/{DEFAULT_MODEL}")

    @patch("ra_aid.anthropic_token_limiter.get_config_repository")
    @patch("litellm.get_model_info")
    def test_get_model_token_limit_fallback(self, mock_get_model_info, mock_get_config_repo):
        # Setup mocks
        mock_config = {"provider": "anthropic", "model": "claude-2"}
        mock_get_config_repo.return_value.get_all.return_value = mock_config
        
        # Make litellm's get_model_info raise an exception to test fallback
        mock_get_model_info.side_effect = Exception("Model not found")
        
        # Test getting token limit from models_params fallback
        with patch("ra_aid.anthropic_token_limiter.models_params", {
            "anthropic": {
                "claude2": {"token_limit": 100000}
            }
        }):
            result = get_model_token_limit(mock_config)
            self.assertEqual(result, 100000)

    @patch("ra_aid.anthropic_token_limiter.get_config_repository")
    @patch("litellm.get_model_info")
    def test_get_model_token_limit_for_different_agent_types(self, mock_get_model_info, mock_get_config_repo):
        from ra_aid.config import DEFAULT_MODEL
        
        # Setup mocks for different agent types
        mock_config = {
            "provider": "anthropic", 
            "model": DEFAULT_MODEL,
            "research_provider": "openai",
            "research_model": "gpt-4",
            "planner_provider": "anthropic",
            "planner_model": "claude-3-sonnet-20240229"
        }
        mock_get_config_repo.return_value.get_all.return_value = mock_config
        
        # Mock different returns for different models
        def model_info_side_effect(model_name):
            if DEFAULT_MODEL in model_name or "claude-3-7-sonnet" in model_name:
                return {"max_input_tokens": 200000}
            elif "gpt-4" in model_name:
                return {"max_input_tokens": 8192}
            elif "claude-3-sonnet" in model_name:
                return {"max_input_tokens": 100000}
            else:
                raise Exception(f"Unknown model: {model_name}")
                
        mock_get_model_info.side_effect = model_info_side_effect
        
        # Test default agent type
        result = get_model_token_limit(mock_config, "default")
        self.assertEqual(result, 200000)
        
        # Test research agent type
        result = get_model_token_limit(mock_config, "research")
        self.assertEqual(result, 8192)
        
        # Test planner agent type
        result = get_model_token_limit(mock_config, "planner")
        self.assertEqual(result, 100000)


if __name__ == "__main__":
    unittest.main()
