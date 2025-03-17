import unittest
from unittest.mock import MagicMock, patch
import litellm

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage, 
    HumanMessage, 
    SystemMessage,
    ToolMessage
)
from langgraph.prebuilt.chat_agent_executor import AgentState

from ra_aid.anthropic_token_limiter import (
    create_token_counter_wrapper,
    estimate_messages_tokens,
    get_model_token_limit,
    state_modifier,
    sonnet_35_state_modifier,
    convert_message_to_litellm_format,
    adjust_claude_37_token_limit
)
from ra_aid.anthropic_message_utils import has_tool_use, is_tool_pair
from ra_aid.models_params import models_params


class TestAnthropicTokenLimiter(unittest.TestCase):
    def setUp(self):
        from ra_aid.config import DEFAULT_MODEL
        
        self.mock_model = MagicMock(spec=ChatAnthropic)
        self.mock_model.model = DEFAULT_MODEL
        
        # Sample messages for testing
        self.system_message = SystemMessage(content="You are a helpful assistant.")
        self.human_message = HumanMessage(content="Hello, can you help me with a task?")
        self.ai_message = AIMessage(content="I'd be happy to help! What do you need?")
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
        
        # Create tool-related messages for testing
        self.ai_with_tool_use = AIMessage(
            content="I'll use a tool to help you",
            additional_kwargs={"tool_calls": [{"name": "calculator", "input": {"expression": "2+2"}}]}
        )
        self.tool_message = ToolMessage(
            content="4",
            tool_call_id="tool_call_1",
            name="calculator"
        )

    def test_convert_message_to_litellm_format(self):
        """Test conversion of BaseMessage to litellm format."""
        # Test human message
        human_result = convert_message_to_litellm_format(self.human_message)
        self.assertEqual(human_result["role"], "human")
        self.assertEqual(human_result["content"], "Hello, can you help me with a task?")
        
        # Test system message
        system_result = convert_message_to_litellm_format(self.system_message)
        self.assertEqual(system_result["role"], "system")
        self.assertEqual(system_result["content"], "You are a helpful assistant.")
        
        # Test AI message
        ai_result = convert_message_to_litellm_format(self.ai_message)
        self.assertEqual(ai_result["role"], "ai")
        self.assertEqual(ai_result["content"], "I'd be happy to help! What do you need?")

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
    @patch("ra_aid.anthropic_token_limiter.anthropic_trim_messages")
    def test_state_modifier(self, mock_trim_messages, mock_create_wrapper):
        # Setup a proper token counter function that returns integers
        def token_counter(msgs):
            # Return token count based on number of messages
            return len(msgs) * 10
        
        # Configure the mock to return our token counter
        mock_create_wrapper.return_value = token_counter
        
        # Configure anthropic_trim_messages to return a subset of messages
        mock_trim_messages.return_value = [self.system_message, self.human_message]
        
        # Call state_modifier with a max token limit of 50
        result = state_modifier(self.state, self.mock_model, max_input_tokens=50)
        
        # Should return what anthropic_trim_messages returned
        self.assertEqual(result, [self.system_message, self.human_message])
        
        # Verify the wrapper was created with the right model
        mock_create_wrapper.assert_called_with(self.mock_model.model)
        
        # Verify anthropic_trim_messages was called with the right parameters
        mock_trim_messages.assert_called_once()
        

    def test_state_modifier_with_messages(self):
        """Test that state_modifier correctly trims recent messages while preserving the first message when total tokens > max_tokens."""
        # Create a state with messages
        messages = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="Human message 1"),
            AIMessage(content="AI response 1"),
            HumanMessage(content="Human message 2"),
            AIMessage(content="AI response 2"),
        ]
        state = AgentState(messages=messages)
        model = MagicMock(spec=ChatAnthropic)
        model.model = "claude-3-opus-20240229"

        with patch("ra_aid.anthropic_token_limiter.create_token_counter_wrapper") as mock_wrapper, \
             patch("ra_aid.anthropic_token_limiter.anthropic_trim_messages") as mock_trim:
            # Setup mock to return a fixed token count per message
            mock_wrapper.return_value = lambda msgs: len(msgs) * 100
            # Setup mock to return a subset of messages
            mock_trim.return_value = [messages[0], messages[-2], messages[-1]]
            
            result = state_modifier(state, model, max_input_tokens=250)
            
            # Should return what anthropic_trim_messages returned
            self.assertEqual(len(result), 3)
            self.assertEqual(result[0], messages[0])  # First message preserved
            self.assertEqual(result[-1], messages[-1])  # Last message preserved
        
    def test_sonnet_35_state_modifier(self):
        """Test the sonnet 35 state modifier function."""
        # Create a state with messages
        state = {"messages": [self.system_message, self.human_message, self.ai_message]}
        
        # Test with empty messages
        empty_state = {"messages": []}
        
        # Instead of patching trim_messages which has complex internal logic,
        # we'll directly patch the sonnet_35_state_modifier's call to trim_messages
        with patch("ra_aid.anthropic_token_limiter.trim_messages") as mock_trim:
            # Setup mock to return our desired messages
            mock_trim.return_value = [self.human_message, self.ai_message]
            
            # Test with empty messages
            self.assertEqual(sonnet_35_state_modifier(empty_state), [])
            
            # Test with messages under the limit
            result = sonnet_35_state_modifier(state, max_input_tokens=10000)
        
        # Should keep the first message and call trim_messages for the rest
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], self.system_message)
        self.assertEqual(result[1:], [self.human_message, self.ai_message])
        
        # Verify trim_messages was called with the right parameters
        mock_trim.assert_called_once()
        # We can check some of the key arguments
        call_args = mock_trim.call_args[1]
        # The actual value is based on the token estimation logic, not a hard-coded 9000
        self.assertIn("max_tokens", call_args)
        self.assertEqual(call_args["strategy"], "last")
        self.assertEqual(call_args["strategy"], "last")
        self.assertEqual(call_args["allow_partial"], False)
        self.assertEqual(call_args["include_system"], True)

    @patch("ra_aid.anthropic_token_limiter.get_config_repository")
    @patch("ra_aid.anthropic_token_limiter.get_model_info")
    @patch("ra_aid.anthropic_token_limiter.is_claude_37")
    @patch("ra_aid.anthropic_token_limiter.adjust_claude_37_token_limit")
    def test_get_model_token_limit_from_litellm(self, mock_adjust, mock_is_claude_37, mock_get_model_info, mock_get_config_repo):
        # Use a specific model name instead of DEFAULT_MODEL to avoid test dependency
        model_name = "claude-3-7-sonnet-20250219"
        
        # Setup mocks
        mock_config = {"provider": "anthropic", "model": model_name}
        mock_get_config_repo.return_value.get.side_effect = lambda key, default=None: mock_config.get(key, default)
        
        # Mock litellm's get_model_info to return a token limit
        mock_get_model_info.return_value = {"max_input_tokens": 100000}
        
        # Mock is_claude_37 to return True
        mock_is_claude_37.return_value = True
        
        # Mock adjust_claude_37_token_limit to return the original value
        mock_adjust.return_value = 100000
        
        # Test getting token limit
        result = get_model_token_limit(mock_config)
        self.assertEqual(result, 100000)
        
        # Verify get_model_info was called with the right model
        mock_get_model_info.assert_called_once_with(f"anthropic/{model_name}")
        
        # Verify adjust_claude_37_token_limit was called
        mock_adjust.assert_called_once_with(100000, None)
        
    def test_get_model_token_limit_research(self):
        """Test get_model_token_limit with research provider and model."""
        config = {
            "provider": "openai",
            "model": "gpt-4",
            "research_provider": "anthropic",
            "research_model": "claude-3-7-sonnet-20250219",
        }
        
        with patch("ra_aid.anthropic_token_limiter.get_config_repository") as mock_get_config_repo, \
             patch("ra_aid.anthropic_token_limiter.get_model_info") as mock_get_info, \
             patch("ra_aid.anthropic_token_limiter.adjust_claude_37_token_limit") as mock_adjust:
            mock_get_config_repo.return_value.get.side_effect = lambda key, default=None: config.get(key, default)
            mock_get_info.return_value = {"max_input_tokens": 150000}
            mock_adjust.return_value = 150000
            
            # Call the function to check the return value
            token_limit = get_model_token_limit(config, "research")
            self.assertEqual(token_limit, 150000)
            
            # Verify get_model_info was called with the research model
            mock_get_info.assert_called_once_with("anthropic/claude-3-7-sonnet-20250219")
            # Verify adjust_claude_37_token_limit was called
            mock_adjust.assert_called_once_with(150000, None)

    def test_get_model_token_limit_planner(self):
        """Test get_model_token_limit with planner provider and model."""
        config = {
            "provider": "openai",
            "model": "gpt-4",
            "planner_provider": "deepseek",
            "planner_model": "dsm-1",
        }
        
        with patch("ra_aid.anthropic_token_limiter.get_config_repository") as mock_get_config_repo, \
             patch("ra_aid.anthropic_token_limiter.get_model_info") as mock_get_info, \
             patch("ra_aid.anthropic_token_limiter.adjust_claude_37_token_limit") as mock_adjust:
            mock_get_config_repo.return_value.get.side_effect = lambda key, default=None: config.get(key, default)
            mock_get_info.return_value = {"max_input_tokens": 120000}
            mock_adjust.return_value = 120000
            
            # Call the function to check the return value
            token_limit = get_model_token_limit(config, "planner")
            self.assertEqual(token_limit, 120000)
            
            # Verify get_model_info was called with the planner model
            mock_get_info.assert_called_once_with("deepseek/dsm-1")
            # Verify adjust_claude_37_token_limit was called
            mock_adjust.assert_called_once_with(120000, None)

    @patch("ra_aid.anthropic_token_limiter.get_config_repository")
    @patch("ra_aid.anthropic_token_limiter.get_model_info")
    def test_get_model_token_limit_fallback(self, mock_get_model_info, mock_get_config_repo):
        # Setup mocks
        mock_config = {"provider": "anthropic", "model": "claude-2"}
        mock_get_config_repo.return_value.get.side_effect = lambda key, default=None: mock_config.get(key, default)
        
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
    @patch("ra_aid.anthropic_token_limiter.get_model_info")
    @patch("ra_aid.anthropic_token_limiter.adjust_claude_37_token_limit")
    def test_get_model_token_limit_for_different_agent_types(self, mock_adjust, mock_get_model_info, mock_get_config_repo):
        # Use specific model names instead of DEFAULT_MODEL to avoid test dependency
        claude_model = "claude-3-7-sonnet-20250219"
        
        # Setup mocks for different agent types
        mock_config = {
            "provider": "anthropic", 
            "model": claude_model,
            "research_provider": "openai",
            "research_model": "gpt-4",
            "planner_provider": "anthropic",
            "planner_model": "claude-3-7-opus-20250301"
        }
        mock_get_config_repo.return_value.get.side_effect = lambda key, default=None: mock_config.get(key, default)
        
        # Mock different returns for different models
        def model_info_side_effect(model_name):
            if "claude-3-7-sonnet" in model_name:
                return {"max_input_tokens": 200000}
            elif "gpt-4" in model_name:
                return {"max_input_tokens": 8192}
            elif "claude-3-7-opus" in model_name:
                return {"max_input_tokens": 250000}
            else:
                raise Exception(f"Unknown model: {model_name}")
                
        mock_get_model_info.side_effect = model_info_side_effect
        
        # Mock adjust_claude_37_token_limit to return the same values
        mock_adjust.side_effect = lambda tokens, model: tokens
        
        # Test default agent type
        result = get_model_token_limit(mock_config, "default")
        self.assertEqual(result, 200000)
        mock_get_model_info.assert_called_with(f"anthropic/{claude_model}")
        
        # Reset mock
        mock_get_model_info.reset_mock()
        
        # Test research agent type
        result = get_model_token_limit(mock_config, "research")
        self.assertEqual(result, 8192)
        mock_get_model_info.assert_called_with("openai/gpt-4")
        
        # Reset mock
        mock_get_model_info.reset_mock()
        
        # Test planner agent type
        result = get_model_token_limit(mock_config, "planner")
        self.assertEqual(result, 250000)
        mock_get_model_info.assert_called_with("anthropic/claude-3-7-opus-20250301")
        
    def test_get_model_token_limit_anthropic(self):
        """Test get_model_token_limit with Anthropic model."""
        config = {"provider": "anthropic", "model": "claude-3-7-sonnet-20250219"}
        
        with patch("ra_aid.anthropic_token_limiter.get_config_repository") as mock_get_config_repo, \
             patch("ra_aid.anthropic_token_limiter.models_params") as mock_models_params, \
             patch("litellm.get_model_info") as mock_get_info, \
             patch("ra_aid.anthropic_token_limiter.adjust_claude_37_token_limit") as mock_adjust:
            
            # Setup mocks
            mock_get_config_repo.return_value.get.side_effect = lambda key, default=None: config.get(key, default)
            mock_get_info.side_effect = Exception("Model not found")
            
            # Create a mock models_params with claude-3-7
            mock_models_params_dict = {
                "anthropic": {
                    "claude-3-7-sonnet-20250219": {"token_limit": 200000}
                }
            }
            mock_models_params.__getitem__.side_effect = mock_models_params_dict.__getitem__
            mock_models_params.get.side_effect = mock_models_params_dict.get
            
            # Mock adjust to return the same value
            mock_adjust.return_value = 200000
            
            token_limit = get_model_token_limit(config, "default")
            self.assertEqual(token_limit, 200000)

    def test_get_model_token_limit_openai(self):
        """Test get_model_token_limit with OpenAI model."""
        config = {"provider": "openai", "model": "gpt-4"}
        
        with patch("ra_aid.anthropic_token_limiter.get_config_repository") as mock_get_config_repo:
            mock_get_config_repo.return_value.get.side_effect = lambda key, default=None: config.get(key, default)
            token_limit = get_model_token_limit(config, "default")
            self.assertEqual(token_limit, models_params["openai"]["gpt-4"]["token_limit"])

    def test_get_model_token_limit_unknown(self):
        """Test get_model_token_limit with unknown provider/model."""
        config = {"provider": "unknown", "model": "unknown-model"}
        
        with patch("ra_aid.anthropic_token_limiter.get_config_repository") as mock_get_config_repo:
            mock_get_config_repo.return_value.get.side_effect = lambda key, default=None: config.get(key, default)
            token_limit = get_model_token_limit(config, "default")
            self.assertIsNone(token_limit)

    def test_get_model_token_limit_missing_config(self):
        """Test get_model_token_limit with missing configuration."""
        config = {}
        
        with patch("ra_aid.anthropic_token_limiter.get_config_repository") as mock_get_config_repo:
            mock_get_config_repo.return_value.get.side_effect = lambda key, default=None: config.get(key, default)
            token_limit = get_model_token_limit(config, "default")
            self.assertIsNone(token_limit)

    def test_get_model_token_limit_litellm_success(self):
        """Test get_model_token_limit successfully getting limit from litellm."""
        config = {"provider": "anthropic", "model": "claude-3-7-sonnet-20250219"}

        with patch("ra_aid.anthropic_token_limiter.get_config_repository") as mock_get_config_repo, \
             patch("ra_aid.anthropic_token_limiter.get_model_info") as mock_get_info, \
             patch("ra_aid.anthropic_token_limiter.adjust_claude_37_token_limit") as mock_adjust:
            mock_get_config_repo.return_value.get.side_effect = lambda key, default=None: config.get(key, default)
            mock_get_info.return_value = {"max_input_tokens": 100000}
            mock_adjust.return_value = 100000
            
            # Call the function to check the return value
            token_limit = get_model_token_limit(config, "default")
            self.assertEqual(token_limit, 100000)
            
            # Verify get_model_info was called with the right model
            mock_get_info.assert_called_once_with("anthropic/claude-3-7-sonnet-20250219")
            mock_adjust.assert_called_once_with(100000, None)

    def test_get_model_token_limit_litellm_not_found(self):
        """Test fallback to models_tokens when litellm raises NotFoundError."""
        config = {"provider": "anthropic", "model": "claude-3-7-sonnet-20250219"}

        with patch("ra_aid.anthropic_token_limiter.get_config_repository") as mock_get_config_repo, \
             patch("litellm.get_model_info") as mock_get_info, \
             patch("ra_aid.anthropic_token_limiter.models_params") as mock_models_params, \
             patch("ra_aid.anthropic_token_limiter.adjust_claude_37_token_limit") as mock_adjust:
            
            mock_get_config_repo.return_value.get.side_effect = lambda key, default=None: config.get(key, default)
            mock_get_info.side_effect = litellm.exceptions.NotFoundError(
                message="Model not found", model="claude-3-7-sonnet-20250219", llm_provider="anthropic"
            )
            
            # Create a mock models_params with claude-3-7
            mock_models_params_dict = {
                "anthropic": {
                    "claude-3-7-sonnet-20250219": {"token_limit": 200000}
                }
            }
            mock_models_params.__getitem__.side_effect = mock_models_params_dict.__getitem__
            mock_models_params.get.side_effect = mock_models_params_dict.get
            
            # Mock adjust to return the same value
            mock_adjust.return_value = 200000
            
            token_limit = get_model_token_limit(config, "default")
            self.assertEqual(token_limit, 200000)

    def test_get_model_token_limit_litellm_error(self):
        """Test fallback to models_tokens when litellm raises other exceptions."""
        config = {"provider": "anthropic", "model": "claude-2"}

        with patch("ra_aid.anthropic_token_limiter.get_config_repository") as mock_get_config_repo, \
             patch("litellm.get_model_info") as mock_get_info:
            mock_get_config_repo.return_value.get.side_effect = lambda key, default=None: config.get(key, default)
            mock_get_info.side_effect = Exception("Unknown error")
            token_limit = get_model_token_limit(config, "default")
            self.assertEqual(token_limit, models_params["anthropic"]["claude2"]["token_limit"])

    def test_get_model_token_limit_unexpected_error(self):
        """Test returning None when unexpected errors occur."""
        config = None  # This will cause an attribute error when accessed

        token_limit = get_model_token_limit(config, "default")
        self.assertIsNone(token_limit)
        
    def test_adjust_claude_37_token_limit(self):
        """Test adjust_claude_37_token_limit function."""
        # Create a mock model
        mock_model = MagicMock()
        mock_model.model = "claude-3.7-sonnet"
        mock_model.max_tokens = 4096
        
        # Test with Claude 3.7 model
        result = adjust_claude_37_token_limit(100000, mock_model)
        self.assertEqual(result, 95904)  # 100000 - 4096
        
        # Test with non-Claude 3.7 model
        mock_model.model = "claude-3-opus"
        result = adjust_claude_37_token_limit(100000, mock_model)
        self.assertEqual(result, 100000)  # No adjustment
        
        # Test with None max_input_tokens
        result = adjust_claude_37_token_limit(None, mock_model)
        self.assertIsNone(result)
        
        # Test with None model
        result = adjust_claude_37_token_limit(100000, None)
        self.assertEqual(result, 100000)
        
    def test_has_tool_use(self):
        """Test the has_tool_use function."""
        # Test with regular AI message
        self.assertFalse(has_tool_use(self.ai_message))
        
        # Test with AI message containing tool_use in string content
        ai_with_tool_str = AIMessage(content="I'll use a tool_use to help you")
        self.assertTrue(has_tool_use(ai_with_tool_str))
        
        # Test with AI message containing tool_use in structured content
        ai_with_tool_dict = AIMessage(content=[
            {"type": "text", "text": "I'll use a tool to help you"},
            {"type": "tool_use", "tool_use": {"name": "calculator", "input": {"expression": "2+2"}}}
        ])
        self.assertTrue(has_tool_use(ai_with_tool_dict))
        
        # Test with AI message containing tool_calls in additional_kwargs
        self.assertTrue(has_tool_use(self.ai_with_tool_use))
        
        # Test with non-AI message
        self.assertFalse(has_tool_use(self.human_message))
        
    def test_is_tool_pair(self):
        """Test the is_tool_pair function."""
        # Test with valid tool pair
        self.assertTrue(is_tool_pair(self.ai_with_tool_use, self.tool_message))
        
        # Test with non-tool pair (wrong order)
        self.assertFalse(is_tool_pair(self.tool_message, self.ai_with_tool_use))
        
        # Test with non-tool pair (wrong types)
        self.assertFalse(is_tool_pair(self.ai_message, self.human_message))
        
        # Test with non-tool pair (AI message without tool use)
        self.assertFalse(is_tool_pair(self.ai_message, self.tool_message))
        
    @patch("ra_aid.anthropic_message_utils.has_tool_use")
    def test_anthropic_trim_messages_with_tool_use(self, mock_has_tool_use):
        """Test anthropic_trim_messages with a sequence of messages including tool use."""
        from ra_aid.anthropic_message_utils import anthropic_trim_messages
        
        # Setup mock for has_tool_use to return True for AI messages at even indices
        def side_effect(msg):
            if isinstance(msg, AIMessage) and hasattr(msg, 'test_index'):
                return msg.test_index % 2 == 0  # Even indices have tool use
            return False
            
        mock_has_tool_use.side_effect = side_effect
        
        # Create a sequence of alternating human and AI messages with tool use
        messages = []
        
        # Start with system message
        system_msg = SystemMessage(content="You are a helpful assistant.")
        messages.append(system_msg)
        
        # Add alternating human and AI messages with tool use
        for i in range(8):
            if i % 2 == 0:
                # Human message
                msg = HumanMessage(content=f"Human message {i}")
                messages.append(msg)
            else:
                # AI message, every other one has tool use
                ai_msg = AIMessage(content=f"AI message {i}")
                # Add a test_index attribute to track position
                ai_msg.test_index = i
                messages.append(ai_msg)
                
                # If this AI message has tool use (even index), add a tool message after it
                if i % 4 == 1:  # 1, 5, etc.
                    tool_msg = ToolMessage(
                        content=f"Tool result {i}",
                        tool_call_id=f"tool_call_{i}",
                        name="test_tool"
                    )
                    messages.append(tool_msg)
        
        # Define a token counter that returns a fixed value per message
        def token_counter(msgs):
            return len(msgs) * 1000
            
        # Test with a token limit that will require trimming
        result = anthropic_trim_messages(
            messages,
            token_counter=token_counter,
            max_tokens=5000,  # This will allow 5 messages
            strategy="last",
            allow_partial=False,
            include_system=True,
            num_messages_to_keep=2  # Keep system and first human message
        )
        
        # We should have kept the first 2 messages (system + human)
        self.assertEqual(len(result), 5)  # 2 kept + 3 more that fit in token limit
        self.assertEqual(result[0], system_msg)
        
        # Verify that we don't have any AI messages with tool use that aren't followed by a tool message
        for i in range(len(result) - 1):
            if isinstance(result[i], AIMessage) and mock_has_tool_use(result[i]):
                self.assertTrue(isinstance(result[i+1], ToolMessage), 
                               f"AI message with tool use at index {i} not followed by ToolMessage")


if __name__ == "__main__":
    unittest.main()
