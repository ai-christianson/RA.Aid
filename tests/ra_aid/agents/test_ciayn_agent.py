from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ra_aid.agents.ciayn_agent import CiaynAgent, validate_function_call_pattern


@pytest.fixture
def mock_model():
    """Create a mock language model."""
    model = Mock()
    model.invoke = Mock()
    return model


@pytest.fixture
def agent(mock_model):
    """Create a CiaynAgent instance with mock model."""
    tools = []  # Empty tools list for testing trimming functionality
    return CiaynAgent(mock_model, tools, max_history_messages=3)


def test_trim_chat_history_preserves_initial_messages(agent):
    """Test that initial messages are preserved during trimming."""
    initial_messages = [
        HumanMessage(content="Initial 1"),
        AIMessage(content="Initial 2"),
    ]
    chat_history = [
        HumanMessage(content="Chat 1"),
        AIMessage(content="Chat 2"),
        HumanMessage(content="Chat 3"),
        AIMessage(content="Chat 4"),
    ]

    result = agent._trim_chat_history(initial_messages, chat_history)

    # Verify initial messages are preserved
    assert result[:2] == initial_messages
    # Verify only last 3 chat messages are kept (due to max_history_messages=3)
    assert len(result[2:]) == 3
    assert result[2:] == chat_history[-3:]


def test_trim_chat_history_under_limit(agent):
    """Test trimming when chat history is under the maximum limit."""
    initial_messages = [HumanMessage(content="Initial")]
    chat_history = [HumanMessage(content="Chat 1"), AIMessage(content="Chat 2")]

    result = agent._trim_chat_history(initial_messages, chat_history)

    # Verify no trimming occurred
    assert len(result) == 3
    assert result == initial_messages + chat_history


def test_trim_chat_history_over_limit(agent):
    """Test trimming when chat history exceeds the maximum limit."""
    initial_messages = [HumanMessage(content="Initial")]
    chat_history = [
        HumanMessage(content="Chat 1"),
        AIMessage(content="Chat 2"),
        HumanMessage(content="Chat 3"),
        AIMessage(content="Chat 4"),
        HumanMessage(content="Chat 5"),
    ]

    result = agent._trim_chat_history(initial_messages, chat_history)

    # Verify correct trimming
    assert len(result) == 4  # initial + max_history_messages
    assert result[0] == initial_messages[0]  # Initial message preserved
    assert result[1:] == chat_history[-3:]  # Last 3 chat messages kept


def test_trim_chat_history_empty_initial(agent):
    """Test trimming with empty initial messages."""
    initial_messages = []
    chat_history = [
        HumanMessage(content="Chat 1"),
        AIMessage(content="Chat 2"),
        HumanMessage(content="Chat 3"),
        AIMessage(content="Chat 4"),
    ]

    result = agent._trim_chat_history(initial_messages, chat_history)

    # Verify only last 3 messages are kept
    assert len(result) == 3
    assert result == chat_history[-3:]


def test_trim_chat_history_empty_chat(agent):
    """Test trimming with empty chat history."""
    initial_messages = [
        HumanMessage(content="Initial 1"),
        AIMessage(content="Initial 2"),
    ]
    chat_history = []

    result = agent._trim_chat_history(initial_messages, chat_history)

    # Verify initial messages are preserved and no trimming occurred
    assert result == initial_messages
    assert len(result) == 2


def test_trim_chat_history_token_limit():
    """Test trimming based on token limit."""
    agent = CiaynAgent(Mock(), [], max_history_messages=10, max_tokens=20)

    initial_messages = [HumanMessage(content="Initial")]  # ~2 tokens
    chat_history = [
        HumanMessage(content="A" * 40),  # ~10 tokens
        AIMessage(content="B" * 40),  # ~10 tokens
        HumanMessage(content="C" * 40),  # ~10 tokens
    ]

    result = agent._trim_chat_history(initial_messages, chat_history)

    # Should keep initial message (~2 tokens) and last message (~10 tokens)
    assert len(result) == 2
    assert result[0] == initial_messages[0]
    assert result[1] == chat_history[-1]


def test_trim_chat_history_no_token_limit():
    """Test trimming with no token limit set."""
    agent = CiaynAgent(Mock(), [], max_history_messages=2, max_tokens=None)

    initial_messages = [HumanMessage(content="Initial")]
    chat_history = [
        HumanMessage(content="A" * 1000),
        AIMessage(content="B" * 1000),
        HumanMessage(content="C" * 1000),
    ]

    result = agent._trim_chat_history(initial_messages, chat_history)

    # Should keep initial message and last 2 messages (max_history_messages=2)
    assert len(result) == 3
    assert result[0] == initial_messages[0]
    assert result[1:] == chat_history[-2:]


def test_trim_chat_history_both_limits():
    """Test trimming with both message count and token limits."""
    agent = CiaynAgent(Mock(), [], max_history_messages=3, max_tokens=15)

    initial_messages = [HumanMessage(content="Init")]  # ~1 token
    chat_history = [
        HumanMessage(content="A" * 40),  # ~10 tokens
        AIMessage(content="B" * 40),  # ~10 tokens
        HumanMessage(content="C" * 40),  # ~10 tokens
        AIMessage(content="D" * 40),  # ~10 tokens
    ]

    result = agent._trim_chat_history(initial_messages, chat_history)

    # Should first apply message limit (keeping last 3)
    # Then token limit should further reduce to fit under 15 tokens
    assert len(result) == 2  # Initial message + 1 message under token limit
    assert result[0] == initial_messages[0]
    assert result[1] == chat_history[-1]


class TestFunctionCallValidation:
    @pytest.mark.parametrize(
        "test_input",
        [
            "basic_func()",
            'func_with_arg("test")',
            'complex_func(1, "two", three)',
            'nested_parens(func("test"))',
            "under_score()",
            "with-dash()",
        ],
    )
    def test_valid_function_calls(self, test_input):
        """Test function call patterns that should pass validation."""
        assert not validate_function_call_pattern(test_input)

    @pytest.mark.parametrize(
        "test_input",
        [
            "",
            "Invalid!function()",
            "missing_parens",
            "unmatched(parens))",
            "multiple()calls()",
            "no spaces()()",
        ],
    )
    def test_invalid_function_calls(self, test_input):
        """Test function call patterns that should fail validation."""
        assert validate_function_call_pattern(test_input)

    @pytest.mark.parametrize(
        "test_input",
        [
            "  leading_space()",
            "trailing_space()  ",
            "func   (arg)",
            "func(  spaced args  )",
        ],
    )
    def test_whitespace_handling(self, test_input):
        """Test whitespace variations in function calls."""
        assert not validate_function_call_pattern(test_input)

    @pytest.mark.parametrize(
        "test_input",
        [
            """multiline(
            arg
        )""",
            "func(\n  arg1,\n  arg2\n)",
        ],
    )
    def test_multiline_responses(self, test_input):
        """Test function calls spanning multiple lines."""
        assert not validate_function_call_pattern(test_input)
