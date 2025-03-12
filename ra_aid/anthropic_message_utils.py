"""Utilities for handling Anthropic-specific message formats and trimming."""

from typing import Callable, List, Literal, Optional, Sequence, Union, cast

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


def _is_message_type(
    message: BaseMessage, message_types: Union[str, type, List[Union[str, type]]]
) -> bool:
    """Check if a message is of a specific type or types.

    Args:
        message: The message to check
        message_types: Type(s) to check against (string name or class)

    Returns:
        bool: True if message matches any of the specified types
    """
    if not isinstance(message_types, list):
        message_types = [message_types]

    types_str = [t for t in message_types if isinstance(t, str)]
    types_classes = tuple(t for t in message_types if isinstance(t, type))

    return message.type in types_str or isinstance(message, types_classes)


def has_tool_use(message: BaseMessage) -> bool:
    """Check if a message contains tool use.

    Args:
        message: The message to check

    Returns:
        bool: True if the message contains tool use
    """
    if not isinstance(message, AIMessage):
        return False

    # Check content for tool_use
    if isinstance(message.content, str) and "tool_use" in message.content:
        return True

    # Check content list for tool_use blocks
    if isinstance(message.content, list):
        for item in message.content:
            if isinstance(item, dict) and item.get("type") == "tool_use":
                return True

    # Check additional_kwargs for tool_calls
    if hasattr(message, "additional_kwargs") and message.additional_kwargs.get(
        "tool_calls"
    ):
        return True

    return False


def is_tool_pair(message1: BaseMessage, message2: BaseMessage) -> bool:
    """Check if two messages form a tool use/result pair.

    Args:
        message1: First message
        message2: Second message

    Returns:
        bool: True if the messages form a tool use/result pair
    """
    return (
        isinstance(message1, AIMessage)
        and isinstance(message2, ToolMessage)
        and has_tool_use(message1)
    )



def anthropic_trim_messages(
    messages: Sequence[BaseMessage],
    *,
    max_tokens: int,
    token_counter: Callable[[List[BaseMessage]], int],
    strategy: Literal["first", "last"] = "last",
    num_messages_to_keep: int = 2,
    allow_partial: bool = False,
    include_system: bool = True,
    start_on: Optional[Union[str, type, List[Union[str, type]]]] = None,
) -> List[BaseMessage]:
    """Trim messages to fit within a token limit, with Anthropic-specific handling.

    Warning - not fully implemented - last strategy is supported and test, not
    allow partial, not 'first' strategy either.
    This function is similar to langchain_core's trim_messages but with special
    handling for Anthropic message formats to avoid API errors.

    It always keeps the first num_messages_to_keep messages.

    Args:
        messages: Sequence of messages to trim
        max_tokens: Maximum number of tokens allowed
        token_counter: Function to count tokens in messages
        strategy: Whether to keep the "first" or "last" messages
        allow_partial: Whether to allow partial messages
        include_system: Whether to always include the system message
        start_on: Message type to start on (only for "last" strategy)

    Returns:
        List[BaseMessage]: Trimmed messages that fit within token limit
    """
    if not messages:
        return []

    messages = list(messages)

    # Always keep the first num_messages_to_keep messages
    kept_messages = messages[:num_messages_to_keep]
    remaining_msgs = messages[num_messages_to_keep:]


    # For Anthropic, we need to maintain the conversation structure where:
    # 1. Every AIMessage with tool_use must be followed by a ToolMessage
    # 2. Every AIMessage that follows a ToolMessage must start with a tool_result

    # First, check if we have any tool_use in the messages
    has_tool_use_anywhere = any(has_tool_use(msg) for msg in messages)

    # If we have tool_use anywhere, we need to be very careful about trimming
    if has_tool_use_anywhere:
        # For safety, just keep all messages if we're under the token limit
        if token_counter(messages) <= max_tokens:
            return messages

        # We need to identify all tool_use/tool_result relationships
        # First, find all AIMessage+ToolMessage pairs
        pairs = []
        i = 0
        while i < len(messages) - 1:
            if is_tool_pair(messages[i], messages[i + 1]):
                pairs.append((i, i + 1))
                i += 2
            else:
                i += 1

        # For Anthropic, we need to ensure that:
        # 1. If we include an AIMessage with tool_use, we must include the following ToolMessage
        # 2. If we include a ToolMessage, we must include the preceding AIMessage with tool_use

        # The safest approach is to always keep complete AIMessage+ToolMessage pairs together
        # First, identify all complete pairs
        complete_pairs = []
        for start, end in pairs:
            complete_pairs.append((start, end))

        # Now we'll build our result, starting with the kept_messages
        # But we need to be careful about the first message if it has tool_use
        result = []

        # Check if the last message in kept_messages has tool_use
        if (
            kept_messages
            and isinstance(kept_messages[-1], AIMessage)
            and has_tool_use(kept_messages[-1])
        ):
            # We need to find the corresponding ToolMessage
            for i, (ai_idx, tool_idx) in enumerate(pairs):
                if messages[ai_idx] is kept_messages[-1]:
                    # Found the pair, add all kept_messages except the last one
                    result.extend(kept_messages[:-1])
                    # Add the AIMessage and ToolMessage as a pair
                    result.extend([messages[ai_idx], messages[tool_idx]])
                    # Remove this pair from the list of pairs to process later
                    pairs = pairs[:i] + pairs[i + 1 :]
                    break
            else:
                # If we didn't find a matching pair, just add all kept_messages
                result.extend(kept_messages)
        else:
            # No tool_use in the last kept message, just add all kept_messages
            result.extend(kept_messages)

        # If we're using the "last" strategy, we'll try to include pairs from the end
        if strategy == "last":
            # First collect all pairs we can include within the token limit
            pairs_to_include = []

            # Process pairs from the end (newest first)
            for pair_idx, (ai_idx, tool_idx) in enumerate(reversed(complete_pairs)):
                # Try adding this pair
                test_msgs = result.copy()

                # Add all previously selected pairs
                for prev_ai_idx, prev_tool_idx in pairs_to_include:
                    test_msgs.extend([messages[prev_ai_idx], messages[prev_tool_idx]])

                # Add this pair
                test_msgs.extend([messages[ai_idx], messages[tool_idx]])

                if token_counter(test_msgs) <= max_tokens:
                    # This pair fits, add it to our list
                    pairs_to_include.append((ai_idx, tool_idx))
                else:
                    # This pair would exceed the token limit
                    break

            # Now add the pairs in the correct order
            # Sort by index to maintain the original conversation flow
            pairs_to_include.sort(key=lambda x: x[0])
            for ai_idx, tool_idx in pairs_to_include:
                result.extend([messages[ai_idx], messages[tool_idx]])

        # No need to sort - we've already added messages in the correct order

        return result

    # If no tool_use, proceed with normal segmentation
    segments = []
    i = 0

    # Group messages into segments
    while i < len(remaining_msgs):
        segments.append([remaining_msgs[i]])
        i += 1

    # Now we have segments that maintain the required structure
    # We'll add segments from the end (for "last" strategy) or beginning (for "first")
    # until we hit the token limit

    if strategy == "last":
        # If we have no segments, just return kept_messages
        if not segments:
            return kept_messages

        result = []

        # Process segments from the end
        for i, segment in enumerate(reversed(segments)):
            # Try adding this segment
            test_msgs = segment + result

            if token_counter(kept_messages + test_msgs) <= max_tokens:
                result = segment + result
            else:
                # This segment would exceed the token limit
                break

        final_result = kept_messages + result

        # For Anthropic, we need to ensure the conversation follows a valid structure
        # We'll do a final check of the entire conversation

        # Validate the conversation structure
        valid_result = []
        i = 0

        # Process messages in order
        while i < len(final_result):
            current_msg = final_result[i]

            # If this is an AIMessage with tool_use, it must be followed by a ToolMessage
            if (
                i < len(final_result) - 1
                and isinstance(current_msg, AIMessage)
                and has_tool_use(current_msg)
            ):
                if isinstance(final_result[i + 1], ToolMessage):
                    # This is a valid tool_use + tool_result pair
                    valid_result.append(current_msg)
                    valid_result.append(final_result[i + 1])
                    i += 2
                else:
                    # Invalid: AIMessage with tool_use not followed by ToolMessage
                    # Skip this message to maintain valid structure
                    i += 1
            else:
                # Regular message, just add it
                valid_result.append(current_msg)
                i += 1

        # Final check: don't end with an AIMessage that has tool_use
        if (
            valid_result
            and isinstance(valid_result[-1], AIMessage)
            and has_tool_use(valid_result[-1])
        ):
            valid_result.pop()  # Remove the last message

        return valid_result

    elif strategy == "first":
        result = []

        # Process segments from the beginning
        for i, segment in enumerate(segments):
            # Try adding this segment
            test_msgs = result + segment
            if token_counter(kept_messages + test_msgs) <= max_tokens:
                result = result + segment
            else:
                # This segment would exceed the token limit
                break

        final_result = kept_messages + result

        return final_result
