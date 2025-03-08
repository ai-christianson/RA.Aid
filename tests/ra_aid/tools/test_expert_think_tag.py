"""Test the think tag functionality in the expert tool."""

import pytest
from unittest.mock import patch, MagicMock

from ra_aid.text.processing import extract_think_tag


def test_extract_think_tag_basic():
    """Test basic functionality of extract_think_tag."""
    # Test basic think tag extraction
    text = "<think>This is thinking content</think>This is the response"
    think_content, remaining_text = extract_think_tag(text)
    
    assert think_content == "This is thinking content"
    assert remaining_text == "This is the response"


def test_extract_think_tag_multiline():
    """Test extract_think_tag with multiline content."""
    text = "<think>Line 1\nLine 2\nLine 3</think>This is the response"
    think_content, remaining_text = extract_think_tag(text)
    
    assert think_content == "Line 1\nLine 2\nLine 3"
    assert remaining_text == "This is the response"


def test_extract_think_tag_no_tag():
    """Test extract_think_tag when no tag is present."""
    text = "This is just regular text with no think tag"
    think_content, remaining_text = extract_think_tag(text)
    
    assert think_content is None
    assert remaining_text == text


def test_expert_think_tag_handling():
    """Test the logic that would be used in the expert tool for think tag handling."""
    # Mimic the implementation from expert.py
    def process_expert_response(text, supports_think_tag=False):
        """Simulate the expert tool's think tag handling."""
        if supports_think_tag:
            think_content, remaining_text = extract_think_tag(text)
            if think_content:
                # In the real implementation, this would display the thoughts
                thoughts_displayed = True
                return thoughts_displayed, think_content, remaining_text
        
        # No think content extracted
        return False, None, text
    
    # Test with think tag and support enabled
    thoughts_displayed, think_content, response = process_expert_response(
        "<think>Here's my reasoning</think>Final answer", 
        supports_think_tag=True
    )
    assert thoughts_displayed
    assert think_content == "Here's my reasoning"
    assert response == "Final answer"
    
    # Test with think tag but support disabled
    thoughts_displayed, think_content, response = process_expert_response(
        "<think>Here's my reasoning</think>Final answer", 
        supports_think_tag=False
    )
    assert not thoughts_displayed
    assert think_content is None
    assert response == "<think>Here's my reasoning</think>Final answer"
    
    # Test with no think tag
    thoughts_displayed, think_content, response = process_expert_response(
        "Just a regular response",
        supports_think_tag=True
    )
    assert not thoughts_displayed
    assert think_content is None
    assert response == "Just a regular response"


def test_expert_think_tag_with_supports_thinking():
    """Test handling of the supports_thinking parameter."""
    # Mimic the implementation from expert.py
    def process_expert_response(text, supports_think_tag=False, supports_thinking=False):
        """Simulate the expert tool's think tag handling with both parameters."""
        if supports_think_tag or supports_thinking:
            think_content, remaining_text = extract_think_tag(text)
            if think_content:
                # In the real implementation, this would display the thoughts
                thoughts_displayed = True
                return thoughts_displayed, think_content, remaining_text
        
        # No think content extracted
        return False, None, text
    
    # Test with supports_thinking=True
    thoughts_displayed, think_content, response = process_expert_response(
        "<think>Thinking with alternate parameter</think>Final answer", 
        supports_think_tag=False,
        supports_thinking=True
    )
    assert thoughts_displayed
    assert think_content == "Thinking with alternate parameter"
    assert response == "Final answer"


def test_expert_think_tag_combined_flags():
    """Test that either flag (supports_think_tag or supports_thinking) enables extraction."""
    # Mimic the implementation from expert.py
    def process_expert_response(text, supports_think_tag=False, supports_thinking=False):
        """Simulate the expert tool's think tag handling with both parameters."""
        if supports_think_tag or supports_thinking:
            think_content, remaining_text = extract_think_tag(text)
            if think_content:
                return think_content, remaining_text
        return None, text
    
    test_input = "<think>Some thoughts</think>Response text"
    
    # Test with both flags False
    think_content, response = process_expert_response(
        test_input,
        supports_think_tag=False,
        supports_thinking=False
    )
    assert think_content is None
    assert response == test_input
    
    # Test with supports_think_tag=True
    think_content, response = process_expert_response(
        test_input,
        supports_think_tag=True,
        supports_thinking=False
    )
    assert think_content == "Some thoughts"
    assert response == "Response text"
    
    # Test with supports_thinking=True
    think_content, response = process_expert_response(
        test_input,
        supports_think_tag=False,
        supports_thinking=True
    )
    assert think_content == "Some thoughts"
    assert response == "Response text"
    
    # Test with both flags True
    think_content, response = process_expert_response(
        test_input,
        supports_think_tag=True,
        supports_thinking=True
    )
    assert think_content == "Some thoughts"
    assert response == "Response text"


