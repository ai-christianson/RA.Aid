import os
import pytest
from ra_aid.text.processing import extract_think_tag

def test_basic_extraction():
    """Test basic extraction of think tag content."""
    content = "<think>This is a test</think>Remaining content"
    expected_extracted = "This is a test"
    expected_remaining = "Remaining content"
    
    extracted, remaining = extract_think_tag(content)
    
    assert extracted == expected_extracted
    assert remaining == expected_remaining

def test_multiline_extraction():
    """Test extraction of multiline think tag content."""
    content = "<think>Line 1\nLine 2\nLine 3</think>Remaining content"
    expected_extracted = "Line 1\nLine 2\nLine 3"
    expected_remaining = "Remaining content"
    
    extracted, remaining = extract_think_tag(content)
    
    assert extracted == expected_extracted
    assert remaining == expected_remaining

def test_multiple_think_tags():
    """Test that only the first think tag is extracted."""
    content = "<think>First tag</think>Middle<think>Second tag</think>End"
    expected_extracted = "First tag"
    expected_remaining = "Middle<think>Second tag</think>End"
    
    extracted, remaining = extract_think_tag(content)
    
    assert extracted == expected_extracted
    assert remaining == expected_remaining

def test_no_think_tag():
    """Test behavior when no think tag is present."""
    content = "This is a string without a think tag"
    
    extracted, remaining = extract_think_tag(content)
    
    assert extracted is None
    assert remaining == content

def test_empty_think_tag():
    """Test extraction of an empty think tag."""
    content = "<think></think>Remaining content"
    expected_extracted = ""
    expected_remaining = "Remaining content"
    
    extracted, remaining = extract_think_tag(content)
    
    assert extracted == expected_extracted
    assert remaining == expected_remaining

def test_whitespace_handling():
    """Test whitespace handling in think tag extraction."""
    content = "<think>  \n  Content with whitespace  \n  </think>Remaining content"
    expected_extracted = "  \n  Content with whitespace  \n  "
    expected_remaining = "Remaining content"
    
    extracted, remaining = extract_think_tag(content)
    
    assert extracted == expected_extracted
    assert remaining == expected_remaining

def test_tag_not_at_start():
    """Test behavior when think tag is not at the start of the string."""
    content = "Some content before <think>Think content</think>Remaining content"
    
    extracted, remaining = extract_think_tag(content)
    
    assert extracted is None
    assert remaining == content

def test_sample_data():
    """Test extraction using sample data from tests/data/think-tag/sample_1.txt."""
    # Get the absolute path to the sample file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sample_file_path = os.path.join(current_dir, '..', '..', 'data', 'think-tag', 'sample_1.txt')
    
    # Read the sample data
    with open(sample_file_path, 'r', encoding='utf-8') as f:
        sample_data = f.read()
    
    # Extract the think tag
    extracted, remaining = extract_think_tag(sample_data)
    
    # Check that extraction worked
    assert extracted is not None
    assert "Okay, the user wants me to write a" in extracted
    assert "return 0;" in extracted
    
    # Check that we got the think tag content without the tags
    assert not extracted.startswith("<think>")
    assert not extracted.endswith("</think>")
    
    # Check that the remaining content doesn't contain the think tag
    assert "<think>" not in remaining
    assert "</think>" not in remaining