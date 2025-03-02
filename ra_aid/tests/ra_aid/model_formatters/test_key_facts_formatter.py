"""Unit tests for the key_facts_formatter module."""

from ra_aid.model_formatters import format_key_fact, format_key_facts_dict


class TestKeyFactsFormatter:
    """Test cases for key facts formatting functions."""

    def test_format_key_fact(self):
        """Test formatting a single key fact."""
        # Test with valid input
        formatted = format_key_fact(1, "This is an important fact")
        assert formatted == "## 🔑 Key Fact #1\n\nThis is an important fact"

        # Test with large ID number
        formatted = format_key_fact(999, "Fact with large ID")
        assert formatted == "## 🔑 Key Fact #999\n\nFact with large ID"

        # Test with empty content
        formatted = format_key_fact(5, "")
        assert formatted == ""

        # Test with multi-line content
        multi_line = "Line 1\nLine 2\nLine 3"
        formatted = format_key_fact(3, multi_line)
        assert formatted == f"## 🔑 Key Fact #3\n\n{multi_line}"

    def test_format_key_facts_dict(self):
        """Test formatting a dictionary of key facts."""
        # Test with multiple facts
        facts_dict = {
            1: "First fact",
            2: "Second fact",
            5: "Fifth fact"
        }
        formatted = format_key_facts_dict(facts_dict)
        expected = (
            "## 🔑 Key Fact #1\n\nFirst fact\n\n"
            "## 🔑 Key Fact #2\n\nSecond fact\n\n"
            "## 🔑 Key Fact #5\n\nFifth fact"
        )
        assert formatted == expected

        # Test with empty dictionary
        formatted = format_key_facts_dict({})
        assert formatted == ""

        # Test with None value
        formatted = format_key_facts_dict(None)
        assert formatted == ""

        # Test with single fact
        formatted = format_key_facts_dict({3: "Only fact"})
        assert formatted == "## 🔑 Key Fact #3\n\nOnly fact"

        # Test ordering - should be ordered by key
        unordered_dict = {
            5: "Fifth",
            1: "First",
            3: "Third"
        }
        formatted = format_key_facts_dict(unordered_dict)
        expected = (
            "## 🔑 Key Fact #1\n\nFirst\n\n"
            "## 🔑 Key Fact #3\n\nThird\n\n"
            "## 🔑 Key Fact #5\n\nFifth"
        )
        assert formatted == expected