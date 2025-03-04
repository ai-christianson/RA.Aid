import unittest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage

from ra_aid.agent_backends.ciayn_agent import CiaynAgent


class TestCiaynStreamBundling(unittest.TestCase):
    """Test that the _execute_tool method correctly formats bundled tool call results."""

    def setUp(self):
        """Set up the test case with mocked model and tools."""
        # Create mock model
        self.model = MagicMock()
        
        # Set up the tools
        self.tools = [
            MagicMock(func=lambda content: f"Result of tool1: {content}"),
            MagicMock(func=lambda content: f"Result of tool2: {content}"),
        ]
        
        # Set up function names
        self.tools[0].func.__name__ = "emit_expert_context"
        self.tools[1].func.__name__ = "ask_expert"
        
        # Create agent
        self.agent = CiaynAgent(model=self.model, tools=self.tools)
        
        # Mock validate_function_call_pattern to always return False (valid)
        self.validate_patcher = patch('ra_aid.agent_backends.ciayn_agent.validate_function_call_pattern', return_value=False)
        self.mock_validate = self.validate_patcher.start()

    def tearDown(self):
        """Clean up after tests."""
        self.validate_patcher.stop()

    @patch('random.choice', side_effect=lambda chars: chars[0])  # Make random IDs predictable for testing
    def test_execute_tool_with_bundled_calls(self, mock_random):
        """Test that _execute_tool correctly formats bundled tool call results."""
        # Create a message with bundled tool calls
        bundled_calls = """emit_expert_context("Test content 1")
ask_expert("Test question 1")"""
        msg = AIMessage(content=bundled_calls)
        
        # Call _execute_tool directly
        result = self.agent._execute_tool(msg)
        
        # Verify the format of the result
        self.assertIn("<result-", result)
        self.assertIn("</result-", result)
        self.assertIn("Result of tool1: Test content 1", result)
        self.assertIn("Result of tool2: Test question 1", result)
        
        # Verify we have the expected number of result sections
        self.assertEqual(2, result.count("<result-"))
        self.assertEqual(2, result.count("</result-"))
        
        # Verify the specific format with predictable IDs (using our mock for random.choice)
        expected_format = "<result-aaaaaa>\nResult of tool1: Test content 1\n</result-aaaaaa>\n\n<result-aaaaaa>\nResult of tool2: Test question 1\n</result-aaaaaa>"
        self.assertEqual(expected_format, result)


if __name__ == "__main__":
    unittest.main()
