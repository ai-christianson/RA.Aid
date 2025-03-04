import unittest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage

from ra_aid.agent_backends.ciayn_agent import CiaynAgent


class TestCiaynSingleTool(unittest.TestCase):
    """Test that single tool calls still work correctly with the bundling implementation."""

    def setUp(self):
        """Set up the test case with mocked model and tools."""
        self.model = MagicMock()
        self.tools = [
            MagicMock(func=lambda content: f"Result: {content}"),
        ]
        
        # Set up function name
        self.tools[0].func.__name__ = "non_bundleable_tool"
        
        # Create agent
        self.agent = CiaynAgent(model=self.model, tools=self.tools)
        
        # Mock validate_function_call_pattern to always return False (valid)
        self.validate_patcher = patch('ra_aid.agent_backends.ciayn_agent.validate_function_call_pattern', return_value=False)
        self.mock_validate = self.validate_patcher.start()

    def tearDown(self):
        """Clean up after tests."""
        self.validate_patcher.stop()

    def test_single_tool_call(self):
        """Test that single tool calls return the result directly without result tags."""
        # Create a message with a single tool call
        msg = AIMessage(content="non_bundleable_tool(\"Test content\")")
        
        # Execute the tool call
        result = self.agent._execute_tool(msg)
        
        # Verify the result is just the plain text result without result tags
        self.assertEqual("Result: Test content", result)
        self.assertNotIn("<result-", result)
        self.assertNotIn("</result-", result)


if __name__ == "__main__":
    unittest.main()
