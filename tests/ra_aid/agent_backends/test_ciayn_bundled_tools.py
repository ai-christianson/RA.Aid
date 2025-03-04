import unittest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage

from ra_aid.agent_backends.ciayn_agent import CiaynAgent

class TestCiaynBundledTools(unittest.TestCase):
    """Test the bundled tool call functionality in the CiaynAgent."""

    def setUp(self):
        """Set up the test case with mocked model and tools."""
        self.model = MagicMock()
        self.tools = [
            MagicMock(func=lambda content: f"Result of tool1: {content}"),
            MagicMock(func=lambda content: f"Result of tool2: {content}"),
        ]
        
        # Set up function names
        self.tools[0].func.__name__ = "emit_expert_context"
        self.tools[1].func.__name__ = "ask_expert"
        
        # Create agent
        self.agent = CiaynAgent(model=self.model, tools=self.tools)
        
        # Mock the validation to always return False (valid)
        self.validate_patcher = patch('ra_aid.agent_backends.ciayn_agent.validate_function_call_pattern', return_value=False)
        self.mock_validate = self.validate_patcher.start()
        
        # Mock _extract_tool_call in case it's needed
        self.extract_patcher = patch.object(self.agent, '_extract_tool_call', return_value="mocked_tool_call")
        self.mock_extract = self.extract_patcher.start()

    def tearDown(self):
        """Clean up patches after the test."""
        self.validate_patcher.stop()
        self.extract_patcher.stop()

    @patch('random.choice', side_effect=lambda chars: chars[0])  # Make random IDs predictable for testing
    def test_bundled_tool_calls(self, mock_random):
        """Test that bundled tool calls return results properly formatted with result tags."""
        # Create a message with multiple bundled tool calls
        msg = AIMessage(content="""emit_expert_context("Expert context 1")
ask_expert("Expert question 1")""")
        
        # Execute the tool calls
        result = self.agent._execute_tool(msg)
        
        # Verify the result has both results with proper format
        self.assertIn("<result-", result)
        self.assertIn("</result-", result)
        self.assertIn("Result of tool1: Expert context 1", result)
        self.assertIn("Result of tool2: Expert question 1", result)
        
        # Verify we have two result sections
        self.assertEqual(2, result.count("<result-"))
        self.assertEqual(2, result.count("</result-"))

    @patch('random.choice', side_effect=lambda chars: chars[0])  # Make random IDs predictable for testing
    def test_single_tool_call(self, mock_random):
        """Test that single tool calls still return just the result without tags."""
        # Create a message with a single tool call
        msg = AIMessage(content="emit_expert_context(\"Expert context 1\")")
        
        # Execute the tool call
        result = self.agent._execute_tool(msg)
        
        # Verify the result is just the plain text result without result tags
        self.assertEqual("Result of tool1: Expert context 1", result)
        self.assertNotIn("<result-", result)
        self.assertNotIn("</result-", result)

    def test_random_id_generation(self):
        """Test that the _generate_random_id method creates IDs of the correct length."""
        # Generate IDs of different lengths
        id1 = self.agent._generate_random_id(length=6)
        id2 = self.agent._generate_random_id(length=10)
        
        # Verify the IDs are of the correct length
        self.assertEqual(6, len(id1))
        self.assertEqual(10, len(id2))
        
        # Verify the IDs are different
        self.assertNotEqual(id1, id2[:6])


if __name__ == "__main__":
    unittest.main()
