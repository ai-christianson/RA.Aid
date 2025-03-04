import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage

from ra_aid.agent_backends.ciayn_agent import CiaynAgent

class TestNoRepeatTools:
    @pytest.fixture
    def mock_model(self):
        mock = MagicMock()
        return mock
    
    @pytest.fixture
    def mock_tool(self):
        mock = MagicMock()
        mock.func.__name__ = "test_tool"
        mock.func.return_value = "Tool execution result"
        return mock
    
    @pytest.fixture
    def agent(self, mock_model, mock_tool):
        # Create the agent with our mock model and tool
        agent = CiaynAgent(mock_model, [mock_tool])
        # Add the test tool to the NO_REPEAT_TOOLS list
        agent.NO_REPEAT_TOOLS.append("test_tool")
        return agent
    
    def test_repeat_tool_call_rejection(self, agent, mock_tool):
        """Test that repeat tool calls with the same parameters are rejected."""
        # First call should succeed
        first_message = AIMessage(content="test_tool(param1='value1', param2='value2')")
        result1 = agent._execute_tool(first_message)
        assert result1 == "Tool execution result"
        
        # Second identical call should be rejected
        second_message = AIMessage(content="test_tool(param1='value1', param2='value2')")
        result2 = agent._execute_tool(second_message)
        assert "Repeat calls of test_tool with the same parameters are not allowed" in result2
        
        # Call with different parameters should succeed
        third_message = AIMessage(content="test_tool(param1='value1', param2='different')")
        result3 = agent._execute_tool(third_message)
        assert result3 == "Tool execution result"
    
    def test_bundled_calls_repeat_rejection(self, agent, mock_tool):
        """Test that repeat tool calls in bundled calls are rejected."""
        # Mock the _detect_multiple_tool_calls method to simulate bundled calls
        with patch.object(agent, '_detect_multiple_tool_calls') as mock_detect:
            # Set up two bundled calls where the second one is a repeat
            mock_detect.return_value = [
                "test_tool(param1='value1', param2='value2')",
                "test_tool(param1='value1', param2='value2')"
            ]
            
            # Execute the bundled calls
            message = AIMessage(content="test_tool(...)\ntest_tool(...)")
            result = agent._execute_tool(message)
            
            # First call should succeed, second should be rejected
            assert "Tool execution result" in result
            assert "Repeat calls of test_tool with the same parameters are not allowed" in result
    
    def test_different_tool_not_affected(self, mock_model):
        """Test that tools not in NO_REPEAT_TOOLS list can be called repeatedly."""
        # Create different mock tools for this test
        mock_tool1 = MagicMock()
        mock_tool1.func.__name__ = "non_repeat_tool"
        mock_tool1.func.return_value = "Tool execution result"
        
        # Create a fresh agent with our mock model and tool
        agent = CiaynAgent(mock_model, [mock_tool1])
        
        # First call
        first_message = AIMessage(content="non_repeat_tool(param1='value1', param2='value2')")
        result1 = agent._execute_tool(first_message)
        assert result1 == "Tool execution result"
        
        # Second identical call should also succeed because this tool is not in NO_REPEAT_TOOLS
        second_message = AIMessage(content="non_repeat_tool(param1='value1', param2='value2')")
        result2 = agent._execute_tool(second_message)
        assert result2 == "Tool execution result"

    def test_run_shell_command_detection(self, mock_model):
        """Test the shell command detection logic to ensure it's not creating false positives."""
        # Create mock tools for this test 
        mock_tool1 = MagicMock()
        mock_tool1.func.__name__ = "run_shell_command"
        mock_tool1.func.return_value = "Shell command result"
        
        # Create a fresh agent
        agent = CiaynAgent(mock_model, [mock_tool1])
        
        # First call to run_shell_command
        first_message = AIMessage(content="run_shell_command(CommandLine='g++ main.cpp -o spinning_cube -lGL -lGLU -lglut', Cwd='/home/user', Blocking=True)")
        result1 = agent._execute_tool(first_message)
        assert result1 == "Shell command result"
        
        # Second call with same command should be detected as duplicate 
        second_message = AIMessage(content="run_shell_command(CommandLine='g++ main.cpp -o spinning_cube -lGL -lGLU -lglut', Cwd='/home/user', Blocking=True)")
        result2 = agent._execute_tool(second_message)
        assert "Repeat calls of run_shell_command with the same parameters are not allowed" in result2
        
        # Different command should work
        third_message = AIMessage(content="run_shell_command(CommandLine='./spinning_cube', Cwd='/home/user', Blocking=True)")
        result3 = agent._execute_tool(third_message)
        assert result3 == "Shell command result"
        
        # Test with same command but different parameter order (should still be detected as duplicate)
        fourth_message = AIMessage(content="run_shell_command(Blocking=True, Cwd='/home/user', CommandLine='./spinning_cube')")
        result4 = agent._execute_tool(fourth_message)
        assert "Repeat calls of run_shell_command with the same parameters are not allowed" in result4
        
        # Test with different boolean value (True vs true) - these should be treated as different
        fifth_message = AIMessage(content="run_shell_command(CommandLine='ls -la', Cwd='/home/user', Blocking=True)")  # different command
        result5 = agent._execute_tool(fifth_message)
        assert result5 == "Shell command result"
        
        # Test with SafeToAutoRun parameter - should be treated as different
        sixth_message = AIMessage(content="run_shell_command(CommandLine='ls -la', Cwd='/home/user', Blocking=True, SafeToAutoRun=True)")
        result6 = agent._execute_tool(sixth_message)
        assert result6 == "Shell command result"

    def test_positional_args_detection(self, mock_model):
        """Test that positional arguments are properly included in fingerprinting."""
        # Create mock tools for this test
        mock_tool1 = MagicMock()
        mock_tool1.func.__name__ = "test_tool"
        mock_tool1.func.return_value = "Tool execution result"
        
        # Create a fresh agent
        agent = CiaynAgent(mock_model, [mock_tool1])
        # Add the test tool to the NO_REPEAT_TOOLS list
        agent.NO_REPEAT_TOOLS.append("test_tool")
        
        # First call with positional args
        first_message = AIMessage(content="test_tool('value1', 'value2')")
        result1 = agent._execute_tool(first_message)
        assert result1 == "Tool execution result"
        
        # Second identical call should be rejected
        second_message = AIMessage(content="test_tool('value1', 'value2')")
        result2 = agent._execute_tool(second_message)
        assert "Repeat calls of test_tool with the same parameters are not allowed" in result2
        
        # Call with different positional args should succeed
        third_message = AIMessage(content="test_tool('value1', 'different')")
        result3 = agent._execute_tool(third_message)
        assert result3 == "Tool execution result"
        
        # Call with same values but as keyword args should be considered different
        fourth_message = AIMessage(content="test_tool(param1='value1', param2='different')")
        result4 = agent._execute_tool(fourth_message)
        assert result4 == "Tool execution result"
        
        # Mixed positional and keyword args
        fifth_message = AIMessage(content="test_tool('value1', param2='different')")
        result5 = agent._execute_tool(fifth_message)
        assert result5 == "Tool execution result"
        
        # Repeat of mixed call should be rejected
        sixth_message = AIMessage(content="test_tool('value1', param2='different')")
        result6 = agent._execute_tool(sixth_message)
        assert "Repeat calls of test_tool with the same parameters are not allowed" in result6
