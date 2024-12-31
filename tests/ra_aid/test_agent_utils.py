"""Tests for agent utilities."""

import pytest
from unittest.mock import patch, Mock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from ra_aid.exceptions import AgentInterrupt, ToolExecutionError
from ra_aid.agents.ciayn_agent import CiaynAgent
from ra_aid.output import print_error
from ra_aid.agent_utils import (
    create_agent,
    run_research_agent,
    run_web_research_agent,
    run_planning_agent,
    run_task_implementation_agent,
    run_conversation_agent,
    _global_memory,
)

# Initialize global memory for tests
_global_memory = {}

@pytest.fixture(autouse=True)
def setup_global_memory():
    """Initialize global memory before each test."""
    _global_memory.clear()
    _global_memory.update({
        'config': {'provider': 'openai', 'model': 'gpt-4'},
        'research_notes': [],
        'related_files': [],
        'key_facts': [],
        'key_snippets': []
    })
    yield
    _global_memory.clear()

@pytest.fixture
def mock_model():
    """Create a mock LLM model."""
    model = Mock()
    model.invoke.return_value = AIMessage(content="output_markdown_message(\"Test message\")")
    return model

@pytest.fixture
def mock_memory():
    """Create a mock memory instance."""
    return Mock()

@pytest.fixture
def mock_config():
    """Create a mock WebUIConfig."""
    class MockConfig:
        def __init__(self):
            self._config = {
                "provider": "openai",
                "model": "gpt-4",
                "research_only": False,
                "web_research_enabled": True,
                "hil": True
            }
        
        def copy(self):
            return self._config.copy()
        
        def get(self, key, default=None):
            return self._config.get(key, default)
    
    return MockConfig()

def test_create_agent_handles_model_name_cleanup(mock_model):
    """Test that create_agent properly cleans up model names."""
    with patch('ra_aid.agent_utils._global_memory', {
        'config': {'provider': 'openai', 'model': 'openai/gpt-4-turbo'}
    }):
        agent = create_agent(mock_model, [])
        assert agent is not None
        # Check that model name was cleaned up in global memory
        assert _global_memory['config']['model'] == 'gpt-4-turbo'

def test_research_agent_tool_configuration(mock_model, mock_memory, mock_config):
    """Test that research agent is configured with correct tools."""
    with patch('ra_aid.agent_utils.get_research_tools') as mock_get_tools:
        mock_get_tools.return_value = []
        
        run_research_agent(
            "Test query",
            mock_model,
            expert_enabled=True,
            research_only=True,
            hil=True,
            web_research_enabled=True,
            memory=mock_memory,
            config=mock_config
        )
        
        # Verify tools were configured correctly
        mock_get_tools.assert_called_once_with(
            research_only=True,
            expert_enabled=True,
            human_interaction=True,
            web_research_enabled=True
        )

def test_web_research_agent_tool_configuration(mock_model, mock_memory, mock_config):
    """Test that web research agent is configured with correct tools."""
    with patch('ra_aid.agent_utils.get_web_research_tools') as mock_get_tools:
        mock_get_tools.return_value = []
        
        run_web_research_agent(
            "Test query",
            mock_model,
            expert_enabled=True,
            hil=True,
            web_research_enabled=True,
            memory=mock_memory,
            config=mock_config
        )
        
        # Verify tools were configured correctly
        mock_get_tools.assert_called_once_with(expert_enabled=True)

def test_planning_agent_tool_configuration(mock_model, mock_memory, mock_config):
    """Test that planning agent is configured with correct tools."""
    with patch('ra_aid.agent_utils.get_planning_tools') as mock_get_tools:
        mock_get_tools.return_value = []
        
        run_planning_agent(
            "Test task",
            mock_model,
            expert_enabled=True,
            hil=True,
            web_research_enabled=True,
            memory=mock_memory,
            config=mock_config
        )
        
        # Verify tools were configured correctly
        mock_get_tools.assert_called_once_with(
            expert_enabled=True,
            web_research_enabled=True
        )

def test_implementation_agent_tool_configuration(mock_model, mock_memory, mock_config):
    """Test that implementation agent is configured with correct tools."""
    with patch('ra_aid.agent_utils.get_implementation_tools') as mock_get_tools:
        mock_get_tools.return_value = []
        
        run_task_implementation_agent(
            "Test task",
            ["task1", "task2"],
            "task1",
            "Test plan",
            ["file1.py", "file2.py"],
            mock_model,
            expert_enabled=True,
            web_research_enabled=True,
            memory=mock_memory,
            config=mock_config
        )
        
        # Verify tools were configured correctly
        mock_get_tools.assert_called_once_with(
            expert_enabled=True,
            web_research_enabled=True
        )

def test_agent_handles_webui_config(mock_model, mock_memory, mock_config):
    """Test that agents properly handle WebUIConfig objects."""
    with patch('ra_aid.agent_utils.run_agent_with_retry') as mock_run:
        run_research_agent(
            "Test query",
            mock_model,
            config=mock_config
        )
        
        # Verify config was properly converted to dict
        call_args = mock_run.call_args
        assert isinstance(call_args[0][2], dict)  # config argument
        assert 'provider' in call_args[0][2]
        assert call_args[0][2]['model'] == 'gpt-4'

def test_agent_error_handling(mock_model, mock_memory):
    """Test that agents properly handle and propagate errors."""
    mock_model.invoke.side_effect = Exception("Test error")
    
    with pytest.raises(Exception) as exc_info:
        run_research_agent(
            "Test query",
            mock_model
        )
    
    assert "Research agent failed" in str(exc_info.value)

def test_agent_tool_execution(mock_model, mock_memory):
    """Test that agents can execute tools properly."""
    # Mock a successful tool execution
    mock_model.invoke.return_value = AIMessage(content='emit_research_notes("Success")')
    
    result = run_research_agent(
        "Test query",
        mock_model,
        memory=mock_memory
    )
    
    assert result == "Agent run completed successfully"

def test_agent_tool_execution_error(mock_model, mock_memory):
    """Test that agents handle tool execution errors properly."""
    # Mock a failed tool execution
    mock_model.invoke.return_value = AIMessage(content='invalid_tool("test")')
    
    with pytest.raises(RuntimeError) as exc_info:
        run_research_agent(
            "Test query",
            mock_model,
            memory=mock_memory
        )
    
    assert "Agent failed to execute tools successfully" in str(exc_info.value)

def test_ciayn_agent_integration(mock_model, mock_memory, mock_config):
    """Test integration of CiaynAgent with WebUI execution flow."""
    with patch('ra_aid.agent_utils.CiaynAgent') as mock_ciayn:
        # Configure mock CiaynAgent
        mock_instance = Mock()
        mock_instance.stream.return_value = [{"tools": {"messages": [{"status": "success", "content": "Test result"}]}}]
        mock_ciayn.return_value = mock_instance
        
        # Test with research agent
        result = run_research_agent(
            "Test query",
            mock_model,
            expert_enabled=True,
            research_only=True,
            hil=True,
            web_research_enabled=True,
            memory=mock_memory,
            config=mock_config
        )
        
        assert result == "Agent run completed successfully"
        assert mock_ciayn.called
        
        # Verify agent configuration
        agent_args = mock_ciayn.call_args[0]
        assert agent_args[0] == mock_model  # model
        assert len(agent_args[1]) > 0  # tools

def test_ciayn_agent_streaming(mock_model, mock_memory, mock_config):
    """Test streaming interface with CiaynAgent."""
    with patch('ra_aid.agent_utils.CiaynAgent') as mock_ciayn:
        # Configure mock CiaynAgent with multiple chunks
        mock_instance = Mock()
        mock_instance.stream.return_value = [
            {"content": "Starting..."},
            {"content": "Step 1"},
            {"content": "Step 2"},
            {"content": "Complete"}
        ]
        mock_ciayn.return_value = mock_instance

        result = run_research_agent(
            "Test query",
            mock_model,
            config=mock_config
        )

        assert result == "Agent run completed successfully"
        # Verify streaming was used
        stream_args = mock_instance.stream.call_args[0][0]
        assert "messages" in stream_args
        assert isinstance(stream_args["messages"][0], HumanMessage)

def test_ciayn_agent_error_handling(mock_model, mock_memory, mock_config):
    """Test error handling with CiaynAgent."""
    with patch('ra_aid.agent_utils.CiaynAgent') as mock_ciayn:
        # Configure mock CiaynAgent to raise error
        mock_instance = Mock()
        mock_instance.stream.side_effect = RuntimeError("Test error")
        mock_ciayn.return_value = mock_instance

        with pytest.raises(RuntimeError) as exc_info:
            run_research_agent(
                "Test query",
                mock_model,
                config=mock_config
            )

        assert "Test error" in str(exc_info.value)

def test_ciayn_agent_concurrent_execution(mock_model, mock_memory, mock_config):
    """Test concurrent execution of CiaynAgent instances."""
    with patch('ra_aid.agent_utils.CiaynAgent') as mock_ciayn:
        # Configure mock CiaynAgent
        mock_instance = Mock()
        mock_instance.stream.return_value = [{"tools": {"messages": [{"status": "success", "content": "Test result"}]}}]
        mock_ciayn.return_value = mock_instance
        
        import threading
        
        def run_agent():
            return run_research_agent(
                "Test query",
                mock_model,
                config=mock_config
            )
        
        # Run multiple agents concurrently
        threads = []
        results = []
        
        for _ in range(3):
            thread = threading.Thread(target=lambda: results.append(run_agent()))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(results) == 3
        assert all(result == "Agent run completed successfully" for result in results)

def test_ciayn_agent_tool_configuration(mock_model, mock_memory, mock_config):
    """Test tool configuration with CiaynAgent."""
    with patch('ra_aid.agent_utils.CiaynAgent') as mock_ciayn:
        mock_instance = Mock()
        mock_instance.stream.return_value = [{"tools": {"messages": [{"status": "success", "content": "Test result"}]}}]
        mock_ciayn.return_value = mock_instance
        
        # Test with different tool configurations
        configs = [
            {"expert_enabled": True, "web_research_enabled": True},
            {"expert_enabled": False, "web_research_enabled": True},
            {"expert_enabled": True, "web_research_enabled": False},
        ]
        
        for config in configs:
            run_research_agent(
                "Test query",
                mock_model,
                memory=mock_memory,
                config=mock_config,
                **config
            )
            
            # Verify correct tools were passed
            agent_args = mock_ciayn.call_args[0]
            tools = agent_args[1]
            
            if config["expert_enabled"]:
                assert any(hasattr(t.func, '__name__') and 'expert' in t.func.__name__ for t in tools)
            if config["web_research_enabled"]:
                assert any(hasattr(t.func, '__name__') and 'web' in t.func.__name__ for t in tools) 

def test_model_name_handling(mock_model, mock_memory):
    """Test different model name formats are handled correctly."""
    test_cases = [
        # (provider, model_name, expected_clean_name)
        ('openai', 'gpt-4', 'gpt-4'),
        ('openai', 'openai/gpt-4', 'gpt-4'),
        ('anthropic', 'claude-2', 'claude-2'),
        ('anthropic', 'anthropic/claude-2', 'claude-2'),
        ('openai', 'gpt-4-turbo', 'gpt-4-turbo'),
    ]

    for provider, model_name, expected in test_cases:
        _global_memory['config'] = {'provider': provider, 'model': model_name}
        agent = create_agent(mock_model, [])
        assert agent is not None
        assert isinstance(agent, CiaynAgent)
        # Verify model name was cleaned up in global memory
        assert _global_memory['config']['model'] == expected

def test_provider_configuration(mock_model, mock_memory):
    """Test agent creation with different providers."""
    test_cases = [
        {'provider': 'openai', 'model': 'gpt-4'},
        {'provider': 'anthropic', 'model': 'claude-2'},
        {'provider': 'unknown', 'model': 'test-model'},
    ]

    for config in test_cases:
        _global_memory['config'] = config.copy()
        agent = create_agent(mock_model, [])
        assert agent is not None
        assert isinstance(agent, CiaynAgent)
        # Verify config was preserved
        assert _global_memory['config']['provider'] == config['provider']
        assert _global_memory['config']['model'] == config['model']

def test_default_configuration(mock_model, mock_memory):
    """Test agent creation with missing or empty configuration."""
    test_cases = [
        {},  # Empty config
        {'provider': '', 'model': ''},  # Empty strings
        {'provider': None, 'model': None},  # None values
    ]

    for config in test_cases:
        _global_memory['config'] = config.copy()
        agent = create_agent(mock_model, [])
        assert agent is not None
        assert isinstance(agent, CiaynAgent)
        # Verify defaults were applied
        assert _global_memory['config'].get('model') == 'gpt-4'

def test_agent_creation_error_recovery(mock_model, mock_memory):
    """Test agent creation recovers gracefully from errors."""
    _global_memory['config'] = {'provider': 'invalid', 'model': 'invalid/model-name'}
    
    # Should still create agent with cleaned up name
    agent = create_agent(mock_model, [])
    assert agent is not None
    assert isinstance(agent, CiaynAgent)
    # Verify model name was cleaned
    assert _global_memory['config']['model'] == 'model-name'

@pytest.mark.asyncio
async def test_run_conversation_agent(mock_model, mock_memory, mock_config):
    """Test running a conversation agent."""
    with patch('ra_aid.agent_utils.create_agent') as mock_create_agent, \
         patch('ra_aid.agent_utils.run_agent_with_retry') as mock_run_agent:
        # Configure mocks
        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent
        mock_run_agent.return_value = "Test response"
        
        # Run conversation agent
        result = await run_conversation_agent(
            "Test task",
            mock_model,
            mock_config
        )
        
        # Verify results
        assert result == "Test response"
        mock_create_agent.assert_called_once_with(mock_model, [], checkpointer=None)
        mock_run_agent.assert_called_once()
        
        # Verify config handling
        run_config = mock_run_agent.call_args[0][2]
        assert "configurable" in run_config
        assert "thread_id" in run_config["configurable"]
        assert "recursion_limit" in run_config

@pytest.mark.asyncio
async def test_run_conversation_agent_error(mock_model, mock_memory, mock_config):
    """Test error handling in conversation agent."""
    with patch('ra_aid.agent_utils.create_agent') as mock_create_agent, \
         patch('ra_aid.agent_utils.run_agent_with_retry') as mock_run_agent:
        # Configure mock to raise error
        mock_run_agent.side_effect = Exception("Test error")
        
        # Run conversation agent and expect error
        with pytest.raises(Exception) as exc_info:
            await run_conversation_agent(
                "Test task",
                mock_model,
                mock_config
            )
        
        assert "Test error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_run_conversation_agent_interrupt(mock_model, mock_memory, mock_config):
    """Test handling interrupts in conversation agent."""
    with patch('ra_aid.agent_utils.create_agent') as mock_create_agent, \
         patch('ra_aid.agent_utils.run_agent_with_retry') as mock_run_agent:
        # Configure mock to raise interrupt
        mock_run_agent.side_effect = AgentInterrupt("Test interrupt")
        
        # Run conversation agent and expect interrupt
        with pytest.raises(AgentInterrupt):
            await run_conversation_agent(
                "Test task",
                mock_model,
                mock_config
            ) 