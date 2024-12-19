import pytest
from ra_aid.tool_configs import (
    get_read_only_tools,
    get_research_tools,
    get_planning_tools,
    get_implementation_tools
)

def test_get_read_only_tools():
    # Test without human interaction
    tools = get_read_only_tools(human_interaction=False)
    assert len(tools) > 0
    assert all(callable(tool) for tool in tools)
    
    # Test with human interaction
    tools_with_human = get_read_only_tools(human_interaction=True)
    assert len(tools_with_human) == len(tools) + 1

def test_get_research_tools():
    # Test basic research tools
    tools = get_research_tools()
    assert len(tools) > 0
    assert all(callable(tool) for tool in tools)
    
    # Test without expert
    tools_no_expert = get_research_tools(expert_enabled=False)
    assert len(tools_no_expert) < len(tools)
    
    # Test research-only mode
    tools_research_only = get_research_tools(research_only=True)
    assert len(tools_research_only) < len(tools)

def test_get_planning_tools():
    # Test with expert enabled
    tools = get_planning_tools(expert_enabled=True)
    assert len(tools) > 0
    assert all(callable(tool) for tool in tools)
    
    # Test without expert
    tools_no_expert = get_planning_tools(expert_enabled=False)
    assert len(tools_no_expert) < len(tools)

def test_get_implementation_tools():
    # Test with expert enabled
    tools = get_implementation_tools(expert_enabled=True)
    assert len(tools) > 0
    assert all(callable(tool) for tool in tools)
    
    # Test without expert
    tools_no_expert = get_implementation_tools(expert_enabled=False)
    assert len(tools_no_expert) < len(tools)
