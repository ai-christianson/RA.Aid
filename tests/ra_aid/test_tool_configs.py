from ra_aid.tool_configs import (
    MODIFICATION_TOOLS,
    get_implementation_tools,
    get_planning_tools,
    get_read_only_tools,
    get_research_tools,
    get_web_research_tools,
    set_modification_tools,
)


def test_get_read_only_tools():
    # Test without human interaction, without aider
    tools = get_read_only_tools(human_interaction=False, use_aider=False)
    assert len(tools) > 0
    assert all(callable(tool) for tool in tools)

    # Check emit_related_files is not included when use_aider is False
    tool_names = [tool.name for tool in tools]
    assert "emit_related_files" not in tool_names

    # Test with use_aider=True
    tools_with_aider = get_read_only_tools(human_interaction=False, use_aider=True)
    tool_names_with_aider = [tool.name for tool in tools_with_aider]
    assert "emit_related_files" in tool_names_with_aider

    # Test with human interaction
    tools_with_human = get_read_only_tools(human_interaction=True, use_aider=False)
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


def test_get_web_research_tools():
    # Test with expert enabled
    tools = get_web_research_tools(expert_enabled=True)
    assert len(tools) == 5
    assert all(callable(tool) for tool in tools)

    # Get tool names and verify exact matches
    tool_names = [tool.name for tool in tools]
    expected_names = [
        "emit_expert_context",
        "ask_expert",
        "web_search_tavily",
        "emit_research_notes",
        "task_completed",
    ]
    assert sorted(tool_names) == sorted(expected_names)

    # Test without expert enabled
    tools_no_expert = get_web_research_tools(expert_enabled=False)
    assert len(tools_no_expert) == 3
    assert all(callable(tool) for tool in tools_no_expert)

    # Verify exact tool names when expert is disabled
    tool_names_no_expert = [tool.name for tool in tools_no_expert]
    assert sorted(tool_names_no_expert) == sorted(
        ["web_search_tavily", "emit_research_notes", "task_completed"]
    )


def test_set_modification_tools():
    # Test with use_aider=False (default setting)
    set_modification_tools(use_aider=False)
    tool_names = [tool.name for tool in MODIFICATION_TOOLS]
    assert "file_str_replace" in tool_names
    assert "put_complete_file_contents" in tool_names
    assert "run_programming_task" not in tool_names

    # Test with use_aider=True
    set_modification_tools(use_aider=True)
    tool_names = [tool.name for tool in MODIFICATION_TOOLS]
    assert "file_str_replace" not in tool_names
    assert "put_complete_file_contents" not in tool_names
    assert "run_programming_task" in tool_names

    # Reset to default for other tests
    set_modification_tools(use_aider=False)
