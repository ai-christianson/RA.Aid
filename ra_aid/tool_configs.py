from langchain_core.tools import BaseTool

from ra_aid.tools import (
    ask_expert,
    ask_human,
    emit_expert_context,
    emit_key_facts,
    emit_key_snippet,
    emit_related_files,
    emit_research_notes,
    file_str_replace,
    fuzzy_find_project_files,
    list_directory_tree,
    put_complete_file_contents,
    read_file_tool,
    ripgrep_search,
    run_programming_task,
    run_shell_command,
    task_completed,
    web_search_tavily,
)
from ra_aid.tools.agent import (
    request_implementation,
    request_research,
    request_research_and_implementation,
    request_task_implementation,
    request_web_research,
)
from ra_aid.tools.memory import plan_implementation_completed


def set_modification_tools(use_aider=False):
    """Set the MODIFICATION_TOOLS list based on configuration.

    Args:
        use_aider: Whether to use run_programming_task (True) or file modification tools (False)
    """
    global MODIFICATION_TOOLS
    if use_aider:
        MODIFICATION_TOOLS.clear()
        MODIFICATION_TOOLS.append(run_programming_task)
    else:
        MODIFICATION_TOOLS.clear()
        MODIFICATION_TOOLS.extend([file_str_replace, put_complete_file_contents])


# Read-only tools that don't modify system state
def get_read_only_tools(
    human_interaction: bool = False,
    web_research_enabled: bool = False,
    use_aider: bool = False,
):
    """Get the list of read-only tools, optionally including human interaction tools.

    Args:
        human_interaction: Whether to include human interaction tools
        web_research_enabled: Whether to include web research tools
        use_aider: Whether aider is being used for code modifications

    Returns:
        List of tool functions
    """
    tools = [
        emit_key_snippet,
        # Only include emit_related_files if use_aider is True
        *([emit_related_files] if use_aider else []),
        emit_key_facts,
        # *TEMPORARILY* disabled to improve tool calling perf.
        # delete_key_facts,
        # delete_key_snippets,
        # deregister_related_files,
        list_directory_tree,
        read_file_tool,
        fuzzy_find_project_files,
        ripgrep_search,
        run_shell_command,  # can modify files, but we still need it for read-only tasks.
    ]

    if web_research_enabled:
        tools.append(request_web_research)

    if human_interaction:
        tools.append(ask_human)

    return tools


def get_all_tools() -> list[BaseTool]:
    """Return a list containing all available tools from different groups."""
    all_tools = []
    all_tools.extend(get_read_only_tools())
    all_tools.extend(MODIFICATION_TOOLS)
    all_tools.extend(EXPERT_TOOLS)
    all_tools.extend(RESEARCH_TOOLS)
    all_tools.extend(get_web_research_tools())
    all_tools.extend(get_chat_tools())
    return all_tools


# Define constant tool groups
# Get config from global memory for use_aider value
_config = {}
try:
    from ra_aid.tools.memory import _global_memory

    _config = _global_memory.get("config", {})
except ImportError:
    pass

READ_ONLY_TOOLS = get_read_only_tools(use_aider=_config.get("use_aider", False))

# MODIFICATION_TOOLS will be set dynamically based on config, default defined here
MODIFICATION_TOOLS = [file_str_replace, put_complete_file_contents]
COMMON_TOOLS = get_read_only_tools(use_aider=_config.get("use_aider", False))
EXPERT_TOOLS = [emit_expert_context, ask_expert]
RESEARCH_TOOLS = [
    emit_research_notes,
    # *TEMPORARILY* disabled to improve tool calling perf.
    # one_shot_completed,
    # monorepo_detected,
    # ui_detected,
]


def get_research_tools(
    research_only: bool = False,
    expert_enabled: bool = True,
    human_interaction: bool = False,
    web_research_enabled: bool = False,
):
    """Get the list of research tools based on mode and whether expert is enabled.

    Args:
        research_only: Whether to exclude modification tools
        expert_enabled: Whether to include expert tools
        human_interaction: Whether to include human interaction tools
        web_research_enabled: Whether to include web research tools
    """
    # Get config for use_aider value
    use_aider = False
    try:
        from ra_aid.tools.memory import _global_memory

        use_aider = _global_memory.get("config", {}).get("use_aider", False)
    except ImportError:
        pass

    # Start with read-only tools
    tools = get_read_only_tools(
        human_interaction, web_research_enabled, use_aider=use_aider
    ).copy()

    tools.extend(RESEARCH_TOOLS)

    # Add modification tools if not research_only
    if not research_only:
        # For now, we ONLY do modifications after planning.
        # tools.extend(MODIFICATION_TOOLS)
        tools.append(request_implementation)

    # Add expert tools if enabled
    if expert_enabled:
        tools.extend(EXPERT_TOOLS)

    # Add chat-specific tools
    tools.append(request_research)

    return tools


def get_planning_tools(
    expert_enabled: bool = True, web_research_enabled: bool = False
) -> list:
    """Get the list of planning tools based on whether expert is enabled.

    Args:
        expert_enabled: Whether to include expert tools
        web_research_enabled: Whether to include web research tools
    """
    # Get config for use_aider value
    use_aider = False
    try:
        from ra_aid.tools.memory import _global_memory

        use_aider = _global_memory.get("config", {}).get("use_aider", False)
    except ImportError:
        pass

    # Start with read-only tools
    tools = get_read_only_tools(
        web_research_enabled=web_research_enabled, use_aider=use_aider
    ).copy()

    # Add planning-specific tools
    planning_tools = [
        request_task_implementation,
        plan_implementation_completed,
        # *TEMPORARILY* disabled to improve tool calling perf.
        # emit_plan,
    ]
    tools.extend(planning_tools)

    # Add expert tools if enabled
    if expert_enabled:
        tools.extend(EXPERT_TOOLS)

    return tools


def get_implementation_tools(
    expert_enabled: bool = True, web_research_enabled: bool = False
) -> list:
    """Get the list of implementation tools based on whether expert is enabled.

    Args:
        expert_enabled: Whether to include expert tools
        web_research_enabled: Whether to include web research tools
    """
    # Get config for use_aider value
    use_aider = False
    try:
        from ra_aid.tools.memory import _global_memory

        use_aider = _global_memory.get("config", {}).get("use_aider", False)
    except ImportError:
        pass

    # Start with read-only tools
    tools = get_read_only_tools(
        web_research_enabled=web_research_enabled, use_aider=use_aider
    ).copy()

    # Add modification tools since it's not research-only
    tools.extend(MODIFICATION_TOOLS)
    tools.extend([task_completed])

    # Add expert tools if enabled
    if expert_enabled:
        tools.extend(EXPERT_TOOLS)

    return tools


def get_web_research_tools(expert_enabled: bool = True):
    """Get the list of tools available for web research.

    Args:
        expert_enabled: Whether expert tools should be included
        human_interaction: Whether to include human interaction tools
        web_research_enabled: Whether to include web research tools

    Returns:
        list: List of tools configured for web research
    """
    tools = [web_search_tavily, emit_research_notes, task_completed]

    if expert_enabled:
        tools.append(emit_expert_context)
        tools.append(ask_expert)

    return tools


def get_chat_tools(expert_enabled: bool = True, web_research_enabled: bool = False):
    """Get the list of tools available in chat mode.

    Chat mode includes research and implementation capabilities but excludes
    complex planning tools. Human interaction is always enabled.

    Args:
        expert_enabled: Whether to include expert tools
        web_research_enabled: Whether to include web research tools
    """
    tools = [
        ask_human,
        request_research,
        request_research_and_implementation,
        emit_key_facts,
        # *TEMPORARILY* disabled to improve tool calling perf.
        # delete_key_facts,
        # delete_key_snippets,
        # deregister_related_files,
    ]

    if web_research_enabled:
        tools.append(request_web_research)

    return tools
