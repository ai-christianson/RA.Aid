from ra_aid.tools import (
    ask_expert, ask_human, run_shell_command, run_programming_task,
    emit_research_notes, emit_plan, emit_related_files, emit_task,
    emit_expert_context, emit_key_facts, delete_key_facts,
    emit_key_snippets, delete_key_snippets, deregister_related_files, delete_tasks, read_file_tool,
    fuzzy_find_project_files, ripgrep_search, list_directory_tree,
    swap_task_order, monorepo_detected, existing_project_detected, ui_detected,
    task_completed, plan_implementation_completed, web_search_tavily
)
from ra_aid.tools.memory import one_shot_completed
from ra_aid.tools.agent import request_research, request_implementation, request_research_and_implementation, request_task_implementation, request_web_research

# Read-only tools that don't modify system state
def get_read_only_tools(human_interaction: bool = False, web_research_enabled: bool = False) -> list:
    """Get the list of read-only tools, optionally including human interaction tools.
    
    Args:
        human_interaction: Whether to include human interaction tools
        web_research_enabled: Whether to include web research tools
    
    Returns:
        List of tool functions
    """
    tools = [
        emit_related_files,
        emit_key_facts,
        delete_key_facts,
        emit_key_snippets,
        delete_key_snippets,
        deregister_related_files,
        list_directory_tree,
        read_file_tool,
        fuzzy_find_project_files,
        ripgrep_search,
        run_shell_command # can modify files, but we still need it for read-only tasks.
    ]

    if web_research_enabled:
        tools.append(request_web_research)
    
    if human_interaction:
        tools.append(ask_human)
    
    return tools

# Define constant tool groups
READ_ONLY_TOOLS = get_read_only_tools()
MODIFICATION_TOOLS = [run_programming_task]
COMMON_TOOLS = get_read_only_tools()
EXPERT_TOOLS = [emit_expert_context, ask_expert]
RESEARCH_TOOLS = [
    emit_research_notes,
    one_shot_completed,
    monorepo_detected,
    existing_project_detected,
    ui_detected
]

def get_research_tools(research_only: bool = False, expert_enabled: bool = True, human_interaction: bool = False, web_research_enabled: bool = False) -> list:
    """Get the list of research tools based on mode and whether expert is enabled.
    
    Args:
        research_only: Whether to exclude modification tools
        expert_enabled: Whether to include expert tools
        human_interaction: Whether to include human interaction tools
        web_research_enabled: Whether to include web research tools
    """
    # Start with read-only tools
    tools = get_read_only_tools(human_interaction, web_research_enabled).copy()
    
    tools.extend(RESEARCH_TOOLS)
    
    # Add modification tools if not research_only
    if not research_only:
        tools.extend(MODIFICATION_TOOLS)
        tools.append(request_implementation)
    
    # Add expert tools if enabled
    if expert_enabled:
        tools.extend(EXPERT_TOOLS)
    
    # Add chat-specific tools
    tools.append(request_research)
    
    return tools

def get_planning_tools(expert_enabled: bool = True, web_research_enabled: bool = False) -> list:
    """Get the list of planning tools based on whether expert is enabled.
    
    Args:
        expert_enabled: Whether to include expert tools
        web_research_enabled: Whether to include web research tools
    """
    # Start with read-only tools
    tools = get_read_only_tools(web_research_enabled=web_research_enabled).copy()
    
    # Add planning-specific tools
    planning_tools = [
        emit_plan,
        request_task_implementation,
        plan_implementation_completed
    ]
    tools.extend(planning_tools)
    
    # Add expert tools if enabled
    if expert_enabled:
        tools.extend(EXPERT_TOOLS)
    
    return tools

def get_implementation_tools(expert_enabled: bool = True, web_research_enabled: bool = False) -> list:
    """Get the list of implementation tools based on whether expert is enabled.
    
    Args:
        expert_enabled: Whether to include expert tools
        web_research_enabled: Whether to include web research tools
    """
    # Start with read-only tools
    tools = get_read_only_tools(web_research_enabled=web_research_enabled).copy()
    
    # Add modification tools since it's not research-only
    tools.extend(MODIFICATION_TOOLS)
    tools.extend([
        task_completed
    ])
    
    # Add expert tools if enabled
    if expert_enabled:
        tools.extend(EXPERT_TOOLS)
    
    return tools

def get_web_research_tools(expert_enabled: bool = True) -> list:
    """Get the list of tools available for web research.
    
    Args:
        expert_enabled: Whether expert tools should be included
        
    Returns:
        list: List of tools configured for web research
    """
    tools = [
        web_search_tavily,
        emit_research_notes,
        task_completed
    ]

    if expert_enabled:
        tools.append(emit_expert_context)
        tools.append(ask_expert)

    return tools

def get_chat_tools(expert_enabled: bool = True, web_research_enabled: bool = False) -> list:
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
        delete_key_facts,
        delete_key_snippets,
        deregister_related_files
    ]

    if web_research_enabled:
        tools.append(request_web_research)

    return tools
