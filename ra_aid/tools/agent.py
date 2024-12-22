"""Tools for spawning and managing sub-agents."""

from langchain_core.tools import tool
from typing import Dict, Any, Union, List
from typing_extensions import TypeAlias

ResearchResult = Dict[str, Union[str, bool, Dict[int, Any], List[Any], None]]
from rich.console import Console
from ra_aid.tools.memory import _global_memory
from ra_aid.console.formatting import print_error, print_interrupt
from .memory import get_memory_value, get_related_files
from ..llm import initialize_llm
from ..console import print_task_header

CANCELLED_BY_USER_REASON = "The operation was explicitly cancelled by the user. This typically is an indication that the action requested was not aligned with the user request."

RESEARCH_AGENT_RECURSION_LIMIT = 2

console = Console()

@tool("request_research")
def request_research(query: str) -> ResearchResult:
    """Spawn a research-only agent to investigate the given query.

    This function creates a new research agent to investigate the given query. It includes 
    recursion depth limiting to prevent infinite recursive research calls.

    Args:
        query: The research question or project description
    """
    # Initialize model from config
    config = _global_memory.get('config', {})
    model = initialize_llm(config.get('provider', 'anthropic'), config.get('model', 'claude-3-5-sonnet-20241022'))
    
    # Check recursion depth
    current_depth = _global_memory.get('research_depth', 0)
    if current_depth >= RESEARCH_AGENT_RECURSION_LIMIT:
        print_error("Maximum research recursion depth reached")
        return {
            "completion_message": "Research stopped - maximum recursion depth reached",
            "key_facts": get_memory_value("key_facts"),
            "related_files": list(get_related_files()),
            "research_notes": get_memory_value("research_notes"),
            "key_snippets": get_memory_value("key_snippets"),
            "success": False,
            "reason": "max_depth_exceeded"
        }

    success = True
    reason = None
    
    try:
        # Increment depth counter
        _global_memory['research_depth'] = current_depth + 1
        
        # Run research agent
        from ..agent_utils import run_research_agent
        result = run_research_agent(
            query,
            model,
            expert_enabled=True,
            research_only=True,
            hil=config.get('hil', False),
            console_message=query
        )
    except KeyboardInterrupt:
        print_interrupt("Research interrupted by user")
        success = False
        reason = CANCELLED_BY_USER_REASON
    except Exception as e:
        print_error(f"Error during research: {str(e)}")
        success = False
        reason = f"error: {str(e)}"
    finally:
        # Always decrement depth counter
        _global_memory['research_depth'] = current_depth
        
    # Get completion message if available
    completion_message = _global_memory.get('completion_message', 'Task was completed successfully.' if success else None)
    
    # Clear completion state from global memory
    _global_memory['completion_message'] = ''
    _global_memory['task_completed'] = False
        
    return {
        "completion_message": completion_message,
        "key_facts": get_memory_value("key_facts"),
        "related_files": list(get_related_files()),
        "research_notes": get_memory_value("research_notes"),
        "key_snippets": get_memory_value("key_snippets"),
        "success": success,
        "reason": reason
    }

@tool("request_research_and_implementation")
def request_research_and_implementation(query: str) -> Dict[str, Any]:
    """Spawn a research agent to investigate and implement the given query.
    
    Args:
        query: The research question or project description
    """
    # Initialize model from config
    config = _global_memory.get('config', {})
    model = initialize_llm(config.get('provider', 'anthropic'), config.get('model', 'claude-3-5-sonnet-20241022'))
    
    try:
        # Run research agent
        from ..agent_utils import run_research_agent
        result = run_research_agent(
            query,
            model,
            expert_enabled=True,
            research_only=False,
            hil=config.get('hil', False),
            console_message=query
        )
        
        success = True
        reason = None
    except KeyboardInterrupt:
        print_interrupt("Task interrupted by user")
        success = False
        reason = CANCELLED_BY_USER_REASON
    except Exception as e:
        console.print(f"\n[red]Error during research: {str(e)}[/red]")
        success = False
        reason = f"error: {str(e)}"
        
    # Get completion message if available
    completion_message = _global_memory.get('completion_message', 'Task was completed successfully.' if success else None)
    
    # Clear completion state from global memory
    _global_memory['completion_message'] = ''
    _global_memory['task_completed'] = False
    _global_memory['plan_completed'] = False
        
    return {
        "completion_message": completion_message,
        "key_facts": get_memory_value("key_facts"),
        "related_files": list(get_related_files()),
        "research_notes": get_memory_value("research_notes"),
        "key_snippets": get_memory_value("key_snippets"),
        "success": success,
        "reason": reason
    }

@tool("request_task_implementation")
def request_task_implementation(task_spec: str) -> Dict[str, Any]:
    """Spawn an implementation agent to execute the given task.
    
    Args:
        task_spec: The full task specification
    """
    # Initialize model from config
    config = _global_memory.get('config', {})
    model = initialize_llm(config.get('provider', 'anthropic'), config.get('model', 'claude-3-5-sonnet-20241022'))
    
    # Get required parameters
    tasks = [_global_memory['tasks'][task_id] for task_id in sorted(_global_memory['tasks'])]
    plan = _global_memory.get('plan', '')
    related_files = list(get_related_files())
    
    try:
        print_task_header(task_spec)
        # Run implementation agent
        from ..agent_utils import run_task_implementation_agent
        result = run_task_implementation_agent(
            base_task=_global_memory.get('base_task', ''),
            tasks=tasks,
            task=task_spec,
            plan=plan, 
            related_files=related_files,
            model=model,
            expert_enabled=True
        )
        
        success = True
        reason = None
    except KeyboardInterrupt:
        print_interrupt("Task implementation interrupted by user")
        success = False
        reason = CANCELLED_BY_USER_REASON
    except Exception as e:
        print_error(f"Error during task implementation: {str(e)}")
        success = False
        reason = f"error: {str(e)}"
        
    # Get completion message if available
    completion_message = _global_memory.get('completion_message', 'Task was completed successfully.' if success else None)
    
    # Clear completion state from global memory
    _global_memory['completion_message'] = ''
    _global_memory['task_completed'] = False
        
    return {
        "key_facts": get_memory_value("key_facts"),
        "related_files": list(get_related_files()),
        "key_snippets": get_memory_value("key_snippets"),
        "completion_message": completion_message,
        "success": success,
        "reason": reason
    }

@tool("request_implementation")
def request_implementation(task_spec: str) -> Dict[str, Any]:
    """Spawn a planning agent to create an implementation plan for the given task.
    
    Args:
        task_spec: The task specification to plan implementation for
    """
    # Initialize model from config
    config = _global_memory.get('config', {})
    model = initialize_llm(config.get('provider', 'anthropic'), config.get('model', 'claude-3-5-sonnet-20241022'))
    
    try:
        # Run planning agent
        from ..agent_utils import run_planning_agent
        result = run_planning_agent(
            task_spec,
            model,
            config=config,
            expert_enabled=True,
            hil=config.get('hil', False)
        )
        
        success = True
        reason = None
    except KeyboardInterrupt:
        print_interrupt("Planning interrupted by user")
        success = False
        reason = CANCELLED_BY_USER_REASON
    except Exception as e:
        print_error(f"Error during planning: {str(e)}")
        success = False
        reason = f"error: {str(e)}"
        
    # Get completion message if available
    completion_message = _global_memory.get('completion_message', 'Task was completed successfully.' if success else None)
    
    # Clear completion state from global memory
    _global_memory['completion_message'] = ''
    _global_memory['task_completed'] = False
    _global_memory['plan_completed'] = False
        
    return {
        "completion_message": completion_message,
        "key_facts": get_memory_value("key_facts"),
        "related_files": list(get_related_files()),
        "key_snippets": get_memory_value("key_snippets"),
        "success": success,
        "reason": reason
    }
