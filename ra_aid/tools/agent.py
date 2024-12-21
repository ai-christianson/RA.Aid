"""Tools for spawning and managing sub-agents."""

from langchain_core.tools import tool
from typing import Dict, Any
from rich.console import Console
from ra_aid.tools.memory import _global_memory
from .memory import get_memory_value, get_related_files
from ..llm import initialize_llm

console = Console()

@tool("request_research")
def request_research(query: str) -> Dict[str, Any]:
    """Spawn a research-only agent to investigate the given query.
    
    Args:
        query: The research question or project description
        
    Returns:
        Dict containing:
        - notes: Research notes from the agent 
        - facts: Current key facts
        - files: Related files
        - success: Whether completed or interrupted
        - reason: Reason for failure, if any
    """
    # Initialize model
    model = initialize_llm("anthropic", "claude-3-sonnet-20240229")
    
    try:
        # Run research agent
        from ..agent_utils import run_research_agent
        result = run_research_agent(
            query,
            model,
            expert_enabled=True,
            research_only=True,
            hil=_global_memory.get('config', {}).get('hil', False),
            console_message=query
        )
        
        success = True
        reason = None
    except KeyboardInterrupt:
        console.print("\n[yellow]Research interrupted by user[/yellow]")
        success = False
        reason = "cancelled_by_user"
    except Exception as e:
        console.print(f"\n[red]Error during research: {str(e)}[/red]")
        success = False
        reason = f"error: {str(e)}"
        
    # Gather results
    return {
        "facts": get_memory_value("key_facts"),
        "files": list(get_related_files()),
        "notes": get_memory_value("research_notes"),
        "success": success,
        "reason": reason
    }
