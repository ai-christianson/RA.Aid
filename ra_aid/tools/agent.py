"""Tools for spawning and managing sub-agents."""

from langchain_core.tools import tool
from typing import Dict, Any, List, Optional
import uuid
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from ra_aid.tools.memory import _global_memory
from ra_aid import run_agent_with_retry
from ..prompts import RESEARCH_PROMPT
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
    """
    # Initialize model and memory
    model = initialize_llm("anthropic", "claude-3-sonnet-20240229")
    memory = MemorySaver()
    memory.memory = _global_memory
    
    # Configure research tools
    from ..tool_configs import get_research_tools
    tools = get_research_tools(research_only=True, expert_enabled=True)
    
    # Basic config matching main process
    config = {
        "thread_id": str(uuid.uuid4()),
        "memory": memory,
        "model": model
    }
    
    from ra_aid.prompts import (
        RESEARCH_PROMPT, 
        EXPERT_PROMPT_SECTION_RESEARCH,
        HUMAN_PROMPT_SECTION_RESEARCH
    )
    
    # Create research agent
    config = _global_memory.get('config', {})
    expert_enabled = config.get('expert_enabled', False)
    hil = config.get('hil', False)
    
    expert_section = EXPERT_PROMPT_SECTION_RESEARCH if expert_enabled else ""
    human_section = HUMAN_PROMPT_SECTION_RESEARCH if hil else ""
    
    agent = create_react_agent(model, tools)
    
    prompt = RESEARCH_PROMPT.format(
        base_task=query,
        research_only_note='',
        expert_section=expert_section,
        human_section=human_section
    )
    
    try:
        console.print(Panel(Markdown(query), title="🔬 Research Task"))
        # Run agent with retry logic
        result = run_agent_with_retry(
            agent,
            prompt,
            {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 100}
        )
        
        success = True
    except KeyboardInterrupt:
        console.print("\n[yellow]Research interrupted by user[/yellow]")
        success = False
    except Exception as e:
        console.print(f"\n[red]Error during research: {str(e)}[/red]")
        success = False
        
    # Gather results
    return {
        "facts": get_memory_value("key_facts"),
        "files": list(get_related_files()),
        "notes": get_memory_value("research_notes"),
        "success": success
    }
