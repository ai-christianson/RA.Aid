"""Utility functions for working with agents."""

import time
import uuid
from typing import Optional, Any, List

from langgraph.prebuilt import create_react_agent
from ra_aid.tool_configs import get_implementation_tools, get_research_tools
from ra_aid.prompts import (
    IMPLEMENTATION_PROMPT,
    EXPERT_PROMPT_SECTION_IMPLEMENTATION,
    HUMAN_PROMPT_SECTION_IMPLEMENTATION,
    EXPERT_PROMPT_SECTION_RESEARCH,
    RESEARCH_PROMPT,
    HUMAN_PROMPT_SECTION_RESEARCH
)
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import HumanMessage
from langchain_core.messages import BaseMessage
from anthropic import APIError, APITimeoutError, RateLimitError, InternalServerError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ra_aid.tools.memory import (
    _global_memory,
    get_memory_value,
)
from ra_aid.globals import RESEARCH_AGENT_RECURSION_LIMIT
from ra_aid.tool_configs import get_research_tools
from ra_aid.prompts import (
    RESEARCH_PROMPT,
    EXPERT_PROMPT_SECTION_RESEARCH,
    HUMAN_PROMPT_SECTION_RESEARCH
)

console = Console()

def run_research_agent(
    base_task_or_query: str,
    model,
    *,
    expert_enabled: bool = False,
    research_only: bool = False,
    hil: bool = False,
    memory: Optional[Any] = None,
    config: Optional[dict] = None,
    thread_id: Optional[str] = None,
    console_message: Optional[str] = None
) -> Optional[str]:
    """Run a research agent with the given configuration.
    
    Args:
        base_task_or_query: The main task or query for research
        model: The LLM model to use
        expert_enabled: Whether expert mode is enabled
        research_only: Whether this is a research-only task
        hil: Whether human-in-the-loop mode is enabled
        memory: Optional memory instance to use
        config: Optional configuration dictionary
        thread_id: Optional thread ID (defaults to new UUID)
        console_message: Optional message to display before running
        
    Returns:
        Optional[str]: The completion message if task completed successfully
        
    Example:
        result = run_research_agent(
            "Research Python async patterns",
            model,
            expert_enabled=True,
            research_only=True
        )
    """
    # Initialize memory if not provided
    if memory is None:
        memory = MemorySaver()

    # Set up thread ID
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    # Configure tools
    tools = get_research_tools(
        research_only=research_only,
        expert_enabled=expert_enabled,
        human_interaction=hil
    )

    # Create agent
    agent = create_react_agent(model, tools, checkpointer=memory)

    # Format prompt sections
    expert_section = EXPERT_PROMPT_SECTION_RESEARCH if expert_enabled else ""
    human_section = HUMAN_PROMPT_SECTION_RESEARCH if hil else ""
    
    # Build prompt
    prompt = RESEARCH_PROMPT.format(
        base_task=base_task_or_query,
        research_only_note='' if research_only else ' Only request implementation if the user explicitly asked for changes to be made.',
        expert_section=expert_section,
        human_section=human_section
    )

    # Set up configuration
    run_config = {
         "configurable": {"thread_id": thread_id},
        "recursion_limit": 100
    }
    if config:
        run_config.update(config)

    # Display console message if provided
    if console_message:
        console.print(Panel(Markdown(console_message), title="🔬 Research Task"))

    # Run agent with retry logic
    return run_agent_with_retry(agent, prompt, run_config)

def print_agent_output(chunk: dict[str, BaseMessage]) -> None:
    """Print agent output chunks."""
    if chunk.get("delta") and chunk["delta"].content:
        console.print(chunk["delta"].content, end="", style="blue")

def print_error(msg: str) -> None:
    """Print error messages."""
    console.print(f"\n{msg}", style="red")

def run_task_implementation_agent(
    base_task: str,
    tasks: list,
    task: str,
    plan: str,
    related_files: list,
    model,
    *,
    expert_enabled: bool = False,
    memory: Optional[Any] = None,
    config: Optional[dict] = None,
    thread_id: Optional[str] = None
) -> Optional[str]:
    """Run an implementation agent for a specific task.
    
    Args:
        base_task: The main task being implemented
        tasks: List of tasks to implement
        plan: The implementation plan
        related_files: List of related files
        model: The LLM model to use
        expert_enabled: Whether expert mode is enabled
        memory: Optional memory instance to use
        config: Optional configuration dictionary
        thread_id: Optional thread ID (defaults to new UUID)
        
    Returns:
        Optional[str]: The completion message if task completed successfully
    """
    # Initialize memory if not provided
    if memory is None:
        memory = MemorySaver()

    # Set up thread ID
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    # Configure tools
    tools = get_implementation_tools(expert_enabled=expert_enabled)

    # Create agent
    agent = create_react_agent(model, tools, checkpointer=memory)

    # Build prompt
    prompt = IMPLEMENTATION_PROMPT.format(
        base_task=base_task,
        task=task,
        tasks=tasks,
        plan=plan,
        related_files=related_files,
        key_facts=get_memory_value('key_facts'),
        key_snippets=get_memory_value('key_snippets'),
        expert_section=EXPERT_PROMPT_SECTION_IMPLEMENTATION if expert_enabled else "",
        human_section=HUMAN_PROMPT_SECTION_IMPLEMENTATION if _global_memory.get('config', {}).get('hil', False) else ""
    )

    # Set up configuration
    run_config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 100
    }
    if config:
        run_config.update(config)

    # Run agent with retry logic
    return run_agent_with_retry(agent, prompt, run_config)

def run_agent_with_retry(agent, prompt: str, config: dict) -> Optional[str]:
    """Run an agent with retry logic for internal server errors and task completion handling.
    
    Args:
        agent: The agent to run
        prompt: The prompt to send to the agent
        config: Configuration dictionary for the agent
        
    Returns:
        Optional[str]: The completion message if task was completed, None otherwise
        
    Handles API errors with exponential backoff retry logic and checks for task
    completion after each chunk of output.
    """
    max_retries = 20
    base_delay = 1  # Initial delay in seconds
    
    for attempt in range(max_retries):
        try:
            for chunk in agent.stream(
                {"messages": [HumanMessage(content=prompt)]},
                config
            ):
                print_agent_output(chunk)
            break
        except (InternalServerError, APITimeoutError, RateLimitError, APIError) as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Max retries ({max_retries}) exceeded. Last error: {str(e)}")
            
            delay = base_delay * (2 ** attempt)  # Exponential backoff
            error_type = e.__class__.__name__
            print_error(f"Encountered {error_type}: {str(e)}. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
            continue
