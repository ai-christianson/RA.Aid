"""Utility functions for working with agents."""

import sys
import time
import uuid
from typing import Optional, Any, List

import signal
import threading
import time
from typing import Optional

class AgentInterrupt(Exception):
    """Exception raised when an agent's execution is interrupted.
    
    This exception is used for internal agent interruption handling,
    separate from KeyboardInterrupt which is reserved for top-level handling.
    """
    pass

from langgraph.prebuilt import create_react_agent
from ra_aid.console.formatting import print_stage_header, print_error
from ra_aid.console.output import print_agent_output
from ra_aid.tool_configs import (
    get_implementation_tools,
    get_research_tools,
    get_planning_tools
)
from ra_aid.prompts import (
    IMPLEMENTATION_PROMPT,
    EXPERT_PROMPT_SECTION_IMPLEMENTATION,
    HUMAN_PROMPT_SECTION_IMPLEMENTATION,
    EXPERT_PROMPT_SECTION_RESEARCH,
    WEB_RESEARCH_PROMPT_SECTION_RESEARCH,
    WEB_RESEARCH_PROMPT_SECTION_CHAT,
    WEB_RESEARCH_PROMPT_SECTION_PLANNING,
    RESEARCH_PROMPT,
    RESEARCH_ONLY_PROMPT,
    HUMAN_PROMPT_SECTION_RESEARCH,
    PLANNING_PROMPT,
    EXPERT_PROMPT_SECTION_PLANNING,
    WEB_RESEARCH_PROMPT_SECTION_PLANNING,
    HUMAN_PROMPT_SECTION_PLANNING,
    WEB_RESEARCH_PROMPT,
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
    get_related_files,
)
from ra_aid.tool_configs import get_research_tools
from ra_aid.prompts import (
    RESEARCH_PROMPT,
    RESEARCH_ONLY_PROMPT,
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
    web_research_section = WEB_RESEARCH_PROMPT_SECTION_RESEARCH if config.get('web_research') else ""
    
    # Get research context from memory
    key_facts = _global_memory.get("key_facts", "")
    code_snippets = _global_memory.get("code_snippets", "")
    related_files = _global_memory.get("related_files", "")
    
    # Build prompt
    prompt = (RESEARCH_ONLY_PROMPT if research_only else RESEARCH_PROMPT).format(
        base_task=base_task_or_query,
        research_only_note='' if research_only else ' Only request implementation if the user explicitly asked for changes to be made.',
        expert_section=expert_section,
        human_section=human_section,
        web_research_section=web_research_section,
        key_facts=key_facts,
        code_snippets=code_snippets,
        related_files=related_files
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
        console.print(Panel(Markdown(console_message), title="🔬 Looking into it..."))

    # Run agent with retry logic
    return run_agent_with_retry(agent, prompt, run_config)

def run_web_research_agent(
    base_task_or_query: str,
    model,
    *,
    expert_enabled: bool = False,
    hil: bool = False,
    memory: Optional[Any] = None,
    config: Optional[dict] = None,
    thread_id: Optional[str] = None,
    console_message: Optional[str] = None
) -> Optional[str]:
    """Run a web research agent with the given configuration.
    
    Args:
        base_task_or_query: The main task or query for web research
        model: The LLM model to use
        expert_enabled: Whether expert mode is enabled
        hil: Whether human-in-the-loop mode is enabled
        memory: Optional memory instance to use
        config: Optional configuration dictionary
        thread_id: Optional thread ID (defaults to new UUID)
        console_message: Optional message to display before running
        
    Returns:
        Optional[str]: The completion message if task completed successfully
        
    Example:
        result = run_web_research_agent(
            "Research latest Python async patterns",
            model,
            expert_enabled=True
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
        expert_enabled=expert_enabled,
        human_interaction=hil
    )

    # Create agent
    agent = create_react_agent(model, tools, checkpointer=memory)

    # Format prompt sections
    expert_section = EXPERT_PROMPT_SECTION_RESEARCH if expert_enabled else ""
    human_section = HUMAN_PROMPT_SECTION_RESEARCH if hil else ""
    
    # Get research context from memory
    key_facts = _global_memory.get("key_facts", "")
    code_snippets = _global_memory.get("code_snippets", "")
    related_files = _global_memory.get("related_files", "")
    
    # Build prompt
    prompt = WEB_RESEARCH_PROMPT.format(
        base_task=base_task_or_query,
        expert_section=expert_section,
        human_section=human_section,
        key_facts=key_facts,
        code_snippets=code_snippets,
        related_files=related_files
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
        console.print(Panel(Markdown(console_message), title="🔍 Searching the Web..."))

    # Run agent with retry logic
    return run_agent_with_retry(agent, prompt, run_config)

def run_planning_agent(
    base_task: str,
    model,
    *,
    expert_enabled: bool = False,
    hil: bool = False,
    memory: Optional[Any] = None,
    config: Optional[dict] = None,
    thread_id: Optional[str] = None
) -> Optional[str]:
    """Run a planning agent to create implementation plans.
    
    Args:
        base_task: The main task to plan implementation for
        model: The LLM model to use
        expert_enabled: Whether expert mode is enabled
        hil: Whether human-in-the-loop mode is enabled
        memory: Optional memory instance to use
        config: Optional configuration dictionary
        thread_id: Optional thread ID (defaults to new UUID)
        
    Returns:
        Optional[str]: The completion message if planning completed successfully
    """
    # Initialize memory if not provided
    if memory is None:
        memory = MemorySaver()

    # Set up thread ID
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    # Configure tools
    tools = get_planning_tools(expert_enabled=expert_enabled)

    # Create agent
    agent = create_react_agent(model, tools, checkpointer=memory)

    # Format prompt sections
    expert_section = EXPERT_PROMPT_SECTION_PLANNING if expert_enabled else ""
    human_section = HUMAN_PROMPT_SECTION_PLANNING if hil else ""
    web_research_section = WEB_RESEARCH_PROMPT_SECTION_PLANNING if config.get('web_research') else ""
    
    # Build prompt
    planning_prompt = PLANNING_PROMPT.format(
        expert_section=expert_section,
        human_section=human_section,
        web_research_section=web_research_section,
        base_task=base_task,
        research_notes=get_memory_value('research_notes'),
        related_files="\n".join(get_related_files()),
        key_facts=get_memory_value('key_facts'),
        key_snippets=get_memory_value('key_snippets'),
        research_only_note='' if config.get('research_only') else ' Only request implementation if the user explicitly asked for changes to be made.'
    )

    # Set up configuration
    run_config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 100
    }
    if config:
        run_config.update(config)

    # Run agent with retry logic
    print_stage_header("Planning Stage")
    return run_agent_with_retry(agent, planning_prompt, run_config)

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
        work_log=get_memory_value('work_log'),
        expert_section=EXPERT_PROMPT_SECTION_IMPLEMENTATION if expert_enabled else "",
        human_section=HUMAN_PROMPT_SECTION_IMPLEMENTATION if _global_memory.get('config', {}).get('hil', False) else "",
        web_research_section=WEB_RESEARCH_PROMPT_SECTION_CHAT if config.get('web_research') else ""
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

_CONTEXT_STACK = []
_INTERRUPT_CONTEXT = None
_FEEDBACK_MODE = False

def _request_interrupt(signum, frame):
    global _INTERRUPT_CONTEXT
    if _CONTEXT_STACK:
        _INTERRUPT_CONTEXT = _CONTEXT_STACK[-1]
    
    if _FEEDBACK_MODE:
        print()
        print(" 👋 Bye!")
        print()
        sys.exit(0)

class InterruptibleSection:
    def __enter__(self):
        _CONTEXT_STACK.append(self)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        _CONTEXT_STACK.remove(self)

def check_interrupt():
    if _CONTEXT_STACK and _INTERRUPT_CONTEXT is _CONTEXT_STACK[-1]:
        raise AgentInterrupt("Interrupt requested")

def run_agent_with_retry(agent, prompt: str, config: dict) -> Optional[str]:
    original_handler = None
    if threading.current_thread() is threading.main_thread():
        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _request_interrupt)

    max_retries = 20
    base_delay = 1

    with InterruptibleSection():
        try:
            # Track agent execution depth
            current_depth = _global_memory.get('agent_depth', 0)
            _global_memory['agent_depth'] = current_depth + 1
            
            for attempt in range(max_retries):
                check_interrupt()
                try:
                    for chunk in agent.stream({"messages": [HumanMessage(content=prompt)]}, config):
                        check_interrupt()
                        print_agent_output(chunk)
                    return "Agent run completed successfully"
                except (KeyboardInterrupt, AgentInterrupt):
                    raise
                except (InternalServerError, APITimeoutError, RateLimitError, APIError) as e:
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"Max retries ({max_retries}) exceeded. Last error: {e}")
                    delay = base_delay * (2 ** attempt)
                    print_error(f"Encountered {e.__class__.__name__}: {e}. Retrying in {delay}s... (Attempt {attempt+1}/{max_retries})")
                    start = time.monotonic()
                    while time.monotonic() - start < delay:
                        check_interrupt()
                        time.sleep(0.1)
        finally:
            # Reset depth tracking
            _global_memory['agent_depth'] = _global_memory.get('agent_depth', 1) - 1
            
            if original_handler and threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGINT, original_handler)
