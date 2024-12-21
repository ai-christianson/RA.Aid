"""Utility functions for working with agents."""

import time
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_core.messages import BaseMessage
from anthropic import APIError, APITimeoutError, RateLimitError, InternalServerError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ra_aid.tools.memory import _global_memory

console = Console()

def print_agent_output(chunk: dict[str, BaseMessage]) -> None:
    """Print agent output chunks."""
    if chunk.get("delta") and chunk["delta"].content:
        console.print(chunk["delta"].content, end="", style="blue")

def print_error(msg: str) -> None:
    """Print error messages."""
    console.print(f"\n{msg}", style="red")

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
                
                # Check for task completion after each chunk
                if _global_memory.get('task_completed'):
                    completion_msg = _global_memory.get('completion_message', 'Task was completed successfully.')
                    console.print(Panel(
                        Markdown(completion_msg),
                        title="✅ Task Completed",
                        style="green"
                    ))
                    return completion_msg
            break
        except (InternalServerError, APITimeoutError, RateLimitError, APIError) as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Max retries ({max_retries}) exceeded. Last error: {str(e)}")
            
            delay = base_delay * (2 ** attempt)  # Exponential backoff
            error_type = e.__class__.__name__
            print_error(f"Encountered {error_type}: {str(e)}. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
            continue
