"""Simplified mock of agent_utils.py for testing binary file detection.

This file includes typical Python constructs like imports, functions, classes, and docstrings
to replicate the characteristics of the real agent_utils.py that's causing issues with
binary file detection.
"""

import os
import sys
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal, Sequence

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# Define a logger
logger = None  # In real code, this would be an actual logger


class MockAgent:
    """Mock agent class to simulate the real agent class structure."""

    def __init__(self, model=None, tools=None, max_tokens=4096, config=None):
        """Initialize a mock agent.

        Args:
            model: The language model to use
            tools: List of tools available to the agent
            max_tokens: Maximum tokens to use in context
            config: Additional configuration
        """
        self.model = model
        self.tools = tools or []
        self.max_tokens = max_tokens
        self.config = config or {}
        self._initialized = True
    
    def run(self, input_text, config=None):
        """Run the agent on input text.
        
        Args:
            input_text: The text to process
            config: Optional runtime configuration
            
        Returns:
            Mock agent response
        """
        # Simulate processing with a delay
        time.sleep(0.1)
        return f"Processed: {input_text[:20]}..."
    
    @staticmethod
    def _estimate_tokens(text):
        """Estimate number of tokens in text.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count (roughly 1 token per 4 characters)
        """
        return len(text) // 4


def run_mock_agent(task: str, model=None, **kwargs) -> Optional[str]:
    """Run a mock agent on a task.

    This function creates a new agent, sets up tools, and runs the agent on the task.
    It includes various parameters and logic to mimic the complexity of the real agent_utils.py.

    Args:
        task: The task to process
        model: The model to use
        **kwargs: Additional keyword arguments
        
    Returns:
        Optional[str]: Result from the agent
    """
    # Create a unique ID for this run
    run_id = str(uuid.uuid4())
    
    # Set up mock console for output
    console = Console()
    
    # Log the start of execution
    console.print(Panel(Markdown(f"Starting agent with ID: {run_id}"), title="🤖 Agent"))
    
    # Setup some complex nested data structures to mimic real code
    memory = {
        "task_history": [],
        "agent_state": {
            "initialized": True,
            "tools_enabled": True
        },
        "config": {
            "max_retries": 3,
            "timeout": 30,
            "debug": False
        }
    }
    
    # Track some metrics
    metrics = {
        "start_time": datetime.now(),
        "steps": 0,
        "tokens_used": 0
    }
    
    # Create a mock agent
    agent = MockAgent(model=model, config=kwargs.get("config"))
    
    try:
        # Process the task
        memory["task_history"].append(task)
        metrics["steps"] += 1
        
        # Simulate token counting
        task_tokens = MockAgent._estimate_tokens(task)
        metrics["tokens_used"] += task_tokens
        
        # Check if we should short-circuit for any reason
        if task.lower() == "exit" or task.lower() == "quit":
            return "Exit requested"
        
        # Run the main agent logic
        result = agent.run(task)
        
        # Update completion time
        metrics["end_time"] = datetime.now()
        metrics["duration"] = (metrics["end_time"] - metrics["start_time"]).total_seconds()
        
        # Generate a fancy completion message with some complex formatting
        completion_message = f"""
## Agent Run Complete

- **Task**: {task[:50]}{"..." if len(task) > 50 else ""}
- **Duration**: {metrics["duration"]:.2f}s
- **Tokens**: {metrics["tokens_used"]}
- **Steps**: {metrics["steps"]}
- **Result**: Success
        """
        
        console.print(Panel(Markdown(completion_message), title="✅ Complete"))
        
        return result
    except Exception as e:
        # Handle errors
        error_message = f"Agent failed: {str(e)}"
        console.print(Panel(error_message, title="❌ Error", style="red"))
        return None


def calculate_something_complex(a: int, b: int, operation: str = "add") -> int:
    """Calculate something using the specified operation.
    
    Args:
        a: First number
        b: Second number
        operation: Operation to perform (add, subtract, multiply, divide)
        
    Returns:
        Result of the calculation
        
    Raises:
        ValueError: If operation is invalid
    """
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")


class DataProcessor:
    """Example class that processes data in various ways."""
    
    def __init__(self, data: List[Any]):
        """Initialize with data.
        
        Args:
            data: List of data to process
        """
        self.data = data
        self.processed = False
        self.results = {}
    
    def process(self, method: str = "default"):
        """Process the data using specified method.
        
        Args:
            method: Processing method to use
            
        Returns:
            Processed data
        """
        if method == "default":
            result = [item for item in self.data if item is not None]
        elif method == "sum":
            result = sum(self.data)
        elif method == "count":
            result = len(self.data)
        else:
            result = self.data
            
        self.results[method] = result
        self.processed = True
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the data.
        
        Returns:
            Dictionary of statistics
        """
        if not self.data:
            return {"count": 0, "empty": True}
            
        return {
            "count": len(self.data),
            "empty": len(self.data) == 0,
            "methods_used": list(self.results.keys()),
            "has_nulls": any(item is None for item in self.data)
        }
    
    def __str__(self):
        """String representation.
        
        Returns:
            String describing the processor
        """
        return f"DataProcessor(items={len(self.data)}, processed={self.processed})"


# Add some multi-line strings with various quotes and formatting
TEMPLATE = """
# Agent Report

## Overview
This report was generated by the agent system.

## Details
- Task: {task}
- Date: {date}
- Status: {status}

## Summary
{summary}
"""

SQL_QUERY = '''
SELECT *
FROM users
WHERE status = 'active'
  AND last_login > '2023-01-01'
ORDER BY last_login DESC
LIMIT 10;
'''

# Regular expression pattern with escapes
PATTERN = r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("

# Add a global dictionary with mixed types
GLOBAL_CONFIG = {
    "debug": False,
    "max_retries": 3,
    "timeout": 30,
    "endpoints": ["api/v1", "api/v2"],
    "rate_limits": {
        "minute": 60,
        "hour": 3600,
        "day": 86400
    },
    "features": {
        "experimental": True,
        "beta_tools": False
    }
}

# Some unusual unicode characters to ensure encoding handling
UNICODE_EXAMPLE = "Hello 世界! This has unicode: ™ ® © ♥ ⚡ ☁ ☀"


def main():
    """Main function to demonstrate the module."""
    # Run a simple example
    result = run_mock_agent("Test the mock agent")
    print(f"Result: {result}")
    
    # Try the data processor
    processor = DataProcessor([1, 2, 3, None, 5])
    processor.process("default")
    stats = processor.get_stats()
    print(f"Stats: {stats}")
    
    # Do a calculation
    calc = calculate_something_complex(10, 5, "multiply")
    print(f"Calculation: {calc}")


if __name__ == "__main__":
    main()