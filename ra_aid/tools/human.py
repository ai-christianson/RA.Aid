"""Tool for asking questions to the human user."""

from langchain_core.tools import tool
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

@tool
def ask_human(question: str) -> str:
    """Ask the human user a question with a nicely formatted display.
    
    Args:
        question: The question to ask the human user (supports markdown)
        
    Returns:
        The user's response as a string
    """
    console.print(Panel(
        Markdown(question),
        title="💭 Question for Human",
        border_style="yellow bold"
    ))
    response = Prompt.ask("\nYour response")
    print()
    return response
