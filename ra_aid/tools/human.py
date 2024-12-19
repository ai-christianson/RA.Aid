"""Tool for asking questions to the human user."""

from langchain.tools import tool
from rich.console import Console
from rich.prompt import Prompt

console = Console()

@tool
def ask_human(question: str) -> str:
    """Ask the human user a question and get their response.
    
    Args:
        question: The question to ask the human user
        
    Returns:
        The user's response as a string
    """
    console.print(f"\n[bold yellow]Human Query:[/] {question}")
    response = Prompt.ask("Your response")
    return response
