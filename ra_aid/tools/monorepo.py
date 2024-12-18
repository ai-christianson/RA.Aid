from langchain_core.tools import tool
from rich.console import Console
from rich.panel import Panel

console = Console()

@tool("monorepo_detected")
def monorepo_detected() -> dict:
    """
    Returns a hint message indicating monorepo detection.
    
    Returns:
        dict: Contains a 'hint' key with the detection message
    """
    console.print(Panel(
        "Mono repo detected.",
        title="🏢 Monorepo Detected",
        border_style="bright_blue"
    ))
    return {
        'hint': 'Found indicators of a monorepo structure. Consider searching across all modules and packages for related code.'
    }
