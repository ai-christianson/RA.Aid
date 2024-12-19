from rich.console import Console
from rich.rule import Rule
from rich.panel import Panel
from rich.markdown import Markdown
from rich.padding import Padding

console = Console()

def print_stage_header(stage: str) -> None:
    """Print a stage header with stage-specific styling and icons.
    
    Args:
        stage: The stage name to print (automatically formatted to Title Case)
    """
    # Define stage icons mapping - using single-width emojis to prevent line wrapping issues
    icons = {
        'research stage': '🔎',
        'planning stage': '📝',
        'implementation stage': '🔧',  # Changed from 🛠️ to prevent wrapping
        'task completed': '✅',
        'debug stage': '🐛',
        'testing stage': '🧪',
        'research subtasks': '📚',
        'skipping implementation stage': '⏭️'
    }

    # Format stage name to Title Case and normalize for mapping lookup
    stage_title = stage.title()
    stage_key = stage.lower()
    
    # Get appropriate icon with fallback
    icon = icons.get(stage_key, '🚀')
    
    # Create styled rule with icon
    rule_content = f"{icon} {stage_title}"
    padded_rule = Padding(Rule(rule_content, style="green bold"), (1, 0))
    console.print(padded_rule)

def print_task_header(task: str) -> None:
    """Print a task header with yellow styling and wrench emoji. Content is rendered as Markdown.
    
    Args:
        task: The task text to print (supports Markdown formatting)
    """
    console.print(Panel(Markdown(task), title="🔧 Task", border_style="yellow bold"))

def print_error(message: str) -> None:
    """Print an error message in a red-bordered panel with warning emoji.
    
    Args:
        message: The error message to display (supports Markdown formatting)
    """
    console.print(Panel(Markdown(message), title="Error", border_style="red bold"))
