from rich.markdown import Markdown
from rich.panel import Panel
from typing import Optional

from ra_aid.console.common import console
from ra_aid.database.repositories.trajectory_repository import get_trajectory_repository
from ra_aid.database.repositories.human_input_repository import (
    get_human_input_repository,
)


def cpm(message: str, title: Optional[str] = None, border_style: str = "blue", subtitle: Optional[str] = None) -> None:
    """
    Print a message using a Panel with Markdown formatting.

    Args:
        message (str): The message content to display.
        title (Optional[str]): An optional title for the panel.
        border_style (str): Border style for the panel.
        subtitle (Optional[str]): An optional subtitle for the panel. If None, will try to get cost subtitle.
    """
    from ra_aid.console.output import get_cost_subtitle
    
    if subtitle is None:
        subtitle = get_cost_subtitle()
        
    console.print(Panel(
        Markdown(message), 
        title=title, 
        border_style=border_style,
        subtitle=subtitle,
        subtitle_align="right" if subtitle else None
    ))


def console_panel(
    message: str, 
    title: Optional[str] = None, 
    border_style: str = "blue",
    subtitle: Optional[str] = None,
    subtitle_align: str = "right",
    padding: tuple = (0, 0),
    expand: bool = True,
    safe_box: bool = True,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
    """
    Print a message using a Panel without Markdown formatting, with all panel options available.

    Args:
        message (str): The message content to display.
        title (Optional[str]): An optional title for the panel.
        border_style (str): Border style for the panel.
        subtitle (Optional[str]): An optional subtitle for the panel. If None, will try to get cost subtitle.
        subtitle_align (str): Alignment for the subtitle ("left", "center", or "right").
        padding (tuple): Padding for the panel content (vertical, horizontal).
        expand (bool): Whether the panel should expand to fill available width.
        safe_box (bool): Whether to use ASCII characters instead of Unicode for the box.
        width (Optional[int]): Optional fixed width for the panel.
        height (Optional[int]): Optional fixed height for the panel.
    """
    from ra_aid.console.output import get_cost_subtitle
    
    if subtitle is None:
        subtitle = get_cost_subtitle()
        
    console.print(
        Panel(
            message,
            title=title,
            border_style=border_style,
            subtitle=subtitle,
            subtitle_align=subtitle_align if subtitle else None,
            padding=padding,
            expand=expand,
            safe_box=safe_box,
            width=width,
            height=height,
        )
    )


def print_stage_header(stage: str) -> None:
    """Print a stage header with stage-specific styling and icons.

    Args:
        stage: The stage name to print (automatically formatted to Title Case)
    """
    # Define stage icons mapping - using single-width emojis to prevent line wrapping issues
    icons = {
        "research stage": "🔎",
        "planning stage": "📝",
        "implementation stage": "🔧",  # Changed from 🛠️ to prevent wrapping
        "task completed": "✅",
        "debug stage": "🐛",
        "testing stage": "🧪",
        "research subtasks": "📚",
        "skipping implementation stage": "⏭️",
    }

    # Format stage name to Title Case and normalize for mapping lookup
    stage_title = stage.title()
    stage_key = stage.lower()

    # Get appropriate icon with fallback
    icon = icons.get(stage_key, "🚀")

    # Create styled panel with icon
    panel_content = f" {icon} {stage_title}"
    console_panel(panel_content, border_style="green bold", padding=(0, 1))


def print_task_header(task: str) -> None:
    """Print a task header with yellow styling and wrench emoji. Content is rendered as Markdown.

    Args:
        task: The task text to print (supports Markdown formatting)
    """
    cpm(task, title="🔧 Task", border_style="yellow bold")


def print_error(message: str) -> None:
    """Print an error message in a red-bordered panel with warning emoji.

    Args:
        message: The error message to display (supports Markdown formatting)
    """
    cpm(message, title="Error", border_style="red bold")


def print_warning(message: str, title: str = "Warning") -> None:
    """Print a warning message in an amber-bordered panel with warning emoji.

    Uses a text-only title to prevent console formatting issues.

    Args:
        message: The warning message to display (supports Markdown formatting)
        title: The title for the panel, defaults to "Warning"
    """
    cpm(message, title=title, border_style="yellow bold")


def print_interrupt(message: str) -> None:
    """Print an interrupt message in a yellow-bordered panel with stop emoji.

    Args:
        message: The interrupt message to display (supports Markdown formatting)
    """
    print()  # Add spacing for ^C
    cpm(message, title="⛔ Interrupt", border_style="yellow bold")
