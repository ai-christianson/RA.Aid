from langchain_core.tools import tool
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

def create_keybindings():
    """Create custom key bindings for Ctrl+D submission."""
    bindings = KeyBindings()

    @bindings.add('c-d')
    def submit(event):
        """Trigger submission when Ctrl+D is pressed."""
        event.current_buffer.validate_and_handle()

    return bindings

@tool
def ask_human(question: str) -> str:
    """Ask the human user a question with a nicely formatted display.
    
    Args:
        question: The question to ask the human user (supports markdown)
        
    Returns:
        The user's response as a string
    """
    console.print(Panel(
        Markdown(question + "\n\n*Multiline input is supported; use Ctrl+D to submit. Use Ctrl+C to exit the program.*"),
        title="💭 Question for Human",
        border_style="yellow bold"
    ))

    session = PromptSession(
        multiline=True,
        key_bindings=create_keybindings(),
        prompt_continuation='. ',
    )

    print()
    
    response = session.prompt(
        "> ",
        wrap_lines=True
    )

    print()
    return response
