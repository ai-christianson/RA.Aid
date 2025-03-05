from langchain_core.tools import tool
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ra_aid.logging_config import get_logger

logger = get_logger(__name__)
console = Console()


def create_keybindings():
    """Create custom key bindings for Ctrl+D submission."""
    bindings = KeyBindings()

    @bindings.add("c-d")
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
    console.print(
        Panel(
            Markdown(
                question
                + "\n\n*Multiline input is supported; use Ctrl+D to submit. Use Ctrl+C to exit the program.*"
            ),
            title="💭 Question for Human",
            border_style="yellow bold",
        )
    )

    session = PromptSession(
        multiline=True,
        key_bindings=create_keybindings(),
        prompt_continuation=". ",
    )

    print()

    response = session.prompt("> ", wrap_lines=True)
    print()
    
    # Record human response in database
    try:
        from ra_aid.database.repositories.human_input_repository import get_human_input_repository
        from ra_aid.database.repositories.config_repository import get_config_repository
        
        # Determine the source based on context
        config = get_config_repository().get_all()
        # If chat_mode is enabled, use 'chat', otherwise determine if hil mode is active
        if config.get("chat_mode", False):
            source = "chat"
        elif config.get("hil", False):
            source = "hil"
        else:
            source = "chat"  # Default fallback
            
        # Get the repository from context and store the input
        human_input_repo = get_human_input_repository()
        human_input_repo.create(content=response, source=source)
        
        # Run garbage collection to ensure we don't exceed 100 inputs
        human_input_repo.garbage_collect()
    except RuntimeError as e:
        logger.error(f"Failed to record human input: No HumanInputRepository available in context. {str(e)}")
    except Exception as e:
        logger.error(f"Failed to record human input: {str(e)}")
    
    return response