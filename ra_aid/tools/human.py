import os
import re
from langchain_core.tools import tool
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ra_aid.logging_config import get_logger

logger = get_logger(__name__)
console = Console()

# Supported image extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}


def normalize_image_paths(text: str) -> str:
    """Convert drag-dropped file paths to the standard @"/path/to/image.png" format.

    When a file is dragged into the terminal, it pastes a backslash-escaped path.
    This function detects that and reformats it properly.
    """
    text = text.strip()

    # Already has @ prefix - leave it alone
    if '@' in text:
        return text

    # Check if this looks like a bare image path (starts with / or ~/, ends with image ext)
    lower_text = text.lower()
    is_image_path = any(lower_text.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp'])

    if is_image_path and (text.startswith('/') or text.startswith('~/')):
        # Unescape backslash-escaped characters
        unescaped = text.replace('\\ ', ' ').replace('\\', '')

        # Verify the file exists
        expanded = os.path.expanduser(unescaped)
        if os.path.isfile(expanded):
            logger.debug(f"Normalized drag-drop path: {unescaped}")
            return f'@"{unescaped}"'

    return text


# Patterns to match image paths:
# 1. Quoted paths (handles spaces): @"/path/to/my image.png" or @'/path/to/image.png'
# 2. Unquoted paths (no spaces): @/path/to/image.png
# 3. Bare paths (drag-and-drop): /path/to/image.png or "/path/to/image.png"
IMAGE_PATH_PATTERNS = [
    # Double-quoted paths with @: @"/path with spaces/image.png"
    re.compile(r'@"([^"]+\.(?:png|jpg|jpeg|gif|webp))"', re.IGNORECASE),
    # Single-quoted paths with @: @'/path with spaces/image.png'
    re.compile(r"@'([^']+\.(?:png|jpg|jpeg|gif|webp))'", re.IGNORECASE),
    # Unquoted paths with @: @/path/to/image.png
    re.compile(r'@((?:/[^\s]+|~/[^\s]+)\.(?:png|jpg|jpeg|gif|webp))', re.IGNORECASE),
    # Bare double-quoted paths (drag-drop with spaces): "/path with spaces/image.png"
    re.compile(r'"(/[^"]+\.(?:png|jpg|jpeg|gif|webp))"', re.IGNORECASE),
    # Bare single-quoted paths (drag-drop with spaces): '/path with spaces/image.png'
    re.compile(r"'(/[^']+\.(?:png|jpg|jpeg|gif|webp))'", re.IGNORECASE),
    # Bare unquoted absolute paths (drag-drop): /path/to/image.png
    re.compile(r'(?:^|\s)(/[^\s"\']+\.(?:png|jpg|jpeg|gif|webp))(?:\s|$)', re.IGNORECASE),
    # Bare paths with ~ (drag-drop): ~/path/to/image.png
    re.compile(r'(?:^|\s)(~/[^\s"\']+\.(?:png|jpg|jpeg|gif|webp))(?:\s|$)', re.IGNORECASE),
]


def transform_pasted_path(text: str) -> str:
    """Transform a pasted/drag-dropped path to @"/path" format if it's an image."""
    text = text.strip()

    # Already formatted or not a path
    if '@' in text or not (text.startswith('/') or text.startswith('~/')):
        return text

    # Check if it ends with an image extension
    lower = text.lower()
    if not any(lower.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
        return text

    # Unescape backslash-escaped characters (from terminal drag-drop)
    unescaped = text.replace('\\ ', ' ').replace('\\,', ',').replace('\\', '')

    # Verify file exists
    expanded = os.path.expanduser(unescaped)
    if os.path.isfile(expanded):
        return f'@"{unescaped}"'

    return text


def create_keybindings():
    """Create custom key bindings for input handling."""
    bindings = KeyBindings()

    @bindings.add("enter")
    def submit(event):
        """Submit on Enter."""
        event.current_buffer.validate_and_handle()

    @bindings.add("c-j")
    def newline(event):
        """Insert newline on Ctrl+J."""
        event.current_buffer.insert_text("\n")

    @bindings.add("c-c")
    def clear_line(event):
        """Clear current line on Ctrl+C."""
        event.current_buffer.reset()

    @bindings.add(Keys.BracketedPaste)
    def handle_paste(event):
        """Handle bracketed paste (drag-drop) and transform image paths."""
        data = event.data
        transformed = transform_pasted_path(data)
        event.current_buffer.insert_text(transformed)

    return bindings


def extract_image_paths(text: str) -> tuple[str, list[str]]:
    """Extract image paths from text and return cleaned text + paths.

    Supports:
    - @"/path/with spaces/image.png" (quoted for spaces)
    - @'/path/with spaces/image.png' (single quotes)
    - @/path/to/image.png (no spaces)
    - ~/path/from/home/image.jpg
    - Drag-and-drop: /path/to/image.png (bare paths)
    - Drag-and-drop: "/path with spaces/image.png" (quoted bare paths)

    Args:
        text: The input text potentially containing image paths

    Returns:
        Tuple of (cleaned_text, list_of_valid_image_paths)
    """
    valid_paths = []
    cleaned_text = text

    # Track what we've already matched to avoid duplicates
    matched_paths = set()

    for pattern in IMAGE_PATH_PATTERNS:
        for match in pattern.finditer(text):
            path = match.group(1)
            full_match = match.group(0)

            # Expand ~ to home directory
            expanded_path = os.path.expanduser(path)

            if expanded_path in matched_paths:
                continue

            if os.path.isfile(expanded_path):
                valid_paths.append(expanded_path)
                matched_paths.add(expanded_path)
                # For bare paths, only remove the path itself (not surrounding whitespace)
                # For @-prefixed and quoted paths, remove the full match
                if full_match.strip() == path or full_match.strip() == f'"{path}"' or full_match.strip() == f"'{path}'":
                    # Bare path - remove just the path (preserve any surrounding context)
                    cleaned_text = cleaned_text.replace(path, "")
                    # Also remove quotes if present
                    cleaned_text = cleaned_text.replace(f'"{path}"', "")
                    cleaned_text = cleaned_text.replace(f"'{path}'", "")
                else:
                    # @-prefixed path - remove the full match including @
                    cleaned_text = cleaned_text.replace(full_match, "")
                logger.debug(f"Found valid image path: {expanded_path}")
            else:
                logger.debug(f"Image path not found: {expanded_path}")

    # Clean up extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text, valid_paths


def promote_to_expert_with_images(question: str, image_paths: list[str]) -> str:
    """Send a question with images to the expert model.
    
    Args:
        question: The user's question
        image_paths: List of valid image file paths
        
    Returns:
        The expert model's response
    """
    from ra_aid.tools.expert_image import ask_expert_with_image
    
    console.print(
        Panel(
            f"🖼️ Detected {len(image_paths)} image(s) - promoting to expert model (Claude Opus)\n"
            + "\n".join(f"  • {p}" for p in image_paths),
            title="Auto-Promotion",
            border_style="magenta"
        )
    )
    
    # Call the expert image tool directly (not as a langchain tool)
    return ask_expert_with_image.func(question=question, image_paths=image_paths)


@tool
def ask_human(question: str) -> str:
    """Ask the human user a question with a nicely formatted display.

    Args:
        question: The question to ask the human user (supports markdown)

    Returns:
        The user's response as a string
        
    Note:
        If the user includes image paths in their response, the request will 
        automatically be promoted to the expert model (Claude Opus) which has 
        vision capabilities.
        
        For paths with spaces, use quotes: @"/path/with spaces/image.png"
    """
    console.print(
        Panel(
            Markdown(
                question
                + "\n\n*Enter to submit. Ctrl+J for new line. Ctrl+C to clear. Type `exit!!` to quit.*\n"
                + '*Drag images or use @"/path/to/image.png"*'
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

    # Check for exit command
    if response.strip() == "exit!!":
        console.print("👋 Bye!")
        import sys
        sys.exit(0)

    # Normalize image paths (convert drag-drop formats to @"/path/to/image.png")
    response = normalize_image_paths(response)

    # Record human response in database
    try:
        from ra_aid.database.repositories.human_input_repository import get_human_input_repository
        from ra_aid.database.repositories.config_repository import get_config_repository
        
        if get_config_repository().get("chat_mode", False):
            source = "chat"
        elif get_config_repository().get("hil", False):
            source = "hil"
        else:
            source = "chat"
            
        human_input_repo = get_human_input_repository()
        human_input_repo.create(content=response, source=source)
        human_input_repo.garbage_collect()
    except RuntimeError as e:
        logger.error(f"Failed to record human input: No HumanInputRepository available in context. {str(e)}")
    except Exception as e:
        logger.error(f"Failed to record human input: {str(e)}")
    
    # Check for image paths in the response
    cleaned_text, image_paths = extract_image_paths(response)
    
    if image_paths:
        # Auto-promote to expert model with images
        logger.info(f"Promoting to expert model with {len(image_paths)} image(s)")
        expert_response = promote_to_expert_with_images(cleaned_text, image_paths)

        # Return both the original request context and expert's analysis
        # Include instruction to continue the conversation
        return f"""[User provided {len(image_paths)} image(s) for analysis]

User's question: {cleaned_text}

Expert analysis (from Claude Opus with vision):
{expert_response}

[Image analysis complete. Call ask_human to check if the user needs anything else or has follow-up questions.]"""
    
    return response
