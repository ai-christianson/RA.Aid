import os
from typing import List, Dict, Union
from ra_aid.tools.memory import _global_memory
from langchain_core.tools import tool
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from ra_aid.proc.interactive import run_interactive_command
from ra_aid.text.processing import truncate_output

console = Console()

@tool
def run_programming_task(instructions: str, files: List[str] = []) -> Dict[str, Union[str, int, bool]]:
    """Assign a programming task to a human programmer. Use this instead of trying to write code to files yourself.

Before using this tool, ensure all related files have been emitted with emit_related_files.

The programmer sees only what you provide, no conversation history.

Give detailed instructions but do not write their code.

They are intelligent and can edit multiple files.

If new files are created, emit them after finishing.

They can add/modify files, but not remove. Use run_shell_command to remove files. If referencing files you’ll delete, remove them after they finish.

Args:
 instructions: REQUIRED Programming task instructions (markdown format, use newlines and as many tokens as needed)
 files: Optional; if not provided, uses related_files

Returns: { "output": stdout+stderr, "return_code": 0 if success, "success": True/False }
    """
    # Get related files if no specific files provided
    file_paths = list(_global_memory['related_files'].values()) if 'related_files' in _global_memory else []

    # Build command
    command = [
        "aider",
        "--yes-always",
        "--no-auto-commits",
        "--dark-mode",
        "--no-suggest-shell-commands",
    ]

    # Add config file if specified
    if 'config' in _global_memory and _global_memory['config'].get('aider_config'):
        command.extend(['--config', _global_memory['config']['aider_config']])

    # if environment variable AIDER_FLAGS exists then parse
    if 'AIDER_FLAGS' in os.environ:
        # wrap in try catch in case of any error and log the error
        try:
            command.extend(parse_aider_flags(os.environ['AIDER_FLAGS']))
        except Exception as e:
            print(f"Error parsing AIDER_FLAGS: {e}")

    # ensure message aider argument is always present
    command.append("-m")

    command.append(instructions)

    # Add files to command
    files_to_use = file_paths + (files or [])
    if files_to_use:
        command.extend(files_to_use)

    # Create a pretty display of what we're doing
    task_display = [
        "## Instructions\n",
        f"{instructions}\n"
    ]

    if files_to_use:
        task_display.extend([
            "\n## Files\n",
            *[f"- `{file}`\n" for file in files_to_use]
        ])

    markdown_content = "".join(task_display)
    console.print(Panel(Markdown(markdown_content), title="🤖 Aider Task", border_style="bright_blue"))

    try:
        # Run the command interactively
        print()
        output, return_code = run_interactive_command(command)
        print()

        # Return structured output
        return {
            "output": truncate_output(output.decode() if output else ""),
            "return_code": return_code,
            "success": return_code == 0
        }

    except Exception as e:
        print()
        error_text = Text()
        error_text.append("Error running programming task:\n", style="bold red")
        error_text.append(str(e), style="red")
        console.print(error_text)

        return {
            "output": str(e),
            "return_code": 1,
            "success": False
        }

def parse_aider_flags(aider_flags: str) -> List[str]:
    """Parse a string of aider flags into a list of flags.

    Args:
        aider_flags: A string containing comma-separated flags, with or without leading dashes.
                    Can contain spaces around flags and commas.

    Returns:
        A list of flags with proper '--' prefix.

    Examples:
        >>> parse_aider_flags("yes-always,dark-mode")
        ['--yes-always', '--dark-mode']
        >>> parse_aider_flags("--yes-always, --dark-mode")
        ['--yes-always', '--dark-mode']
        >>> parse_aider_flags("")
        []
    """
    if not aider_flags.strip():
        return []

    # Split by comma and strip whitespace
    flags = [flag.strip() for flag in aider_flags.split(",")]

    # Add '--' prefix if not present and filter out empty flags
    return [f"--{flag.lstrip('-')}" for flag in flags if flag.strip()]

# Export the functions
__all__ = ['run_programming_task']
