import subprocess
from typing import List, Optional, Dict, Union, Set
from langchain_core.tools import tool
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.text import Text
from ra_aid.proc.interactive import run_interactive_command
from pydantic import BaseModel, Field
from ra_aid.text.processing import truncate_output

console = Console()


class RunProgrammingTaskInput(BaseModel):
    instructions: str = Field(description="Instructions for the programming task")
    files: Optional[List[str]] = Field(None, description="Optional list of files for Aider to examine")

@tool
def run_programming_task(input: RunProgrammingTaskInput) -> Dict[str, Union[str, int, bool]]:
    """Assigns a programming task to a human programmer.
    
    Before invoking this tool, make sure all existing related files have been emitted using the emit_related_files tool.

    The programmer cannot see files you have seen and only can see what you explicitly give it.

    Be very detailed in your instructions, but do not write the full code for the programmer, as that's the job of the programmer.

    The programmer can edit multiple files at once and is intelligent.

    If any new files are created, remember to emit them using the emit_related_files tool once this tool completes.

    The programmer can add or modify files, but cannot remove them. Use the run_shell_command tool to remove files.
      If you need the programmer to reference files you intend to delete, delete them after the programmer finises their work.
    
    Args:
        instructions: Instructions for the programming task
        files: Optional list of files for Aider to examine. If not provided, uses related_files.
        
    Returns:
        A dictionary containing:
            - output: The command output (stdout + stderr combined)
            - return_code: The process return code (0 typically means success)
            - success: Boolean indicating if the command succeeded
    """
    # Build command
    command = [
        "aider",
        "--yes-always",
        "--no-auto-commits",
        "--dark-mode",
        "--no-suggest-shell-commands",
        "-m"
    ]
    
    command.append(input.instructions)
    
    if input.files:
        command.extend(input.files)
        
    # Create a pretty display of what we're doing
    task_display = [
        "## Instructions\n",
        f"{input.instructions}\n"
    ]
    
    if input.files:
        task_display.extend([
            "\n## Files\n",
            *[f"- `{file}`\n" for file in input.files]
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

# Export the functions
__all__ = ['run_programming_task']
