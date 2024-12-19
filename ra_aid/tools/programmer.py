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
    """Assign a programming task to a human programmer.

Before using this tool, ensure all related files have been emitted with emit_related_files.

The programmer sees only what you provide, no conversation history.

Give detailed instructions but do not write their code.

They are intelligent and can edit multiple files.

If new files are created, emit them after finishing.

They can add/modify files, but not remove. Use run_shell_command to remove files. If referencing files you’ll delete, remove them after they finish.

Args: instructions: Programming task instructions files: Optional; if not provided, uses related_files

Returns: { "output": stdout+stderr, "return_code": 0 if success, "success": True/False }
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
