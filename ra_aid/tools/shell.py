from typing import Dict, Union
from langchain_core.tools import tool
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from ra_aid.tools.memory import _global_memory
from ra_aid.proc.interactive import run_interactive_command
from ra_aid.text.processing import truncate_output
from ra_aid.console.cowboy_messages import get_cowboy_message

console = Console()

@tool
def run_shell_command(command: str) -> Dict[str, Union[str, int, bool]]:
    """Execute a shell command and return its output.

    Important notes:
    1. Try to constrain/limit the output. Output processing is expensive, and infinite/looping output will cause us to fail.
    2. When using commands like 'find', 'grep', or similar recursive search tools, always exclude common 
       development directories and files that can cause excessive output or slow performance:
       - Version control: .git
       - Dependencies: node_modules, vendor, .venv
       - Cache: __pycache__, .cache
       - Build: dist, build
       - Environment: .env, venv, env
       - IDE: .idea, .vscode
    3. Avoid doing recursive lists, finds, etc. that could be slow and have a ton of output. Likewise, avoid flags like '-l' that needlessly increase the output. But if you really need to, you can.
    4. Add flags e.g. git --no-pager in order to reduce interaction required by the human.
    """
    # Check if we need approval
    cowboy_mode = _global_memory.get('config', {}).get('cowboy_mode', False)
    
    if cowboy_mode:
        console.print("")
        console.print(" " + get_cowboy_message())
        console.print("")

    # Show just the command in a simple panel
    console.print(Panel(command, title="🐚 Shell", border_style="bright_yellow"))
    
    if not cowboy_mode:
        choices = ["y", "n", "c"]
        response = Prompt.ask(
            "Execute this command? (y=yes, n=no, c=enable cowboy mode for session)",
            choices=choices,
            default="y",
            show_choices=True,
            show_default=True
        )
        
        if response == "n":
            print()
            return {
                "output": "Command execution cancelled by user",
                "return_code": 1,
                "success": False
            }
        elif response == "c":
            _global_memory['config']['cowboy_mode'] = True
            console.print("")
            console.print(" " + get_cowboy_message())
            console.print("")
    
    try:
        print()
        output, return_code = run_interactive_command(['/bin/bash', '-c', command])
        print()
        return {
            "output": truncate_output(output.decode()) if output else "",
            "return_code": return_code,
            "success": return_code == 0
        }
    except Exception as e:
        print()
        console.print(Panel(str(e), title="❌ Error", border_style="red"))
        return {
            "output": str(e),
            "return_code": 1,
            "success": False
        }
