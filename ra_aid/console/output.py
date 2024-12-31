from typing import Dict, Any
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from ra_aid.console.formatting import print_error

console = Console()

def print_agent_output(chunk: Dict[str, Any]) -> None:
    """Print only the agent's message content, not tool calls.
    
    Args:
        chunk: A dictionary containing agent or tool messages
    """
    if 'content' in chunk:
        # Handle direct conversation responses
        if chunk['content'].strip():
            console.print(Panel(Markdown(chunk['content'].strip()), title="🤖 Assistant"))
    elif 'agent' in chunk and 'messages' in chunk['agent']:
        messages = chunk['agent']['messages']
        for msg in messages:
            if isinstance(msg, AIMessage):
                # Handle text content
                if isinstance(msg.content, list):
                    for content in msg.content:
                        if content['type'] == 'text' and content['text'].strip():
                            console.print(Panel(Markdown(content['text']), title="🤖 Assistant"))
                else:
                    if msg.content.strip():
                        console.print(Panel(Markdown(msg.content.strip()), title="🤖 Assistant"))
    elif 'tools' in chunk and 'messages' in chunk['tools']:
        for msg in chunk['tools']['messages']:
            if isinstance(msg, dict):
                if msg.get('status') == 'error' and msg.get('content'):
                    print_error(msg['content'])
            elif hasattr(msg, 'status') and msg.status == 'error' and msg.content:
                print_error(msg.content)