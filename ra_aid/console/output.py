from typing import Any, Dict, Literal, Optional

from langchain_core.messages import AIMessage
from rich.markdown import Markdown
from rich.panel import Panel

from ra_aid.exceptions import ToolExecutionError

# Import shared console instance
from .formatting import console


def print_agent_output(
    chunk: Dict[str, Any], agent_type: Literal["CiaynAgent", "React"]
) -> None:
    """Print only the agent's message content, not tool calls.

    Args:
        chunk: A dictionary containing agent or tool messages
    """
    if "agent" in chunk and "messages" in chunk["agent"]:
        messages = chunk["agent"]["messages"]
        for msg in messages:
            if isinstance(msg, AIMessage):
                # Handle text content
                if isinstance(msg.content, list):
                    for content in msg.content:
                        if content["type"] == "text" and content["text"].strip():
                            console.print(
                                Panel(Markdown(content["text"]), title="🤖 Assistant")
                            )
                else:
                    if msg.content.strip():
                        console.print(
                            Panel(Markdown(msg.content.strip()), title="🤖 Assistant")
                        )
    elif "tools" in chunk and "messages" in chunk["tools"]:
        for msg in chunk["tools"]["messages"]:
            if msg.status == "error" and msg.content:
                err_msg = msg.content.strip()
                console.print(
                    Panel(
                        Markdown(err_msg),
                        title="❌ Tool Error",
                        border_style="red bold",
                    )
                )
                tool_name = getattr(msg, "name", None)

                # CiaynAgent handles this internally
                if agent_type == "React":
                    raise ToolExecutionError(
                        err_msg, tool_name=tool_name, base_message=msg
                    )


def cpm(message: str, title: Optional[str] = None, border_style: str = "blue") -> None:
    """
    Print a message using a Panel with Markdown formatting.

    Args:
        message (str): The message content to display.
        title (Optional[str]): An optional title for the panel.
        border_style (str): Border style for the panel.
    """

    console.print(Panel(Markdown(message), title=title, border_style=border_style))
