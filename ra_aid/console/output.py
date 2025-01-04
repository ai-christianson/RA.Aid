from typing import Dict, Any, Optional
from rich.console import Console
from rich.syntax import Syntax
from ra_aid.console.formatting import print_error
import json
import streamlit as st
import asyncio
import os
from pathlib import Path

console = Console()

def ensure_directory_exists(path: str) -> None:
    """Ensure a directory exists, create it if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)

def format_json_response(content: str) -> Dict[str, Any]:
    """Format content as a structured JSON response."""
    # Clean up function calls like ask_human("message") to just show message
    content = content.strip()
    if content.startswith('ask_human(') and content.endswith(')'):
        # Extract the message from ask_human("message")
        message = content[10:-2]  # Remove ask_human(" and ")
        # Remove any escaped quotes
        message = message.replace('\\"', '"')
        content = message
    return {
        "assistant_reply": content
    }

async def print_agent_output(chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Print agent output with streaming JSON preview in Streamlit.
    Returns the formatted response for UI display.
    
    Args:
        chunk: A dictionary containing agent or tool messages
    Returns:
        Optional[Dict[str, Any]]: The formatted response or None if no content
    """
    response = None
    
    # Ensure logs directory exists
    if os.path.exists(os.path.join(os.getcwd(), 'webui')):
        ensure_directory_exists(os.path.join(os.getcwd(), 'webui', 'logs'))
    
    if 'content' in chunk:
        content = chunk['content'].strip()
        if content:
            try:
                # Try to parse as JSON first
                parsed = json.loads(content)
                # Show in Streamlit
                st.write(parsed)
                # Show in console
                console.print(Syntax(json.dumps(parsed, indent=2), "json", theme="monokai"))
                response = parsed
            except json.JSONDecodeError:
                # If not JSON, wrap in our structure
                response = format_json_response(content)
                st.write(response)
                console.print(Syntax(json.dumps(response, indent=2), "json", theme="monokai"))
            
    elif 'agent' in chunk and 'messages' in chunk['agent']:
        messages = chunk['agent']['messages']
        for msg in messages:
            if isinstance(msg, AIMessage):
                if isinstance(msg.content, list):
                    for content in msg.content:
                        if content['type'] == 'text' and content['text'].strip():
                            response = format_json_response(content['text'])
                            st.write(response)
                            console.print(Syntax(json.dumps(response, indent=2), "json", theme="monokai"))
                else:
                    content = msg.content.strip()
                    if content:
                        try:
                            parsed = json.loads(content)
                            st.write(parsed)
                            console.print(Syntax(json.dumps(parsed, indent=2), "json", theme="monokai"))
                            response = parsed
                        except json.JSONDecodeError:
                            response = format_json_response(content)
                            st.write(response)
                            console.print(Syntax(json.dumps(response, indent=2), "json", theme="monokai"))
                        
    elif 'tools' in chunk and 'messages' in chunk['tools']:
        for msg in chunk['tools']['messages']:
            if isinstance(msg, dict):
                if msg.get('status') == 'error' and msg.get('content'):
                    error_msg = msg['content']
                    if 'Path does not exist' in error_msg:
                        # Try to create the directory if it's a path error
                        path = error_msg.split(': ')[-1].strip()
                        try:
                            ensure_directory_exists(path)
                            st.info(f"Created directory: {path}")
                        except Exception as e:
                            print_error(f"Failed to create directory {path}: {str(e)}")
                            st.error(f"Failed to create directory {path}: {str(e)}")
                    else:
                        print_error(error_msg)
                        st.error(error_msg)
                elif msg.get('content'):
                    try:
                        content = json.loads(msg['content'])
                        st.write(content)
                        console.print(Syntax(json.dumps(content, indent=2), "json", theme="monokai"))
                        response = content
                    except json.JSONDecodeError:
                        response = format_json_response(msg['content'])
                        st.write(response)
                        console.print(Syntax(json.dumps(response, indent=2), "json", theme="monokai"))
            elif hasattr(msg, 'status'):
                if msg.status == 'error' and msg.content:
                    error_msg = msg.content
                    if 'Path does not exist' in error_msg:
                        # Try to create the directory if it's a path error
                        path = error_msg.split(': ')[-1].strip()
                        try:
                            ensure_directory_exists(path)
                            st.info(f"Created directory: {path}")
                        except Exception as e:
                            print_error(f"Failed to create directory {path}: {str(e)}")
                            st.error(f"Failed to create directory {path}: {str(e)}")
                    else:
                        print_error(error_msg)
                        st.error(error_msg)
                elif msg.content:
                    try:
                        content = json.loads(msg.content)
                        st.write(content)
                        console.print(Syntax(json.dumps(content, indent=2), "json", theme="monokai"))
                        response = content
                    except json.JSONDecodeError:
                        response = format_json_response(msg.content)
                        st.write(response)
                        console.print(Syntax(json.dumps(response, indent=2), "json", theme="monokai"))
    
    # Ensure we're properly awaiting any async operations
    await asyncio.sleep(0)
    return response