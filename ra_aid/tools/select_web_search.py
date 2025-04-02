import json
from typing import Dict, Any, Union, Literal

from langchain_core.tools import tool
from rich.console import Console
from rich.panel import Panel

# Import the actual search tools
from .quick_web_search import quick_web_search
from .web_search_jina import web_search_jina

console = Console()

@tool
def select_web_search(query: str, search_type: Literal['quick', 'deep'] = 'deep') -> Union[str, Dict[str, Any]]:
    """Routes web search requests to the appropriate tool (quick or deep) based on the specified search_type.
    
    Call this single tool for any web search need, specifying the type of search required.
    The agent calling this tool is responsible for determining the appropriate search_type 
    based on the user's original request (e.g., keywords like 'quick', 'simple' vs 'deep', 'comprehensive').
    
    Args:
        query (str): The user's core search query.
        search_type (Literal['quick', 'deep'], optional): The type of search needed. Defaults to 'deep'. 
                                                        Set to 'quick' for simple, concise answers.
        
    Returns:
        Union[str, Dict[str, Any]]: The result from the selected search tool (string for quick search, dict for deep search).
    """
    console.print(Panel(f"Routing web search for: '{query}'\nRequested Type: {search_type}", title="🔍 Web Search Router", border_style="purple"))
    
    # Route based on the explicit search_type argument
    if search_type == 'quick':
        console.print(Panel("Routing to: quick_web_search (Tavily) based on search_type='quick'", style="yellow"))
        try:
            result = quick_web_search.invoke({"query": query})
            # Ensure result is string
            return str(result) 
        except Exception as e:
            error_msg = f"Error calling quick_web_search internally: {e}"
            console.print(Panel(error_msg, title="Router Error", border_style="red"))
            # Format error as a JSON string to somewhat match expected string return type
            return json.dumps({"error": error_msg}) 
    else: # Default to 'deep'
        console.print(Panel("Routing to: web_search_jina (Jina) based on search_type='deep'", style="bright_blue"))
        try:
            # web_search_jina returns a Dict.
            result = web_search_jina.invoke({"query": query}) 
            # Ensure result is Dict
            if isinstance(result, dict):
                return result
            else:
                return {"content": str(result), "error": "Unexpected return type from web_search_jina"}
        except Exception as e:
            error_msg = f"Error calling web_search_jina internally: {e}"
            console.print(Panel(error_msg, title="Router Error", border_style="red"))
            return {"error": error_msg} 