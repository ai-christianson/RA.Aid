import os
from typing import Dict
from tavily import TavilyClient
from langchain_core.tools import tool
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

@tool
def web_search_tavily(query: str) -> Dict:
    """
    Perform a web search using Tavily API.
    
    Args:
        query: The search query string
        
    Returns:
        Dict containing search results from Tavily
    """
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    console.print(Panel(Markdown(query), title="🔍 Searching Tavily", border_style="bright_blue"))
    search_result = client.search(query=query)
    return search_result
