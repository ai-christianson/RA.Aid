import os
from typing import Dict
from tavily import TavilyClient
from langchain_core.tools import tool

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
    search_result = client.search(query=query)
    return search_result
