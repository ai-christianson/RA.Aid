import os
from typing import Dict

from langchain_core.tools import tool
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from tavily import TavilyClient

from ra_aid.database.repositories.trajectory_repository import get_trajectory_repository
from ra_aid.database.repositories.human_input_repository import get_human_input_repository

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
    # Record trajectory before displaying panel
    trajectory_repo = get_trajectory_repository()
    human_input_id = get_human_input_repository().get_most_recent_id()
    trajectory_repo.create(
        tool_name="web_search_tavily",
        tool_parameters={"query": query},
        step_data={
            "query": query,
            "display_title": "Web Search",
        },
        record_type="tool_execution",
        human_input_id=human_input_id
    )
    
    # Display search query panel
    console.print(
        Panel(Markdown(query), title="🔍 Searching Tavily", border_style="bright_blue")
    )
    
    try:
        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        search_result = client.search(query=query)
        return search_result
    except Exception as e:
        # Record error in trajectory
        trajectory_repo.create(
            tool_name="web_search_tavily",
            tool_parameters={"query": query},
            step_data={
                "query": query,
                "display_title": "Web Search Error",
                "error": str(e)
            },
            record_type="tool_execution",
            human_input_id=human_input_id,
            is_error=True,
            error_message=str(e),
            error_type=type(e).__name__
        )
        # Re-raise the exception to maintain original behavior
        raise