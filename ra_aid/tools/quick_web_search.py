import os
from typing import Dict, Any

from langchain_core.tools import tool
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from tavily import TavilyClient

# TODO: Re-evaluate if trajectory/human_input repo interaction is needed here
# from ra_aid.database.repositories.trajectory_repository import get_trajectory_repository
# from ra_aid.database.repositories.human_input_repository import get_human_input_repository

console = Console()


@tool
def quick_web_search(query: str) -> str:
    """
    Perform a *quick* web search using Tavily API for *simple* queries needing *concise* answers.

    Use this for fast lookups of specific facts, definitions, or simple questions where a brief 
    text answer is expected. Avoid using for complex research.

    Args:
        query (str): The simple search query string.

    Returns:
        str: A string containing the concise answer or a brief summary of top results from Tavily.
             Returns an error message string on failure.
    """
    # TODO: Re-evaluate trajectory recording for this simpler tool
    # trajectory_repo = get_trajectory_repository()
    # human_input_id = get_human_input_repository().get_most_recent_id()
    # trajectory_repo.create(
    #     tool_name="quick_web_search", # Updated tool name
    #     tool_parameters={"query": query},
    #     step_data={
    #         "query": query,
    #         "display_title": "Quick Web Search", # Updated title
    #     },
    #     record_type="tool_execution",
    #     human_input_id=human_input_id
    # )

    # Display search query panel
    console.print(
        Panel(Markdown(query), title="⚡ Quick Searching Tavily", border_style="yellow") # Adjusted title and style
    )

    try:
        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        # Request fewer, more concise results
        search_result: Dict[str, Any] = client.search(
            query=query,
            search_depth="basic", # Use basic search depth
            max_results=3 # Limit results
        )

        # TODO: Potentially refine the answer extraction/summarization logic
        # For now, return the 'answer' if available, otherwise a summary.
        answer = search_result.get("answer")
        if answer:
            # Record success trajectory?
            return answer
        else:
            # Fallback to summarizing results if no direct answer
            results_summary = "\n".join([f"- {res['title']}: {res['url']}" for res in search_result.get("results", [])[:3]])
            # Record success trajectory?
            return f"Could not find a direct answer. Top results:\n{results_summary}"

    except Exception as e:
        # TODO: Re-evaluate error trajectory recording
        # trajectory_repo.create(
        #     tool_name="quick_web_search", # Updated tool name
        #     tool_parameters={"query": query},
        #     step_data={
        #         "query": query,
        #         "display_title": "Quick Web Search Error", # Updated title
        #         "error": str(e)
        #     },
        #     record_type="tool_execution",
        #     human_input_id=human_input_id,
        #     is_error=True,
        #     error_message=str(e),
        #     error_type=type(e).__name__
        # )
        console.print(Panel(f"Error during quick search: {e}", title="Error", border_style="red"))
        # Return error message instead of raising exception for a "quick" search
        return f"Error performing quick web search: {e}" 