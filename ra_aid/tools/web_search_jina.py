import os
from typing import Dict, Optional, List, Union
import json
import requests
from datetime import datetime
import sys

from langchain_core.tools import tool
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from ra_aid.database.repositories.trajectory_repository import get_trajectory_repository
from ra_aid.database.repositories.human_input_repository import get_human_input_repository

console = Console()

class JinaDeepSearchClient:
    """Client for interacting with Jina DeepSearch API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("JINA_API_KEY environment variable is not set")
        self.base_url = "https://deepsearch.jina.ai/v1/chat/completions"
        
    # REMOVED: Search method logic moved to the tool function for live display
    # def search(...) -> Dict:
    #    ...


@tool
def web_search_jina(
    query: str,
    reasoning_effort: str = "medium",
    no_direct_answer: bool = False,
    good_domains: Optional[List[str]] = None,
    bad_domains: Optional[List[str]] = None,
    only_domains: Optional[List[str]] = None,
) -> Dict:
    """
    Perform a deep, comprehensive web search using Jina AI DeepSearch.

    Use this tool for complex questions requiring detailed research, synthesis,
    and potentially exploring multiple sources or perspectives. It provides streamed results.

    Args:
        query (str): The primary search query.
        reasoning_effort (str, optional): The level of reasoning effort ('low', 'medium', 'high'). Defaults to "medium".
        no_direct_answer (bool, optional): If True, forces the API to avoid giving a direct answer. Defaults to False.
        good_domains (Optional[List[str]], optional): List of preferred domains.
        bad_domains (Optional[List[str]], optional): List of domains to exclude.
        only_domains (Optional[List[str]], optional): List of domains to exclusively search within.

    Returns:
        Dict: A dictionary containing the search results summary, URLs, usage stats, annotations, and timestamp.
               On error, returns a dictionary with an 'error' key.
    """
    # Record trajectory before displaying panel
    trajectory_repo = get_trajectory_repository()
    human_input_id = get_human_input_repository().get_most_recent_id()
    trajectory_id = trajectory_repo.create(
        tool_name="web_search_jina",
        tool_parameters={
            "query": query,
            "reasoning_effort": reasoning_effort,
            "no_direct_answer": no_direct_answer,
            "good_domains": good_domains,
            "bad_domains": bad_domains,
            "only_domains": only_domains
        },
        step_data={
            "query": query,
            "display_title": "Web Search",
        },
        record_type="tool_execution",
        human_input_id=human_input_id
    )
    
    # --- Start Live Streaming Logic ---
    console = Console()
    api_key = os.environ.get("JINA_API_KEY")
    if not api_key:
         # Handle missing key error appropriately (maybe raise or return error dict)
        error_msg = "JINA_API_KEY environment variable is not set"
        console.print(Panel(error_msg, title="❌ Jina Search Error", border_style="red"))
        # Update trajectory with error
        trajectory_repo.update(trajectory_id, {
            "is_error": True,
            "error_message": error_msg,
            "error_type": "ValueError",
            "step_data": json.dumps({"query": query, "display_title": "Web Search Error", "error": error_msg})
        })
        return {"error": error_msg}

    base_url = "https://deepsearch.jina.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "jina-deepsearch-v1",
        "messages": [{"role": "user", "content": query}],
        "stream": True, # Ensure streaming is requested
        "reasoning_effort": reasoning_effort,
        "no_direct_answer": no_direct_answer,
        "max_returned_urls": 10 # Keep a reasonable default
    }
    # Add optional domain filters
    if good_domains: data["good_domains"] = good_domains
    if bad_domains: data["bad_domains"] = bad_domains
    if only_domains: data["only_domains"] = only_domains

    query_panel = Panel(Markdown(f"**Query:** {query}"), title="🔍 Searching with Jina DeepSearch", border_style="bright_blue")
    streamed_content_md = Markdown("", style="green") # Markdown object for streamed content
    display_group = Group(query_panel, streamed_content_md)
    
    final_response_chunk = None
    streamed_buffer = ""
    urls_buffer = []
    usage_buffer = {}
    annotations_buffer = []

    try:
        with Live(display_group, console=console, refresh_per_second=5, vertical_overflow="visible") as live:
            response = requests.post(base_url, headers=headers, json=data, stream=True)
            response.raise_for_status() # Check for HTTP errors immediately

            for line in response.iter_lines():
                if line:
                    try:
                        line_content = line.decode('utf-8').strip()
                        if line_content.startswith('data: '):
                            json_str = line_content[len('data: '):]
                            if json_str == "[DONE]": continue
                            
                            chunk = json.loads(json_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content_part = delta.get("content")
                            
                            if content_part:
                                streamed_buffer += content_part
                                # REVISED: Create new Markdown and Group for update
                                streamed_content_md = Markdown(streamed_buffer, style="green")
                                display_group = Group(query_panel, streamed_content_md)
                                live.update(display_group)
                                
                            # Check for final chunk to store metadata
                            if chunk.get("choices") and chunk["choices"][0].get("finish_reason") == "stop":
                                final_response_chunk = chunk 
                                # Extract final metadata if available in the last chunk
                                urls_buffer = chunk.get("visitedURLs", [])
                                usage_buffer = chunk.get("usage", {})
                                if "annotations" in delta:
                                     annotations_buffer = delta["annotations"]
                                # Exit loop once final chunk is processed
                                break 
                                
                    except json.JSONDecodeError as e:
                        print(f"[Jina Stream Error] Failed to decode JSON: {e} - Line: '{line_content}'", file=sys.stderr)
                        continue
                    except Exception as e:
                        print(f"[Jina Stream Error] Unexpected error processing line: {e} - Line: '{line_content}'", file=sys.stderr)
                        continue
            
            # Final update after loop
            streamed_content_md = Markdown(streamed_buffer, style="green")
            display_group = Group(query_panel, streamed_content_md)
            live.update(display_group)

        # --- End Live Streaming Logic ---
        
        # Construct final result dictionary
        if not streamed_buffer and final_response_chunk is None:
             # Handle cases where stream might end abruptly or return no content
             error_msg = "Jina search returned no content or stream ended unexpectedly."
             console.print(Panel(error_msg, title="⚠️ Jina Search Warning", border_style="yellow"))
             # Update trajectory
             trajectory_repo.update(trajectory_id, {
                 "is_error": True, # Mark as error or warning? Maybe just warning.
                 "error_message": error_msg,
                 "error_type": "EmptyResponseError",
                 "step_data": json.dumps({"query": query, "display_title": "Web Search Warning", "error": error_msg})
             })
             return {"error": error_msg, "content": "", "urls": [], "usage": {}, "timestamp": datetime.now().isoformat()}
             
        # REVISED: Return a summary message instead of the full buffer
        # result = {
        #     "content": streamed_buffer,
        #     "urls": urls_buffer,
        #     "usage": usage_buffer,
        #     "annotations": annotations_buffer,
        #     "timestamp": datetime.now().isoformat()
        # }
        summary_message = f"Jina search completed. {len(streamed_buffer)} characters received. URLs found: {len(urls_buffer)}."
        result = {
            "content": summary_message,
            "urls": urls_buffer,      # Keep URLs and usage for potential agent use
            "usage": usage_buffer,
            "annotations": annotations_buffer,
            "timestamp": datetime.now().isoformat()
        }
        return result
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Jina API request failed: {e}"
        console.print(Panel(error_msg, title="❌ Jina Search Error", border_style="red"))
        trajectory_repo.update(trajectory_id, {
            "is_error": True,
            "error_message": error_msg,
            "error_type": type(e).__name__,
            "step_data": json.dumps({"query": query, "display_title": "Web Search Error", "error": error_msg})
        })
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Error during Jina search: {str(e)}"
        console.print(Panel(error_msg, title="❌ Jina Search Error", border_style="red"))
        trajectory_repo.update(trajectory_id, {
            "is_error": True,
            "error_message": error_msg,
            "error_type": type(e).__name__,
            "step_data": json.dumps({"query": query, "display_title": "Web Search Error", "error": error_msg})
        })
        # Re-raise unexpected exceptions?
        # For now, return error dict to prevent agent crash
        return {"error": error_msg} 