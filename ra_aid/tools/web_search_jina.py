import os
from typing import Dict, Optional, List, Union
import json
import requests
from datetime import datetime

from langchain_core.tools import tool
from rich.console import Console
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
        
    def search(
        self,
        query: str,
        reasoning_effort: str = "medium",
        stream: bool = True,
        no_direct_answer: bool = False,
        max_returned_urls: int = 10,
        good_domains: Optional[List[str]] = None,
        bad_domains: Optional[List[str]] = None,
        only_domains: Optional[List[str]] = None,
    ) -> Dict:
        """
        Perform a search query using Jina DeepSearch.
        
        Args:
            query: The search query
            reasoning_effort: Level of reasoning effort ("low", "medium", "high")
            stream: Whether to stream responses
            no_direct_answer: Force deeper search even for simple queries
            max_returned_urls: Maximum number of URLs to return
            good_domains: Domains to prioritize
            bad_domains: Domains to exclude
            only_domains: Domains to exclusively include
            
        Returns:
            Dict containing search results and metadata
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": "jina-deepsearch-v1",
            "messages": [{"role": "user", "content": query}],
            "stream": stream,
            "reasoning_effort": reasoning_effort,
            "no_direct_answer": no_direct_answer,
            "max_returned_urls": max_returned_urls
        }
        
        if good_domains:
            data["good_domains"] = good_domains
        if bad_domains:
            data["bad_domains"] = bad_domains
        if only_domains:
            data["only_domains"] = only_domains
            
        response = requests.post(self.base_url, headers=headers, json=data)
        response.raise_for_status()
        
        # Process streaming response
        if stream:
            final_response = None
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8').replace('data: ', ''))
                        if chunk.get("choices") and chunk["choices"][0].get("finish_reason") == "stop":
                            final_response = chunk
                    except json.JSONDecodeError:
                        continue
            return final_response
        
        return response.json()


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
    Perform a web search using Jina DeepSearch API.

    Args:
        query: The search query string
        reasoning_effort: Level of reasoning effort ("low", "medium", "high")
        no_direct_answer: Force deeper search even for simple queries
        good_domains: Domains to prioritize
        bad_domains: Domains to exclude
        only_domains: Domains to exclusively include

    Returns:
        Dict containing search results from Jina DeepSearch
    """
    # Record trajectory before displaying panel
    trajectory_repo = get_trajectory_repository()
    human_input_id = get_human_input_repository().get_most_recent_id()
    trajectory_repo.create(
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
    
    # Display search query panel
    console.print(
        Panel(Markdown(query), title="🔍 Searching with Jina DeepSearch", border_style="bright_blue")
    )
    
    try:
        client = JinaDeepSearchClient()
        search_result = client.search(
            query=query,
            reasoning_effort=reasoning_effort,
            no_direct_answer=no_direct_answer,
            good_domains=good_domains,
            bad_domains=bad_domains,
            only_domains=only_domains
        )
        
        # Extract relevant information from the response
        if search_result and search_result.get("choices"):
            choice = search_result["choices"][0]
            result = {
                "content": choice.get("delta", {}).get("content", ""),
                "urls": search_result.get("visitedURLs", []),
                "usage": search_result.get("usage", {}),
                "timestamp": datetime.now().isoformat()
            }
            
            # Add any annotations if present
            if "annotations" in choice.get("delta", {}):
                result["annotations"] = choice["delta"]["annotations"]
                
            return result
        return search_result
        
    except Exception as e:
        # Record error in trajectory
        trajectory_repo.create(
            tool_name="web_search_jina",
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