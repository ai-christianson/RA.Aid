"""
Key snippets gc agent implementation.

This agent is responsible for maintaining the code snippet knowledge base by pruning less important
snippets when the total number exceeds a specified threshold. The agent evaluates all
key snippets and deletes the least valuable ones to keep the database clean and relevant.
"""

from typing import List

from langchain_core.tools import tool
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# Import agent_utils functions at runtime to avoid circular imports
from ra_aid import agent_utils
from ra_aid.console.formatting import console_panel
from ra_aid.database.repositories.key_snippet_repository import get_key_snippet_repository
from ra_aid.database.repositories.human_input_repository import get_human_input_repository
from ra_aid.database.repositories.config_repository import get_config_repository
from ra_aid.database.repositories.trajectory_repository import get_trajectory_repository
from ra_aid.llm import initialize_llm
from ra_aid.prompts.key_snippets_gc_prompts import KEY_SNIPPETS_GC_PROMPT
from ra_aid.tools.memory import log_work_event
from ra_aid.agent_context import mark_should_exit


console = Console()


@tool
def delete_key_snippets(snippet_ids: List[int]) -> str:
    """Delete multiple key snippets from the database by their IDs.
    Silently skips any IDs that don't exist.

    Args:
        snippet_ids: List of snippet IDs to delete
        
    Returns:
        str: Success or failure message
    """
    results = []
    not_found_snippets = []
    failed_snippets = []
    protected_snippets = []
    
    # Try to get the current human input to protect its snippets
    current_human_input_id = None
    try:
        current_human_input_id = get_human_input_repository().get_most_recent_id()
    except Exception as e:
        console.print(f"Warning: Could not retrieve current human input: {str(e)}")
    
    for snippet_id in snippet_ids:
        # Get the snippet first to capture filepath for the message
        snippet = get_key_snippet_repository().get(snippet_id)
        if snippet:
            filepath = snippet.filepath
            
            # Check if this snippet is associated with the current human input
            if current_human_input_id is not None and snippet.human_input_id == current_human_input_id:
                protected_snippets.append((snippet_id, filepath))
                continue
                
            # Delete from database if not protected
            success = get_key_snippet_repository().delete(snippet_id)
            if success:
                success_msg = f"Successfully deleted snippet #{snippet_id} from {filepath}"
                # Record GC operation in trajectory
                try:
                    trajectory_repo = get_trajectory_repository()
                    human_input_id = get_human_input_repository().get_most_recent_id()
                    trajectory_repo.create(
                        step_data={
                            "deleted_snippet_id": snippet_id,
                            "filepath": filepath,
                            "display_title": "Snippet Deleted",
                        },
                        record_type="gc_operation",
                        human_input_id=human_input_id,
                        tool_name="key_snippets_gc_agent"
                    )
                except Exception:
                    pass  # Continue if trajectory recording fails
                
                console.print(
                    Panel(
                        Markdown(success_msg), title="Snippet Deleted", border_style="green"
                    )
                )
                results.append((snippet_id, filepath))
                log_work_event(f"Deleted snippet {snippet_id}.")
            else:
                failed_snippets.append(snippet_id)
        else:
            not_found_snippets.append(snippet_id)

    # Prepare result message
    result_parts = []
    if results:
        deleted_msg = "Successfully deleted snippets:\n" + "\n".join([f"- #{snippet_id}: {filepath}" for snippet_id, filepath in results])
        result_parts.append(deleted_msg)
    
    if protected_snippets:
        protected_msg = "Protected snippets (associated with current request):\n" + "\n".join([f"- #{snippet_id}: {filepath}" for snippet_id, filepath in protected_snippets])
        result_parts.append(protected_msg)
        # Record GC operation in trajectory
        try:
            trajectory_repo = get_trajectory_repository()
            human_input_id = get_human_input_repository().get_most_recent_id()
            trajectory_repo.create(
                step_data={
                    "protected_snippets": protected_snippets,
                    "display_title": "Snippets Protected",
                },
                record_type="gc_operation",
                human_input_id=human_input_id,
                tool_name="key_snippets_gc_agent"
            )
        except Exception:
            pass  # Continue if trajectory recording fails
            
        console.print(
            Panel(Markdown(protected_msg), title="Snippets Protected", border_style="blue")
        )
    
    if not_found_snippets:
        not_found_msg = f"Snippets not found: {', '.join([f'#{snippet_id}' for snippet_id in not_found_snippets])}"
        result_parts.append(not_found_msg)
    
    if failed_snippets:
        failed_msg = f"Failed to delete snippets: {', '.join([f'#{snippet_id}' for snippet_id in failed_snippets])}"
        result_parts.append(failed_msg)
    
    # Mark that the agent should exit
    mark_should_exit()
    
    return "Snippets deleted."


def run_key_snippets_gc_agent() -> None:
    """Run the key snippets gc agent to maintain a reasonable number of key snippets.
    
    The agent analyzes all key snippets and determines which are the least valuable,
    deleting them to maintain a manageable collection size of high-value snippets.
    Snippets associated with the current human input are excluded from deletion.
    """
    # Get the count of key snippets
    snippets = get_key_snippet_repository().get_all()
    snippet_count = len(snippets)
    
    # Display status panel with snippet count included
    try:
        trajectory_repo = get_trajectory_repository()
        human_input_id = get_human_input_repository().get_most_recent_id()
        trajectory_repo.create(
            step_data={
                "snippet_count": snippet_count,
                "display_title": "Garbage Collection",
            },
            record_type="gc_operation",
            human_input_id=human_input_id,
            tool_name="key_snippets_gc_agent"
        )
    except Exception:
        pass  # Continue if trajectory recording fails
        
    console_panel(f"Gathering my thoughts...\nCurrent number of key snippets: {snippet_count}", title="🗑 Garbage Collection")
    
    # Only run the agent if we actually have snippets to clean
    if snippet_count > 0:
        # Try to get the current human input ID to exclude its snippets
        current_human_input_id = None
        try:
            current_human_input_id = get_human_input_repository().get_most_recent_id()
        except Exception as e:
            console.print(f"Warning: Could not retrieve current human input: {str(e)}")
        
        # Get all snippets that are not associated with the current human input
        eligible_snippets = []
        protected_snippets = []
        for snippet in snippets:
            if current_human_input_id is not None and snippet.human_input_id == current_human_input_id:
                protected_snippets.append(snippet)
            else:
                eligible_snippets.append(snippet)
        
        # Only process if we have snippets that can be deleted
        if eligible_snippets:
            # Get eligible snippets as a formatted string for the prompt
            snippets_dict = {
                snippet.id: {
                    'filepath': snippet.filepath,
                    'line_number': snippet.line_number,
                    'snippet': snippet.snippet,
                    'description': snippet.description
                } 
                for snippet in eligible_snippets
            }
            
            formatted_snippets = "\n".join([
                f"Snippet #{k}: filepath={v['filepath']}, line_number={v['line_number']}, description={v['description']}\n```python\n{v['snippet']}\n```" 
                for k, v in snippets_dict.items()
            ])
            
            # Initialize the LLM model
            model = initialize_llm(
                get_config_repository().get("provider", "anthropic"),
                get_config_repository().get("model", "claude-3-7-sonnet-20250219"),
                temperature=get_config_repository().get("temperature")
            )
            
            # Create the agent with the delete_key_snippets tool
            agent = agent_utils.create_agent(model, [delete_key_snippets])
            
            # Format the prompt with the eligible snippets
            prompt = KEY_SNIPPETS_GC_PROMPT.format(key_snippets=formatted_snippets)
            
            # Set up the agent configuration
            agent_config = {
                "recursion_limit": 50  # Set a reasonable recursion limit
            }
            
            # Run the agent
            agent_utils.run_agent_with_retry(agent, prompt, agent_config)
            
            # Get updated count
            updated_snippets = get_key_snippet_repository().get_all()
            updated_count = len(updated_snippets)
            
            # Show info panel with updated count and protected snippets count
            protected_count = len(protected_snippets)
            if protected_count > 0:
                # Record GC completion in trajectory
                try:
                    trajectory_repo = get_trajectory_repository()
                    human_input_id = get_human_input_repository().get_most_recent_id()
                    trajectory_repo.create(
                        step_data={
                            "original_count": snippet_count,
                            "updated_count": updated_count,
                            "protected_count": protected_count,
                            "display_title": "GC Complete",
                        },
                        record_type="gc_operation",
                        human_input_id=human_input_id,
                        tool_name="key_snippets_gc_agent"
                    )
                except Exception:
                    pass  # Continue if trajectory recording fails
                
                console_panel(
                    f"Cleaned key snippets: {snippet_count} → {updated_count}\nProtected snippets (associated with current request): {protected_count}",
                    title="🗑 GC Complete"
                )
            else:
                # Record GC completion in trajectory
                try:
                    trajectory_repo = get_trajectory_repository()
                    human_input_id = get_human_input_repository().get_most_recent_id()
                    trajectory_repo.create(
                        step_data={
                            "original_count": snippet_count,
                            "updated_count": updated_count,
                            "protected_count": 0,
                            "display_title": "GC Complete",
                        },
                        record_type="gc_operation",
                        human_input_id=human_input_id,
                        tool_name="key_snippets_gc_agent"
                    )
                except Exception:
                    pass  # Continue if trajectory recording fails
                
                console_panel(
                    f"Cleaned key snippets: {snippet_count} → {updated_count}",
                    title="🗑 GC Complete"
                )
        else:
            # Record GC info in trajectory
            try:
                trajectory_repo = get_trajectory_repository()
                human_input_id = get_human_input_repository().get_most_recent_id()
                trajectory_repo.create(
                    step_data={
                        "protected_count": len(protected_snippets),
                        "message": "All snippets are protected",
                        "display_title": "GC Info",
                    },
                    record_type="gc_operation",
                    human_input_id=human_input_id,
                    tool_name="key_snippets_gc_agent"
                )
            except Exception:
                pass  # Continue if trajectory recording fails
                
            console_panel(f"All {len(protected_snippets)} snippets are associated with the current request and protected from deletion.", title="🗑 GC Info")
    else:
        # Record GC info in trajectory
        try:
            trajectory_repo = get_trajectory_repository()
            human_input_id = get_human_input_repository().get_most_recent_id()
            trajectory_repo.create(
                step_data={
                    "snippet_count": 0,
                    "message": "No key snippets to clean",
                    "display_title": "GC Info",
                },
                record_type="gc_operation",
                human_input_id=human_input_id,
                tool_name="key_snippets_gc_agent"
            )
        except Exception:
            pass  # Continue if trajectory recording fails
            
        console_panel("No key snippets to clean.", title="🗑 GC Info")
