"""
Research notes gc agent implementation.

This agent is responsible for maintaining the collection of research notes by pruning less
important notes when the total number exceeds a specified threshold. The agent evaluates all
research notes and deletes the least valuable ones to keep the database clean and relevant.
"""

import logging
from typing import List

from langchain_core.tools import tool
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ra_aid.console.formatting import console_panel

from ra_aid.agent_context import mark_should_exit

from ra_aid.agent_utils import create_agent, run_agent_with_retry
from ra_aid.database.repositories.research_note_repository import get_research_note_repository
from ra_aid.database.repositories.human_input_repository import get_human_input_repository
from ra_aid.database.repositories.config_repository import get_config_repository
from ra_aid.database.repositories.trajectory_repository import get_trajectory_repository
from ra_aid.llm import initialize_llm
from ra_aid.tools.memory import log_work_event

logger = logging.getLogger(__name__)


console = Console()


@tool
def delete_research_notes(note_ids: List[int]) -> str:
    """Delete multiple research notes by their IDs.

    Args:
        note_ids: List of IDs of the research notes to delete
        
    Returns:
        str: Success or failure message
    """
    deleted_notes = []
    not_found_notes = []
    failed_notes = []
    protected_notes = []
    
    # Try to get the current human input to protect its notes
    current_human_input_id = None
    try:
        current_human_input_id = get_human_input_repository().get_most_recent_id()
    except Exception as e:
        console.print(f"Warning: Could not retrieve current human input: {str(e)}")
    
    for note_id in note_ids:
        try:
            # Get the note first to display information
            note = get_research_note_repository().get(note_id)
            if note:
                # Check if this note is associated with the current human input
                if current_human_input_id is not None and note.human_input_id == current_human_input_id:
                    protected_notes.append((note_id, note.content))
                    continue
                
                # Delete the note if it's not protected
                was_deleted = get_research_note_repository().delete(note_id)
                if was_deleted:
                    deleted_notes.append((note_id, note.content))
                    log_work_event(f"Deleted research note {note_id}.")
                else:
                    failed_notes.append(note_id)
            else:
                not_found_notes.append(note_id)
        except RuntimeError as e:
            logger.error(f"Failed to access research note repository: {str(e)}")
            failed_notes.append(note_id)
        except Exception as e:
            # For any other exceptions, log and continue
            logger.error(f"Error processing research note {note_id}: {str(e)}")
            failed_notes.append(note_id)
            
    # Prepare result message
    result_parts = []
    if deleted_notes:
        deleted_msg = "Successfully deleted research notes:\n" + "\n".join([f"- #{note_id}: {content[:100]}..." if len(content) > 100 else f"- #{note_id}: {content}" for note_id, content in deleted_notes])
        result_parts.append(deleted_msg)
        # Record GC operation in trajectory
        try:
            trajectory_repo = get_trajectory_repository()
            human_input_id = get_human_input_repository().get_most_recent_id()
            trajectory_repo.create(
                step_data={
                    "deleted_notes": deleted_notes,
                    "display_title": "Research Notes Deleted",
                },
                record_type="gc_operation",
                human_input_id=human_input_id,
                tool_name="research_notes_gc_agent"
            )
        except Exception:
            pass  # Continue if trajectory recording fails
            
        console.print(
            Panel(Markdown(deleted_msg), title="Research Notes Deleted", border_style="green")
        )
    
    if protected_notes:
        protected_msg = "Protected research notes (associated with current request):\n" + "\n".join([f"- #{note_id}: {content[:100]}..." if len(content) > 100 else f"- #{note_id}: {content}" for note_id, content in protected_notes])
        result_parts.append(protected_msg)
        # Record GC operation in trajectory
        try:
            trajectory_repo = get_trajectory_repository()
            human_input_id = get_human_input_repository().get_most_recent_id()
            trajectory_repo.create(
                step_data={
                    "protected_notes": protected_notes,
                    "display_title": "Research Notes Protected",
                },
                record_type="gc_operation",
                human_input_id=human_input_id,
                tool_name="research_notes_gc_agent"
            )
        except Exception:
            pass  # Continue if trajectory recording fails
            
        console.print(
            Panel(Markdown(protected_msg), title="Research Notes Protected", border_style="blue")
        )
    
    if not_found_notes:
        not_found_msg = f"Research notes not found: {', '.join([f'#{note_id}' for note_id in not_found_notes])}"
        result_parts.append(not_found_msg)
    
    if failed_notes:
        failed_msg = f"Failed to delete research notes: {', '.join([f'#{note_id}' for note_id in failed_notes])}"
        result_parts.append(failed_msg)
    
    # Mark the agent to exit after performing the cleanup operation
    mark_should_exit()
    
    return "\n".join(result_parts)


def run_research_notes_gc_agent(threshold: int = 30) -> None:
    """Run the research notes gc agent to maintain a reasonable number of research notes.
    
    The agent analyzes all research notes and determines which are the least valuable,
    deleting them to maintain a manageable collection size of high-value notes.
    Notes associated with the current human input are excluded from deletion.
    
    Args:
        threshold: Maximum number of research notes to keep before triggering cleanup
    """
    # Get the count of research notes
    try:
        notes = get_research_note_repository().get_all()
        note_count = len(notes)
    except RuntimeError as e:
        logger.error(f"Failed to access research note repository: {str(e)}")
        # Record GC error in trajectory
        try:
            trajectory_repo = get_trajectory_repository()
            human_input_id = get_human_input_repository().get_most_recent_id()
            trajectory_repo.create(
                step_data={
                    "error": str(e),
                    "display_title": "GC Error",
                },
                record_type="gc_operation",
                human_input_id=human_input_id,
                tool_name="research_notes_gc_agent",
                is_error=True,
                error_message=str(e),
                error_type="Repository Error"
            )
        except Exception:
            pass  # Continue if trajectory recording fails
            
        console_panel(f"Error: {str(e)}", title="🗑 GC Error", border_style="red")
        return  # Exit the function if we can't access the repository
    
    # Display status panel with note count included
    try:
        trajectory_repo = get_trajectory_repository()
        human_input_id = get_human_input_repository().get_most_recent_id()
        trajectory_repo.create(
            step_data={
                "note_count": note_count,
                "display_title": "Garbage Collection",
            },
            record_type="gc_operation",
            human_input_id=human_input_id,
            tool_name="research_notes_gc_agent"
        )
    except Exception:
        pass  # Continue if trajectory recording fails
        
    console_panel(f"Gathering my thoughts...\nCurrent number of research notes: {note_count}", title="🗑 Garbage Collection")
    
    # Only run the agent if we actually have notes to clean and we're over the threshold
    if note_count > threshold:
        # Try to get the current human input ID to exclude its notes
        current_human_input_id = None
        try:
            current_human_input_id = get_human_input_repository().get_most_recent_id()
        except Exception as e:
            console.print(f"Warning: Could not retrieve current human input: {str(e)}")
        
        # Get all notes that are not associated with the current human input
        eligible_notes = []
        protected_notes = []
        for note in notes:
            if current_human_input_id is not None and note.human_input_id == current_human_input_id:
                protected_notes.append(note)
            else:
                eligible_notes.append(note)
        
        # Only process if we have notes that can be deleted
        if eligible_notes:
            # Format notes as a dictionary for the prompt
            notes_dict = {note.id: note.content for note in eligible_notes}
            formatted_notes = "\n".join([f"Note #{k}: {v}" for k, v in notes_dict.items()])
            
            # Initialize the LLM model
            model = initialize_llm(
                get_config_repository().get("provider", "anthropic"),
                get_config_repository().get("model", "claude-3-7-sonnet-20250219"),
                temperature=get_config_repository().get("temperature")
            )
            
            # Create the agent with the delete_research_notes tool
            agent = create_agent(model, [delete_research_notes])
            
            # Build the prompt for the research notes gc agent
            prompt = f"""
You are a Research Notes Cleaner agent responsible for maintaining the research notes collection by pruning less important notes.

<research notes>
{formatted_notes}
</research notes>

Task:
Your task is to analyze all the research notes in the system and determine which ones should be kept and which ones should be removed.

Guidelines for evaluation:
1. Review all research notes and their IDs
2. Identify which notes are lowest value/most ephemeral based on:
   - Relevance to the overall project
   - Specificity and actionability of the information
   - Long-term value vs. temporary relevance
   - Uniqueness of the information (avoid redundancy)
   - How fundamental the note is to understanding the context

3. Trim down the collection to keep no more than {threshold} highest value, longest-lasting notes
4. For each note you decide to delete, provide a brief explanation of your reasoning

Retention priority (from highest to lowest):
- Core research findings directly relevant to the project requirements
- Important technical details that affect implementation decisions
- API documentation and usage examples
- Configuration information and best practices
- Alternative approaches considered with pros and cons
- General background information
- Information that is easily found elsewhere or outdated

For notes of similar importance, prefer to keep more recent notes if they supersede older information.

Output:
1. List the IDs of notes to be deleted using the delete_research_notes tool with the IDs provided as a list [ids...], NOT as a comma-separated string
2. Provide a brief explanation for each deletion decision
3. Explain your overall approach to selecting which notes to keep

IMPORTANT: 
- Use the delete_research_notes tool with multiple IDs at once in a single call, rather than making multiple individual deletion calls
- The delete_research_notes tool accepts a list of IDs in the format [id1, id2, id3, ...], not as a comma-separated string
- Batch deletion is much more efficient than calling the deletion function multiple times
- Collect all IDs to delete first, then make a single call to delete_research_notes with the complete list

Remember: Your goal is to maintain a concise, high-value collection of research notes that preserves essential information while removing ephemeral or less critical details.
"""
            
            # Set up the agent configuration
            agent_config = {
                "recursion_limit": 50  # Set a reasonable recursion limit
            }
            
            # Run the agent
            run_agent_with_retry(agent, prompt, agent_config)
            
            # Get updated count
            try:
                updated_notes = get_research_note_repository().get_all()
                updated_count = len(updated_notes)
            except RuntimeError as e:
                logger.error(f"Failed to access research note repository for update count: {str(e)}")
                updated_count = "unknown"
            
            # Show info panel with updated count and protected notes count
            protected_count = len(protected_notes)
            if protected_count > 0:
                # Record GC completion in trajectory
                try:
                    trajectory_repo = get_trajectory_repository()
                    human_input_id = get_human_input_repository().get_most_recent_id()
                    trajectory_repo.create(
                        step_data={
                            "original_count": note_count,
                            "updated_count": updated_count,
                            "protected_count": protected_count,
                            "display_title": "GC Complete",
                        },
                        record_type="gc_operation",
                        human_input_id=human_input_id,
                        tool_name="research_notes_gc_agent"
                    )
                except Exception:
                    pass  # Continue if trajectory recording fails
                
                console_panel(
                    f"Cleaned research notes: {note_count} → {updated_count}\nProtected notes (associated with current request): {protected_count}",
                    title="🗑 GC Complete"
                )
            else:
                # Record GC completion in trajectory
                try:
                    trajectory_repo = get_trajectory_repository()
                    human_input_id = get_human_input_repository().get_most_recent_id()
                    trajectory_repo.create(
                        step_data={
                            "original_count": note_count,
                            "updated_count": updated_count,
                            "protected_count": 0,
                            "display_title": "GC Complete",
                        },
                        record_type="gc_operation",
                        human_input_id=human_input_id,
                        tool_name="research_notes_gc_agent"
                    )
                except Exception:
                    pass  # Continue if trajectory recording fails
                
                console_panel(
                    f"Cleaned research notes: {note_count} → {updated_count}",
                    title="🗑 GC Complete"
                )
        else:
            # Record GC info in trajectory
            try:
                trajectory_repo = get_trajectory_repository()
                human_input_id = get_human_input_repository().get_most_recent_id()
                trajectory_repo.create(
                    step_data={
                        "protected_count": len(protected_notes),
                        "message": "All research notes are protected",
                        "display_title": "GC Info",
                    },
                    record_type="gc_operation",
                    human_input_id=human_input_id,
                    tool_name="research_notes_gc_agent"
                )
            except Exception:
                pass  # Continue if trajectory recording fails
                
            console_panel(f"All {len(protected_notes)} research notes are associated with the current request and protected from deletion.", title="🗑 GC Info")
    else:
        # Record GC info in trajectory
        try:
            trajectory_repo = get_trajectory_repository()
            human_input_id = get_human_input_repository().get_most_recent_id()
            trajectory_repo.create(
                step_data={
                    "note_count": note_count,
                    "threshold": threshold,
                    "message": "Below threshold - no cleanup needed",
                    "display_title": "GC Info",
                },
                record_type="gc_operation",
                human_input_id=human_input_id,
                tool_name="research_notes_gc_agent"
            )
        except Exception:
            pass  # Continue if trajectory recording fails
            
        console_panel(f"Research notes count ({note_count}) is below threshold ({threshold}). No cleanup needed.", title="🗑 GC Info")
