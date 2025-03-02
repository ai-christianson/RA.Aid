"""
Key facts gc agent implementation.

This agent is responsible for maintaining the knowledge base by pruning less important
facts when the total number exceeds a specified threshold. The agent evaluates all
key facts and deletes the least valuable ones to keep the database clean and relevant.
"""

from typing import List

from langchain_core.tools import tool
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ra_aid.agent_utils import create_agent, run_agent_with_retry
from ra_aid.database.repositories.key_fact_repository import KeyFactRepository
from ra_aid.llm import initialize_llm
from ra_aid.prompts.key_facts_gc_prompts import KEY_FACTS_GC_PROMPT
from ra_aid.tools.memory import log_work_event, _global_memory


console = Console()
key_fact_repository = KeyFactRepository()


@tool
def delete_key_fact(fact_id: int) -> str:
    """Delete a key fact by its ID.

    Args:
        fact_id: The ID of the key fact to delete
        
    Returns:
        str: Success or failure message
    """
    # Get the fact first to display information
    fact = key_fact_repository.get(fact_id)
    if fact:
        # Delete the fact
        was_deleted = key_fact_repository.delete(fact_id)
        if was_deleted:
            success_msg = f"Successfully deleted fact #{fact_id}: {fact.content}"
            console.print(
                Panel(Markdown(success_msg), title="Fact Deleted", border_style="green")
            )
            log_work_event(f"Deleted fact {fact_id}.")
            return success_msg
        else:
            return f"Failed to delete fact #{fact_id}"
    else:
        return f"Fact #{fact_id} not found"


def run_key_facts_gc_agent() -> None:
    """Run the key facts gc agent to maintain a reasonable number of key facts.
    
    The agent analyzes all key facts and determines which are the least valuable,
    deleting them to maintain a manageable collection size of high-value facts.
    """
    # Get the count of key facts
    facts = key_fact_repository.get_all()
    fact_count = len(facts)
    
    # Display status panel with fact count included
    console.print(Panel(f"Gathering my thoughts...\nCurrent number of key facts: {fact_count}", title="🗑️ Garbage Collection"))
    
    # Only run the agent if we actually have facts to clean
    if fact_count > 0:
        # Get all facts as a formatted string for the prompt
        facts_dict = key_fact_repository.get_facts_dict()
        formatted_facts = "\n".join([f"Fact #{k}: {v}" for k, v in facts_dict.items()])
        
        # Retrieve configuration
        llm_config = _global_memory.get("config", {})

        # Initialize the LLM model
        model = initialize_llm(
            llm_config.get("provider", "anthropic"),
            llm_config.get("model", "claude-3-7-sonnet-20250219"),
            temperature=llm_config.get("temperature")
        )
        
        # Create the agent with the delete_key_fact tool
        agent = create_agent(model, [delete_key_fact])
        
        # Format the prompt with the current facts
        prompt = KEY_FACTS_GC_PROMPT.format(key_facts=formatted_facts)
        
        # Set up the agent configuration
        agent_config = {
            "recursion_limit": 50  # Set a reasonable recursion limit
        }
        
        # Run the agent
        run_agent_with_retry(agent, prompt, agent_config)
        
        # Get updated count
        updated_facts = key_fact_repository.get_all()
        updated_count = len(updated_facts)
        
        # Show info panel with updated count
        console.print(
            Panel(
                f"Cleaned key facts: {fact_count} → {updated_count}",
                title="🗑️ GC Complete"
            )
        )
    else:
        console.print(Panel("No key facts to clean.", title="🗑️ GC Info"))