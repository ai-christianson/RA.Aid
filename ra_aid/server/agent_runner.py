"""
Agent runner module for managing agent thread creation and execution.

This module provides functions for starting agent threads with proper configuration
and session management. It encapsulates the logic for initializing LLMs, selecting tools,
creating agent instances, and starting threads.
"""

import threading
from typing import Optional, Dict, Any, List

from ra_aid.utils.agent_thread_manager import register_agent, is_agent_running
from ra_aid.agent_utils import run_agent_with_retry, create_agent
from ra_aid.llm import initialize_llm
from ra_aid.config import get_config_repository
from ra_aid.logging_config import get_logger
from ra_aid.database.repositories.session_repository import get_session_repository

logger = get_logger(__name__)

def start_agent_thread_for_session(
    session_id: int,
    task_prompt: str,
    agent_type_name: str = "master",  # e.g., "master", "research", "web_research"
) -> bool:
    """
    Initializes and starts an agent in a new thread for a given session.
    
    This function encapsulates the agent creation and execution logic.
    It's responsible for setting the initial session status to 'running'.
    
    Args:
        session_id: The ID of the session to start an agent for
        task_prompt: The prompt to give to the agent
        agent_type_name: The type of agent to create (default: "master")
        
    Returns:
        bool: True if the agent thread was started successfully, False otherwise
    """
    logger.info(f"Attempting to start agent thread for session {session_id}, type: {agent_type_name}")
    logger.info(f"Task prompt: '{task_prompt[:50]}...'")  # Log first 50 chars of prompt

    # Check if an agent is already running for this session
    if is_agent_running(session_id):
        logger.warning(f"Agent for session {session_id} is already running. Aborting start.")
        return False

    session_repo = get_session_repository()

    try:
        # Get configuration for the agent
        config_repo = get_config_repository()
        provider = config_repo.get("provider", "anthropic")
        model_name = config_repo.get("model", config_repo.get(f"{agent_type_name}_model", "claude-3-opus-20240229"))
        temperature = config_repo.get("temperature", 0.1)

        # Initialize LLM
        llm_model = initialize_llm(provider, model_name, temperature=temperature)
        
        # Determine tools based on agent_type_name
        tools_key_map = {
            "master": "master_agent_tools",
            "research": "research_agent_tools",
            "web_research": "web_research_agent_tools",
        }
        tools_config_key = tools_key_map.get(agent_type_name, "master_agent_tools")  # Default to master
        
        # Get tools for the agent
        try:
            # This is a placeholder - the actual implementation would depend on how tools are loaded
            # For now, we'll assume there's a function to get tools based on a config key
            from ra_aid.tools import get_tools_for_agent_type
            tools = get_tools_for_agent_type(tools_config_key)
        except Exception as e:
            logger.error(f"Failed to get tools for agent type {tools_config_key}: {e}")
            tools = []  # Fallback to empty tools list

        if not tools:
            logger.warning(f"No tools loaded for agent type {tools_config_key}. Agent may be limited.")

        # Create agent instance
        agent_instance = create_agent(llm_model, tools, session_id=session_id)

        # Define the target function for the thread
        def agent_task_wrapper():
            logger.info(f"Agent task wrapper started for session {session_id}.")
            # run_agent_with_retry handles its own status updates and unregisters itself
            run_agent_with_retry(agent_instance, task_prompt, session_id=session_id)
            logger.info(f"Agent task wrapper for session {session_id} has finished.")

        # Create thread and stop event
        stop_event = threading.Event()
        thread_name = f"agent_session_{session_id}"
        
        agent_thread = threading.Thread(target=agent_task_wrapper, name=thread_name)
        agent_thread.daemon = True 

        # Register the agent thread
        register_agent(session_id, agent_thread, stop_event)
        
        # Update session status to 'running' before starting the thread
        session_repo.update_session_status(session_id, 'running')
        
        # Start the thread
        agent_thread.start()
        
        logger.info(f"Agent thread {thread_name} for session {session_id} started successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize or start agent thread for session {session_id}: {e}", exc_info=True)
        # If setup fails, ensure session status is 'error'
        try:
            session_repo.update_session_status(session_id, 'error')
        except Exception as db_error:
            logger.error(f"Additionally, failed to update session {session_id} status to error: {db_error}")
        return False
