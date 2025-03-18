"""API router for spawning an RA.Aid agent."""

import threading
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ra_aid.database.repositories.session_repository import SessionRepository, get_session_repository
from ra_aid.database.connection import DatabaseManager
from ra_aid.database.repositories.session_repository import SessionRepositoryManager
from ra_aid.database.repositories.key_fact_repository import KeyFactRepositoryManager
from ra_aid.database.repositories.key_snippet_repository import KeySnippetRepositoryManager
from ra_aid.database.repositories.human_input_repository import HumanInputRepositoryManager, get_human_input_repository
from ra_aid.database.repositories.research_note_repository import ResearchNoteRepositoryManager
from ra_aid.database.repositories.related_files_repository import RelatedFilesRepositoryManager
from ra_aid.database.repositories.trajectory_repository import TrajectoryRepositoryManager
from ra_aid.database.repositories.work_log_repository import WorkLogRepositoryManager
from ra_aid.database.repositories.config_repository import ConfigRepositoryManager, get_config_repository
from ra_aid.env_inv_context import EnvInvManager
from ra_aid.env_inv import EnvDiscovery
from ra_aid.llm import initialize_llm, get_model_default_temperature

# Create logger
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(
    prefix="/v1/spawn-agent",
    tags=["agent"],
    responses={
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Agent spawn error"},
    },
)

class SpawnAgentRequest(BaseModel):
    """
    Pydantic model for agent spawn requests.
    
    This model provides validation for spawning a new agent.
    
    Attributes:
        message: The message or task for the agent to process
        research_only: Whether to use research-only mode (default: False)
    """
    message: str = Field(
        description="The message or task for the agent to process"
    )
    research_only: bool = Field(
        default=False,
        description="Whether to use research-only mode"
    )

class SpawnAgentResponse(BaseModel):
    """
    Pydantic model for agent spawn responses.
    
    This model defines the response format for the spawn-agent endpoint.
    
    Attributes:
        session_id: The ID of the created session
    """
    session_id: str = Field(
        description="The ID of the created session"
    )
    
    
def run_agent_thread(
    message: str,
    session_id: str,
    source_config_repo: "ConfigRepository",
    research_only: bool = False,
    **kwargs
):
    """
    Run a research agent in a separate thread with proper repository initialization.
    
    Args:
        message: The message or task for the agent to process
        session_id: The ID of the session to associate with this agent
        source_config_repo: The source ConfigRepository to copy for this thread
        research_only: Whether to use research-only mode
        
    Note:
        Values for expert_enabled and web_research_enabled are retrieved from the
        config repository, which stores the values set during server startup.
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Starting agent thread for session {session_id}")
    
    try:
        # Initialize database connection
        db = DatabaseManager()
        
        env_discovery = EnvDiscovery()
        env_discovery.discover()
        env_data = env_discovery.format_markdown()
        
        # Get the thread configuration from kwargs
        thread_config = kwargs.get("thread_config", {})
        
        with DatabaseManager() as db, \
             SessionRepositoryManager(db) as session_repo, \
             KeyFactRepositoryManager(db) as key_fact_repo, \
             KeySnippetRepositoryManager(db) as key_snippet_repo, \
             HumanInputRepositoryManager(db) as human_input_repo, \
             ResearchNoteRepositoryManager(db) as research_note_repo, \
             RelatedFilesRepositoryManager() as related_files_repo, \
             TrajectoryRepositoryManager(db) as trajectory_repo, \
             WorkLogRepositoryManager() as work_log_repo, \
             ConfigRepositoryManager(source_repo=source_config_repo) as config_repo, \
             EnvInvManager(env_data) as env_inv:
            
            # Update config repo with values for this thread
            config_repo.set("research_only", research_only)
            
            # Update config with any thread-specific configurations
            if thread_config:
                config_repo.update(thread_config)
            
            # Import here to avoid circular imports
            from ra_aid.__main__ import run_research_agent
            
            # Get configuration values from config repository
            provider = config_repo.get("provider", "anthropic")
            model_name = config_repo.get("model", "claude-3-7-sonnet-20250219")
            temperature = kwargs.get("temperature")
            
            # If temperature is None but model supports it, use the default from model_config
            if temperature is None:
                temperature = get_model_default_temperature(provider, model_name)
            
            # Get expert_enabled and web_research_enabled from config repository
            expert_enabled = config_repo.get("expert_enabled", True)
            web_research_enabled = config_repo.get("web_research_enabled", False)
            
            # Initialize model with provider and model name from config
            model = initialize_llm(provider, model_name, temperature=temperature)
            
            # Set thread_id in config repository too
            config_repo.set("thread_id", session_id)
            
            # Create a human input record with the message and associate it with the session
            try:
                human_input_repository = get_human_input_repository()
                human_input_repository.create(
                    content=message,
                    source="server",
                    session_id=int(session_id)
                )
                logger.debug(f"Created human input record for session {session_id}")
            except Exception as e:
                logger.error(f"Failed to create human input record: {str(e)}")
            
            # Run the research agent
            run_research_agent(
                base_task_or_query=message,
                model=model,  # Use the initialized model from config
                expert_enabled=expert_enabled,
                research_only=research_only,
                hil=False,  # No human-in-the-loop for API
                web_research_enabled=web_research_enabled,
                thread_id=session_id
            )
            
            logger.info(f"Agent completed for session {session_id}")
    except Exception as e:
        logger.error(f"Error in agent thread for session {session_id}: {str(e)}")

@router.post(
    "",
    response_model=SpawnAgentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Spawn agent",
    description="Spawn a new RA.Aid agent to process a message or task",
)
async def spawn_agent(
    request: SpawnAgentRequest,
    repo: SessionRepository = Depends(get_session_repository),
) -> SpawnAgentResponse:
    """
    Spawn a new RA.Aid agent to process a message or task.
    
    Args:
        request: Request body with message and agent configuration.
        repo: SessionRepository dependency injection
        
    Returns:
        SpawnAgentResponse: Response with session ID
        
    Raises:
        HTTPException: With a 500 status code if there's an error spawning the agent
    """
    try:
        # Get configuration values from config repository
        config_repo = get_config_repository()
        expert_enabled = config_repo.get("expert_enabled", True)
        web_research_enabled = config_repo.get("web_research_enabled", False)
        provider = config_repo.get("provider", "anthropic")
        model_name = config_repo.get("model", "claude-3-7-sonnet-20250219")
        # Get temperature value (or None if not provided)
        temperature = config_repo.get("temperature")
        
        # If temperature is None, use the model's default temperature
        if temperature is None:
            temperature = get_model_default_temperature(provider, model_name)
        
        # Create a new session with config values (not request parameters)
        metadata = {
            "agent_type": "research-only" if request.research_only else "research",
            "expert_enabled": expert_enabled,
            "web_research_enabled": web_research_enabled,
        }
        session = repo.create_session(metadata=metadata)
        
        # Set the thread_id in the config repository
        config_repo.set("thread_id", str(session.id))
        
        # Get the current config values
        thread_config = {
            "provider": provider,
            "model": model_name,
            "temperature": temperature,
            "expert_enabled": expert_enabled,
            "web_research_enabled": web_research_enabled,
            "thread_id": str(session.id),
        }
        
        # Start the agent thread
        thread = threading.Thread(
            target=run_agent_thread,
            args=(
                request.message,
                str(session.id),
                config_repo,
                request.research_only,
            ),
            kwargs={
                "temperature": temperature,
                "thread_config": thread_config,
            }
        )
        thread.daemon = True  # Thread will terminate when main process exits
        thread.start()
        
        # Return the session ID
        return SpawnAgentResponse(session_id=str(session.id))
    except Exception as e:
        logger.error(f"Error spawning agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error spawning agent: {str(e)}",
        )
