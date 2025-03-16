"""API router for spawning an RA.Aid agent."""

import threading
import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ra_aid.database.repositories.session_repository import SessionRepository, get_session_repository
from ra_aid.database.connection import DatabaseManager
from ra_aid.database.repositories.session_repository import SessionRepositoryManager
from ra_aid.database.repositories.key_fact_repository import KeyFactRepositoryManager
from ra_aid.database.repositories.key_snippet_repository import KeySnippetRepositoryManager
from ra_aid.database.repositories.human_input_repository import HumanInputRepositoryManager
from ra_aid.database.repositories.research_note_repository import ResearchNoteRepositoryManager
from ra_aid.database.repositories.related_files_repository import RelatedFilesRepositoryManager
from ra_aid.database.repositories.trajectory_repository import TrajectoryRepositoryManager
from ra_aid.database.repositories.work_log_repository import WorkLogRepositoryManager
from ra_aid.database.repositories.config_repository import ConfigRepositoryManager
from ra_aid.env_inv_context import EnvInvManager
from ra_aid.env_inv import EnvDiscovery
from ra_aid.database import ensure_migrations_applied

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
        expert_enabled: Whether to enable expert assistance (default: True)
        web_research_enabled: Whether to enable web research (default: False)
    """
    message: str = Field(
        description="The message or task for the agent to process"
    )
    research_only: bool = Field(
        default=False,
        description="Whether to use research-only mode"
    )
    expert_enabled: bool = Field(
        default=True,
        description="Whether to enable expert assistance"
    )
    web_research_enabled: bool = Field(
        default=False,
        description="Whether to enable web research"
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
    
    
def get_repository() -> SessionRepository:
    """
    Get the SessionRepository instance.
    
    This function is used as a FastAPI dependency and can be overridden
    in tests using dependency_overrides.
    
    Returns:
        SessionRepository: The repository instance
    """
    return get_session_repository()

def run_agent_thread(
    message: str,
    session_id: str,
    research_only: bool = False,
    expert_enabled: bool = True,
    web_research_enabled: bool = False,
):
    """
    Run a research agent in a separate thread with proper repository initialization.
    
    Args:
        message: The message or task for the agent to process
        session_id: The ID of the session to associate with this agent
        research_only: Whether to use research-only mode
        expert_enabled: Whether to enable expert assistance
        web_research_enabled: Whether to enable web research
    """
    try:
        logger.info(f"Starting agent thread for session {session_id}")
        
        # Initialize environment discovery
        env_discovery = EnvDiscovery()
        env_discovery.discover()
        env_data = env_discovery.format_markdown()
        
        # Apply any pending database migrations
        try:
            migration_result = ensure_migrations_applied()
            if not migration_result:
                logger.warning("Database migrations failed but execution will continue")
        except Exception as e:
            logger.error(f"Database migration error: {str(e)}")
        
        # Initialize empty config dictionary
        config = {}
        
        # Initialize database connection and repositories
        with DatabaseManager() as db, \
             SessionRepositoryManager(db) as session_repo, \
             KeyFactRepositoryManager(db) as key_fact_repo, \
             KeySnippetRepositoryManager(db) as key_snippet_repo, \
             HumanInputRepositoryManager(db) as human_input_repo, \
             ResearchNoteRepositoryManager(db) as research_note_repo, \
             RelatedFilesRepositoryManager() as related_files_repo, \
             TrajectoryRepositoryManager(db) as trajectory_repo, \
             WorkLogRepositoryManager() as work_log_repo, \
             ConfigRepositoryManager(config) as config_repo, \
             EnvInvManager(env_data) as env_inv:
            
            # Import here to avoid circular imports
            from ra_aid.__main__ import run_research_agent
            
            # Run the research agent
            run_research_agent(
                base_task_or_query=message,
                model=None,  # Use default model
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
    repo: SessionRepository = Depends(get_repository),
) -> SpawnAgentResponse:
    """
    Spawn a new RA.Aid agent to process a message or task.
    
    Args:
        request: Request body with message and agent configuration
        repo: SessionRepository dependency injection
        
    Returns:
        SpawnAgentResponse: Response with session ID
        
    Raises:
        HTTPException: With a 500 status code if there's an error spawning the agent
    """
    try:
        # Create a new session
        metadata = {
            "agent_type": "research-only" if request.research_only else "research",
            "expert_enabled": request.expert_enabled,
            "web_research_enabled": request.web_research_enabled,
        }
        session = repo.create_session(metadata=metadata)
        
        # Start the agent thread
        thread = threading.Thread(
            target=run_agent_thread,
            args=(
                request.message,
                str(session.id),
                request.research_only,
                request.expert_enabled,
                request.web_research_enabled,
            )
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