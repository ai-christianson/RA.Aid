#!/usr/bin/env python3
"""
API v1 Session Endpoints.

This module provides RESTful API endpoints for managing sessions.
It implements routes for creating, listing, and retrieving sessions
with proper validation and error handling.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
import peewee
from pydantic import BaseModel, Field

from ra_aid.database.repositories.session_repository import SessionRepository, get_session_repository
from ra_aid.database.repositories.trajectory_repository import TrajectoryRepository, get_trajectory_repository
from ra_aid.database.repositories.human_input_repository import HumanInputRepository, get_human_input_repository
from ra_aid.database.pydantic_models import SessionModel, TrajectoryModel
from ra_aid.utils.agent_thread_manager import stop_agent, is_agent_running

# Create API router
router = APIRouter(
    prefix="/v1/session",
    tags=["sessions"],
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Session not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Database error"},
    },
)


class PaginatedResponse(BaseModel):
    """
    Pydantic model for paginated API responses.

    This model provides a standardized format for API responses that include
    pagination, with a total count and the requested items.

    Attributes:
        total: The total number of items available
        items: List of items for the current page
        limit: The limit parameter that was used
        offset: The offset parameter that was used
    """
    total: int
    items: List[Any]
    limit: int
    offset: int


class CreateSessionRequest(BaseModel):
    """
    Pydantic model for session creation requests.

    This model provides validation for creating new sessions.

    Attributes:
        metadata: Optional dictionary of additional metadata to store with the session
    """
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional dictionary of additional metadata to store with the session"
    )


class PaginatedSessionResponse(PaginatedResponse):
    """
    Pydantic model for paginated session responses.

    This model specializes the generic PaginatedResponse for SessionModel items.

    Attributes:
        items: List of SessionModel items for the current page
    """
    items: List[SessionModel]


# Dependency to get the session repository
def get_repository() -> SessionRepository:
    """
    Get the SessionRepository instance.

    This function is used as a FastAPI dependency and can be overridden
    in tests using dependency_overrides.

    Returns:
        SessionRepository: The repository instance
    """
    return get_session_repository()


@router.get(
    "",
    response_model=PaginatedSessionResponse,
    summary="List sessions",
    description="Get a paginated list of sessions",
)
async def list_sessions(
    offset: int = Query(0, ge=0, description="Number of sessions to skip"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of sessions to return"),
    repo: SessionRepository = Depends(get_repository),
) -> PaginatedSessionResponse:
    """
    Get a paginated list of sessions.

    Args:
        offset: Number of sessions to skip (default: 0)
        limit: Maximum number of sessions to return (default: 10)
        repo: SessionRepository dependency injection

    Returns:
        PaginatedSessionResponse: Response with paginated sessions

    Raises:
        HTTPException: With a 500 status code if there's a database error
    """
    try:
        sessions, total = repo.get_all(offset=offset, limit=limit)
        return PaginatedSessionResponse(
            total=total,
            items=sessions,
            limit=limit,
            offset=offset,
        )
    except peewee.DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )


@router.get(
    "/{session_id}",
    response_model=SessionModel,
    summary="Get session",
    description="Get a specific session by ID",
)
async def get_session(
    session_id: int,
    repo: SessionRepository = Depends(get_repository),
) -> SessionModel:
    """
    Get a specific session by ID.

    Args:
        session_id: The ID of the session to retrieve
        repo: SessionRepository dependency injection

    Returns:
        SessionModel: The requested session

    Raises:
        HTTPException: With a 404 status code if the session is not found
        HTTPException: With a 500 status code if there's a database error
    """
    try:
        session = repo.get(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session with ID {session_id} not found",
            )
        return session
    except peewee.DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )


@router.post(
    "",
    response_model=SessionModel,
    status_code=status.HTTP_201_CREATED,
    summary="Create session",
    description="Create a new session",
)
async def create_session(
    request: Optional[CreateSessionRequest] = None,
    repo: SessionRepository = Depends(get_repository),
) -> SessionModel:
    """
    Create a new session.

    Args:
        request: Optional request body with session metadata
        repo: SessionRepository dependency injection

    Returns:
        SessionModel: The newly created session

    Raises:
        HTTPException: With a 500 status code if there's a database error
    """
    try:
        metadata = request.metadata if request else None
        return repo.create_session(metadata=metadata)
    except peewee.DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )


@router.get(
    "/{session_id}/trajectory",
    response_model=List[TrajectoryModel],
    summary="Get session trajectories",
    description="Get all trajectory records associated with a specific session",
)
async def get_session_trajectories(
    session_id: int,
    session_repo: SessionRepository = Depends(get_repository),
    trajectory_repo: TrajectoryRepository = Depends(get_trajectory_repository),
) -> List[TrajectoryModel]:
    """
    Get all trajectory records for a specific session.

    Args:
        session_id: The ID of the session to get trajectories for
        session_repo: SessionRepository dependency injection
        trajectory_repo: TrajectoryRepository dependency injection

    Returns:
        List[TrajectoryModel]: List of trajectory records associated with the session

    Raises:
        HTTPException: With a 404 status code if the session is not found
        HTTPException: With a 500 status code if there's a database error
    """
    # Import the logger
    from ra_aid.logging_config import get_logger
    logger = get_logger(__name__)

    logger.info(f"Fetching trajectories for session ID: {session_id}")

    try:
        # Verify the session exists
        session = session_repo.get(session_id)
        if not session:
            logger.warning(f"Session with ID {session_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session with ID {session_id} not found",
            )

        # Get trajectories for the session
        trajectories = trajectory_repo.get_trajectories_by_session(session_id)

        # Log the number of trajectories found
        logger.info(f"Found {len(trajectories)} trajectories for session ID: {session_id}")

        # If no trajectories were found, check if the database has any trajectories at all
        if not trajectories:
            # Try to get total trajectory count to verify if the DB is populated
            from ra_aid.database.models import Trajectory
            try:
                total_trajectories = Trajectory.select().count()
                logger.info(f"Total trajectories in database: {total_trajectories}")

                # Check if the migrations were applied
                from ra_aid.database.migrations import get_migration_status
                migration_status = get_migration_status()
                logger.info(
                    f"Migration status: {migration_status['applied_count']} applied, "
                    f"{migration_status['pending_count']} pending"
                )

                # If no trajectories but migrations applied, it's just empty data
                if total_trajectories == 0 and migration_status['pending_count'] == 0:
                    logger.warning(
                        "Database has no trajectories but all migrations are applied. "
                        "The database is properly set up but contains no data."
                    )
                elif migration_status['pending_count'] > 0:
                    logger.warning(
                        f"There are {migration_status['pending_count']} pending migrations. "
                        "Run migrations to ensure database is properly set up."
                    )
            except Exception as count_error:
                logger.error(f"Error checking trajectory count: {str(count_error)}")

        return trajectories
    except peewee.DatabaseError as e:
        logger.error(f"Database error fetching trajectories for session {session_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Kill/Stop session",
    description="Stop a running or pending session by ID. Sets status to 'halting', agent thread handles transition to 'halted'.",
)
async def delete_session(
    session_id: int,
    session_repo: SessionRepository = Depends(get_repository),
) -> None:
    """
    Kill a session by ID.

    Args:
        session_id: The ID of the session to kill
        session_repo: SessionRepository dependency injection

    Raises:
        HTTPException: With a 404 status code if the session is not found
        HTTPException: With a 500 status code if there's a database error
    """
    from ra_aid.logging_config import get_logger
    logger = get_logger(__name__)

    session = session_repo.get(session_id)

    if not session:
        logger.warning(f"Attempt to delete non-existent session {session_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with ID {session_id} not found",
        )

    logger.info(f"Request to stop session {session_id} with current status: {session.status}")

    # If session is already in a terminal state, no action needed
    if session.status in ['completed', 'error', 'halted']:
        logger.info(f"Session {session_id} is already in a terminal state ({session.status}). No action taken.")
        # HTTP 204 is fine, request to stop a stopped session is idempotent
        return

    # Check if agent is running
    if not is_agent_running(session_id):
        # Agent is not in the registry or thread is not alive,
        # but DB status is active (e.g. 'running', 'pending', 'halting')
        logger.warning(
            f"Session {session_id} has status '{session.status}' but no active agent thread found. "
            f"Forcing status to 'halted'."
        )
        session_repo.update_session_status(session_id, 'halted')
        from ra_aid.utils.agent_thread_manager import unregister_agent
        unregister_agent(session_id)  # Clean up registry if entry somehow exists
        return

    # Agent is running, signal it to stop
    logger.info(f"Signaling agent for session {session_id} to stop.")
    success = stop_agent(session_id)

    if not success:
        # This implies session_id was not in agent_thread_registry,
        # which contradicts is_agent_running(). This is an inconsistent state.
        logger.error(
            f"Inconsistency: is_agent_running was true for session {session_id}, but stop_agent failed. "
            f"Forcing status to 'error'."
        )
        session_repo.update_session_status(session_id, 'error')
        from ra_aid.utils.agent_thread_manager import unregister_agent
        unregister_agent(session_id)  # Attempt cleanup
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to signal stop for session {session_id} due to internal inconsistency."
        )

    # Successfully signaled. Set status to 'halting'.
    # The agent thread is responsible for updating to 'halted' upon actual termination.
    session_repo.update_session_status(session_id, 'halting')
    logger.info(f"Session {session_id} status set to 'halting'. Agent thread will handle final 'halted' state.")


@router.post(
    "/{session_id}/resume",
    response_model=SessionModel,
    summary="Resume a session",
    description="Resumes a session that was previously 'halted'.",
)
async def resume_session(
    session_id: int,
    session_repo: SessionRepository = Depends(get_repository),
    human_input_repo: HumanInputRepository = Depends(get_human_input_repository),
) -> SessionModel:
    """
    Resume a halted session.

    Args:
        session_id: The ID of the session to resume
        session_repo: SessionRepository dependency injection
        human_input_repo: HumanInputRepository dependency injection

    Returns:
        SessionModel: The resumed session

    Raises:
        HTTPException: With a 404 status code if the session is not found
        HTTPException: With a 409 status code if the session is not in a resumable state
        HTTPException: With a 500 status code if there's an error resuming the session
    """
    from ra_aid.logging_config import get_logger
    logger = get_logger(__name__)

    session = session_repo.get(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    logger.info(f"Request to resume session {session_id} with current status: {session.status}")

    # Only 'halted' sessions can be resumed
    if session.status != 'halted':
        logger.warning(f"Session {session_id} cannot be resumed. Status is '{session.status}'.")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Session is not in a resumable state (current status: {session.status}). Only 'halted' sessions can be resumed."
        )

    # Check if an agent is already running for this session
    if is_agent_running(session_id):
        logger.warning(f"Agent for session {session_id} is already running. Cannot resume.")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Agent for session {session_id} is already running."
        )

    # Retrieve the original task prompt
    initial_human_input = human_input_repo.get_first_input_for_session(session_id)
    if not initial_human_input or not initial_human_input.content:
        logger.error(f"Cannot find original task prompt for session {session_id} to resume.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Original task prompt not found for session."
        )

    original_task_prompt = initial_human_input.content

    # Determine agent_type_name based on session metadata or command_line
    agent_type_name = "master"  # Default
    if session.command_line:
        if "research-agent" in session.command_line or "run_research_agent" in session.command_line:
            agent_type_name = "research"
        elif "web-research-agent" in session.command_line or "run_web_research_agent" in session.command_line:
            agent_type_name = "web_research"

    logger.info(f"Attempting to resume session {session_id} as agent type '{agent_type_name}' with prompt: '{original_task_prompt[:50]}...'")

    # Start the agent thread
    from ra_aid.server.agent_runner import start_agent_thread_for_session
    thread_started = start_agent_thread_for_session(
        session_id=session_id,
        task_prompt=original_task_prompt,
        agent_type_name=agent_type_name,
    )

    if not thread_started:
        # start_agent_thread_for_session should set status to 'error' if it fails internally
        updated_session_on_fail = session_repo.get(session_id)
        error_detail = "Failed to start agent thread for resumption."
        if updated_session_on_fail and updated_session_on_fail.status == 'error':
            error_detail += " Session marked as 'error'."
        else:
            # If status wasn't updated to error by the starter, do it now
            session_repo.update_session_status(session_id, 'error')
            error_detail += " Session marked as 'error'."

        logger.error(f"Failed to resume session {session_id}. {error_detail}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail
        )

    # Get the updated session
    resumed_session = session_repo.get(session_id)
    if not resumed_session:
        logger.error(f"Session {session_id} disappeared after resumption attempt.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Session data lost after resumption."
        )

    logger.info(f"Session {session_id} resumed successfully. Current status: {resumed_session.status}")
    return resumed_session