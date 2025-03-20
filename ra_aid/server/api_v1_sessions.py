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
from ra_aid.database.pydantic_models import SessionModel, TrajectoryModel

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