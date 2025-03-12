"""
Trajectory repository implementation for database access.

This module provides a repository implementation for the Trajectory model,
following the repository pattern for data access abstraction. It handles
operations for storing and retrieving agent action trajectories.
"""

from typing import Dict, List, Optional, Any, Union
import contextvars
import json
import logging

import peewee

from ra_aid.database.models import Trajectory, HumanInput
from ra_aid.logging_config import get_logger

logger = get_logger(__name__)

# Create contextvar to hold the TrajectoryRepository instance
trajectory_repo_var = contextvars.ContextVar("trajectory_repo", default=None)


class TrajectoryRepositoryManager:
    """
    Context manager for TrajectoryRepository.

    This class provides a context manager interface for TrajectoryRepository,
    using the contextvars approach for thread safety.

    Example:
        with DatabaseManager() as db:
            with TrajectoryRepositoryManager(db) as repo:
                # Use the repository
                trajectory = repo.create(
                    tool_name="ripgrep_search",
                    tool_parameters={"pattern": "example"}
                )
                all_trajectories = repo.get_all()
    """

    def __init__(self, db):
        """
        Initialize the TrajectoryRepositoryManager.

        Args:
            db: Database connection to use (required)
        """
        self.db = db

    def __enter__(self) -> 'TrajectoryRepository':
        """
        Initialize the TrajectoryRepository and return it.

        Returns:
            TrajectoryRepository: The initialized repository
        """
        repo = TrajectoryRepository(self.db)
        trajectory_repo_var.set(repo)
        return repo

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[object],
    ) -> None:
        """
        Reset the repository when exiting the context.

        Args:
            exc_type: The exception type if an exception was raised
            exc_val: The exception value if an exception was raised
            exc_tb: The traceback if an exception was raised
        """
        # Reset the contextvar to None
        trajectory_repo_var.set(None)

        # Don't suppress exceptions
        return False


def get_trajectory_repository() -> 'TrajectoryRepository':
    """
    Get the current TrajectoryRepository instance.

    Returns:
        TrajectoryRepository: The current repository instance
        
    Raises:
        RuntimeError: If no repository has been initialized with TrajectoryRepositoryManager
    """
    repo = trajectory_repo_var.get()
    if repo is None:
        raise RuntimeError(
            "No TrajectoryRepository available. "
            "Make sure to initialize one with TrajectoryRepositoryManager first."
        )
    return repo


class TrajectoryRepository:
    """
    Repository for managing Trajectory database operations.
    
    This class provides methods for performing CRUD operations on the Trajectory model,
    abstracting the database access details from the business logic. It handles
    serialization and deserialization of JSON fields for tool parameters, results,
    and UI rendering data.
    
    Example:
        with DatabaseManager() as db:
            with TrajectoryRepositoryManager(db) as repo:
                trajectory = repo.create(
                    tool_name="ripgrep_search",
                    tool_parameters={"pattern": "example"}
                )
                all_trajectories = repo.get_all()
    """
    
    def __init__(self, db):
        """
        Initialize the repository with a database connection.
        
        Args:
            db: Database connection to use (required)
        """
        if db is None:
            raise ValueError("Database connection is required for TrajectoryRepository")
        self.db = db
    
    def create(
        self,
        tool_name: Optional[str] = None,
        tool_parameters: Optional[Dict[str, Any]] = None,
        tool_result: Optional[Dict[str, Any]] = None,
        step_data: Optional[Dict[str, Any]] = None,
        record_type: str = "tool_execution",
        human_input_id: Optional[int] = None,
        cost: Optional[float] = None,
        tokens: Optional[int] = None,
        is_error: bool = False,
        error_message: Optional[str] = None,
        error_type: Optional[str] = None,
        error_details: Optional[str] = None
    ) -> Trajectory:
        """
        Create a new trajectory record in the database.
        
        Args:
            tool_name: Optional name of the tool that was executed
            tool_parameters: Optional parameters passed to the tool (will be JSON encoded)
            tool_result: Result returned by the tool (will be JSON encoded)
            step_data: UI rendering data (will be JSON encoded)
            record_type: Type of trajectory record
            human_input_id: Optional ID of the associated human input
            cost: Optional cost of the operation (placeholder)
            tokens: Optional token usage (placeholder)
            is_error: Flag indicating if this record represents an error (default: False)
            error_message: The error message (if is_error is True)
            error_type: The type/class of the error (if is_error is True)
            error_details: Additional error details like stack traces (if is_error is True)
            
        Returns:
            Trajectory: The newly created trajectory instance
            
        Raises:
            peewee.DatabaseError: If there's an error creating the record
        """
        try:
            # Serialize JSON fields
            tool_parameters_json = json.dumps(tool_parameters) if tool_parameters is not None else None
            tool_result_json = json.dumps(tool_result) if tool_result is not None else None
            step_data_json = json.dumps(step_data) if step_data is not None else None
            
            # Create human input reference if provided
            human_input = None
            if human_input_id is not None:
                try:
                    human_input = HumanInput.get_by_id(human_input_id)
                except peewee.DoesNotExist:
                    logger.warning(f"Human input with ID {human_input_id} not found")
            
            # Create the trajectory record
            trajectory = Trajectory.create(
                human_input=human_input,
                tool_name=tool_name or "",  # Use empty string if tool_name is None
                tool_parameters=tool_parameters_json,
                tool_result=tool_result_json,
                step_data=step_data_json,
                record_type=record_type,
                cost=cost,
                tokens=tokens,
                is_error=is_error,
                error_message=error_message,
                error_type=error_type,
                error_details=error_details
            )
            if tool_name:
                logger.debug(f"Created trajectory record ID {trajectory.id} for tool: {tool_name}")
            else:
                logger.debug(f"Created trajectory record ID {trajectory.id} of type: {record_type}")
            return trajectory
        except peewee.DatabaseError as e:
            logger.error(f"Failed to create trajectory record: {str(e)}")
            raise
    
    def get(self, trajectory_id: int) -> Optional[Trajectory]:
        """
        Retrieve a trajectory record by its ID.
        
        Args:
            trajectory_id: The ID of the trajectory record to retrieve
            
        Returns:
            Optional[Trajectory]: The trajectory instance if found, None otherwise
            
        Raises:
            peewee.DatabaseError: If there's an error accessing the database
        """
        try:
            return Trajectory.get_or_none(Trajectory.id == trajectory_id)
        except peewee.DatabaseError as e:
            logger.error(f"Failed to fetch trajectory {trajectory_id}: {str(e)}")
            raise
    
    def update(
        self,
        trajectory_id: int,
        tool_result: Optional[Dict[str, Any]] = None,
        step_data: Optional[Dict[str, Any]] = None,
        cost: Optional[float] = None,
        tokens: Optional[int] = None,
        is_error: Optional[bool] = None,
        error_message: Optional[str] = None,
        error_type: Optional[str] = None,
        error_details: Optional[str] = None
    ) -> Optional[Trajectory]:
        """
        Update an existing trajectory record.
        
        This is typically used to update the result or metrics after tool execution completes.
        
        Args:
            trajectory_id: The ID of the trajectory record to update
            tool_result: Updated tool result (will be JSON encoded)
            step_data: Updated UI rendering data (will be JSON encoded)
            cost: Updated cost information
            tokens: Updated token usage information
            is_error: Flag indicating if this record represents an error
            error_message: The error message 
            error_type: The type/class of the error
            error_details: Additional error details like stack traces
            
        Returns:
            Optional[Trajectory]: The updated trajectory if found, None otherwise
            
        Raises:
            peewee.DatabaseError: If there's an error updating the record
        """
        try:
            # First check if the trajectory exists
            trajectory = self.get(trajectory_id)
            if not trajectory:
                logger.warning(f"Attempted to update non-existent trajectory {trajectory_id}")
                return None
            
            # Update the fields if provided
            update_data = {}
            
            if tool_result is not None:
                update_data["tool_result"] = json.dumps(tool_result)
            
            if step_data is not None:
                update_data["step_data"] = json.dumps(step_data)
            
            if cost is not None:
                update_data["cost"] = cost
            
            if tokens is not None:
                update_data["tokens"] = tokens
                
            if is_error is not None:
                update_data["is_error"] = is_error
                
            if error_message is not None:
                update_data["error_message"] = error_message
                
            if error_type is not None:
                update_data["error_type"] = error_type
                
            if error_details is not None:
                update_data["error_details"] = error_details
            
            if update_data:
                query = Trajectory.update(**update_data).where(Trajectory.id == trajectory_id)
                query.execute()
                logger.debug(f"Updated trajectory record ID {trajectory_id}")
                return self.get(trajectory_id)
            
            return trajectory
        except peewee.DatabaseError as e:
            logger.error(f"Failed to update trajectory {trajectory_id}: {str(e)}")
            raise
    
    def delete(self, trajectory_id: int) -> bool:
        """
        Delete a trajectory record by its ID.
        
        Args:
            trajectory_id: The ID of the trajectory record to delete
            
        Returns:
            bool: True if the record was deleted, False if it wasn't found
            
        Raises:
            peewee.DatabaseError: If there's an error deleting the record
        """
        try:
            # First check if the trajectory exists
            trajectory = self.get(trajectory_id)
            if not trajectory:
                logger.warning(f"Attempted to delete non-existent trajectory {trajectory_id}")
                return False
            
            # Delete the trajectory
            trajectory.delete_instance()
            logger.debug(f"Deleted trajectory record ID {trajectory_id}")
            return True
        except peewee.DatabaseError as e:
            logger.error(f"Failed to delete trajectory {trajectory_id}: {str(e)}")
            raise
    
    def get_all(self) -> Dict[int, Trajectory]:
        """
        Retrieve all trajectory records from the database.
        
        Returns:
            Dict[int, Trajectory]: Dictionary mapping trajectory IDs to trajectory instances
            
        Raises:
            peewee.DatabaseError: If there's an error accessing the database
        """
        try:
            return {trajectory.id: trajectory for trajectory in Trajectory.select().order_by(Trajectory.id)}
        except peewee.DatabaseError as e:
            logger.error(f"Failed to fetch all trajectories: {str(e)}")
            raise
    
    def get_trajectories_by_human_input(self, human_input_id: int) -> List[Trajectory]:
        """
        Retrieve all trajectory records associated with a specific human input.
        
        Args:
            human_input_id: The ID of the human input to get trajectories for
            
        Returns:
            List[Trajectory]: List of trajectory instances associated with the human input
            
        Raises:
            peewee.DatabaseError: If there's an error accessing the database
        """
        try:
            return list(Trajectory.select().where(Trajectory.human_input == human_input_id).order_by(Trajectory.id))
        except peewee.DatabaseError as e:
            logger.error(f"Failed to fetch trajectories for human input {human_input_id}: {str(e)}")
            raise
    
    def parse_json_field(self, json_str: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Parse a JSON string into a Python dictionary.
        
        Args:
            json_str: JSON string to parse
            
        Returns:
            Optional[Dict[str, Any]]: Parsed dictionary or None if input is None or invalid
        """
        if not json_str:
            return None
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON field: {str(e)}")
            return None
    
    def get_parsed_trajectory(self, trajectory_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a trajectory record with JSON fields parsed into dictionaries.
        
        Args:
            trajectory_id: ID of the trajectory to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: Dictionary with trajectory data and parsed JSON fields,
                                   or None if not found
        """
        trajectory = self.get(trajectory_id)
        if trajectory is None:
            return None
        
        return {
            "id": trajectory.id,
            "created_at": trajectory.created_at,
            "updated_at": trajectory.updated_at,
            "tool_name": trajectory.tool_name,
            "tool_parameters": self.parse_json_field(trajectory.tool_parameters),
            "tool_result": self.parse_json_field(trajectory.tool_result),
            "step_data": self.parse_json_field(trajectory.step_data),
            "record_type": trajectory.record_type,
            "cost": trajectory.cost,
            "tokens": trajectory.tokens,
            "human_input_id": trajectory.human_input.id if trajectory.human_input else None,
            "is_error": trajectory.is_error,
            "error_message": trajectory.error_message,
            "error_type": trajectory.error_type,
            "error_details": trajectory.error_details,
        }