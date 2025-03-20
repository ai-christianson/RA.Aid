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
import sys

import peewee

from ra_aid.database.models import Trajectory, HumanInput
from ra_aid.database.pydantic_models import TrajectoryModel
from ra_aid.database.repositories.session_repository import get_session_repository
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

    def __enter__(self) -> "TrajectoryRepository":
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


def get_trajectory_repository() -> "TrajectoryRepository":
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

    def _to_model(self, trajectory: Optional[Trajectory]) -> Optional[TrajectoryModel]:
        """
        Convert a Peewee Trajectory object to a Pydantic TrajectoryModel.

        Args:
            trajectory: Peewee Trajectory instance or None

        Returns:
            Optional[TrajectoryModel]: Pydantic model representation or None if trajectory is None
        """
        if trajectory is None:
            return None

        return TrajectoryModel.model_validate(trajectory, from_attributes=True)

    def create(
        self,
        tool_name: Optional[str] = None,
        tool_parameters: Optional[Dict[str, Any]] = None,
        tool_result: Optional[Dict[str, Any]] = None,
        step_data: Optional[Dict[str, Any]] = None,
        record_type: str = "tool_execution",
        human_input_id: Optional[int] = None,
        session_id: Optional[int] = None,
        current_cost: Optional[float] = None,  # Cost of the last LLM message
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        is_error: bool = False,
        error_message: Optional[str] = None,
        error_type: Optional[str] = None,
        error_details: Optional[str] = None,
    ) -> TrajectoryModel:
        """
        Create a new trajectory record in the database.

        Args:
            tool_name: Optional name of the tool that was executed
            tool_parameters: Optional parameters passed to the tool (will be JSON encoded)
            tool_result: Result returned by the tool (will be JSON encoded)
            step_data: UI rendering data (will be JSON encoded)
            record_type: Type of trajectory record
            human_input_id: Optional ID of the associated human input
            current_cost: cost of last llm message
            input_tokens: Optional input/prompt token usage
            output_tokens: Optional output/completion token usage
            is_error: Flag indicating if this record represents an error (default: False)
            error_message: The error message (if is_error is True)
            error_type: The type/class of the error (if is_error is True)
            error_details: Additional error details like stack traces (if is_error is True)
            session_id: Specify session_id or fetches current session id by default

        Returns:
            TrajectoryModel: The newly created trajectory instance as a Pydantic model

        Raises:
            peewee.DatabaseError: If there's an error creating the record
        """
        try:
            # Serialize JSON fields
            tool_parameters_json = (
                json.dumps(tool_parameters) if tool_parameters is not None else None
            )
            tool_result_json = (
                json.dumps(tool_result) if tool_result is not None else None
            )
            step_data_json = json.dumps(step_data) if step_data is not None else None

            # Create human input reference if provided
            human_input = None
            if human_input_id is not None:
                try:
                    human_input = HumanInput.get_by_id(human_input_id)
                except peewee.DoesNotExist:
                    logger.warning(f"Human input with ID {human_input_id} not found")

            new_session_id = session_id
            if not session_id:
                session_repo = get_session_repository()
                session_record = session_repo.get_current_session_record()
                new_session_id = session_record.get_id()
                

            trajectory = Trajectory.create(
                human_input=human_input,
                session=new_session_id,
                tool_name=tool_name or "",  # Use empty string if tool_name is None
                tool_parameters=tool_parameters_json,
                tool_result=tool_result_json,
                step_data=step_data_json,
                record_type=record_type,
                current_cost=current_cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                is_error=is_error,
                error_message=error_message,
                error_type=error_type,
                error_details=error_details,
            )
            if tool_name:
                logger.debug(
                    f"Created trajectory record ID {trajectory.id} for tool: {tool_name}"
                )
            else:
                logger.debug(
                    f"Created trajectory record ID {trajectory.id} of type: {record_type}"
                )
            return self._to_model(trajectory)
        except peewee.DatabaseError as e:
            logger.error(f"Failed to create trajectory record: {str(e)}")
            raise


    def get(self, trajectory_id: int) -> Optional[TrajectoryModel]:
        """
        Retrieve a trajectory record by its ID.

        Args:
            trajectory_id: The ID of the trajectory record to retrieve

        Returns:
            Optional[TrajectoryModel]: The trajectory instance as a Pydantic model if found, None otherwise

        Raises:
            peewee.DatabaseError: If there's an error accessing the database
        """
        try:
            trajectory = Trajectory.get_or_none(Trajectory.id == trajectory_id)
            return self._to_model(trajectory)
        except peewee.DatabaseError as e:
            logger.error(f"Failed to fetch trajectory {trajectory_id}: {str(e)}")
            raise

    def update(
        self,
        trajectory_id: int,
        tool_result: Optional[Dict[str, Any]] = None,
        step_data: Optional[Dict[str, Any]] = None,
        current_cost: Optional[float] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        is_error: Optional[bool] = None,
        error_message: Optional[str] = None,
        error_type: Optional[str] = None,
        error_details: Optional[str] = None,
    ) -> Optional[TrajectoryModel]:
        """
        Update an existing trajectory record.

        This is typically used to update the result or metrics after tool execution completes.

        Args:
            trajectory_id: The ID of the trajectory record to update
            tool_result: Updated tool result (will be JSON encoded)
            step_data: Updated UI rendering data (will be JSON encoded)
            current_cost: Cost of last llm message
            input_tokens: Updated input/prompt token usage
            output_tokens: Updated output/completion token usage
            is_error: Flag indicating if this record represents an error
            error_message: The error message
            error_type: The type/class of the error
            error_details: Additional error details like stack traces

        Returns:
            Optional[TrajectoryModel]: The updated trajectory as a Pydantic model if found, None otherwise

        Raises:
            peewee.DatabaseError: If there's an error updating the record
        """
        try:
            # First check if the trajectory exists
            peewee_trajectory = Trajectory.get_or_none(Trajectory.id == trajectory_id)
            if not peewee_trajectory:
                logger.warning(
                    f"Attempted to update non-existent trajectory {trajectory_id}"
                )
                return None

            # Update the fields if provided
            update_data = {}

            if tool_result is not None:
                update_data["tool_result"] = json.dumps(tool_result)

            if step_data is not None:
                update_data["step_data"] = json.dumps(step_data)

            if current_cost is not None:
                update_data["current_cost"] = current_cost

            if input_tokens is not None:
                update_data["input_tokens"] = input_tokens

            if output_tokens is not None:
                update_data["output_tokens"] = output_tokens

            if is_error is not None:
                update_data["is_error"] = is_error

            if error_message is not None:
                update_data["error_message"] = error_message

            if error_type is not None:
                update_data["error_type"] = error_type

            if error_details is not None:
                update_data["error_details"] = error_details

            if update_data:
                query = Trajectory.update(**update_data).where(
                    Trajectory.id == trajectory_id
                )
                query.execute()
                logger.debug(f"Updated trajectory record ID {trajectory_id}")
                return self.get(trajectory_id)

            return self._to_model(peewee_trajectory)
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
            trajectory = Trajectory.get_or_none(Trajectory.id == trajectory_id)
            if not trajectory:
                logger.warning(
                    f"Attempted to delete non-existent trajectory {trajectory_id}"
                )
                return False

            # Delete the trajectory
            trajectory.delete_instance()
            logger.debug(f"Deleted trajectory record ID {trajectory_id}")
            return True
        except peewee.DatabaseError as e:
            logger.error(f"Failed to delete trajectory {trajectory_id}: {str(e)}")
            raise

    def get_all(self) -> Dict[int, TrajectoryModel]:
        """
        Retrieve all trajectory records from the database.

        Returns:
            Dict[int, TrajectoryModel]: Dictionary mapping trajectory IDs to trajectory Pydantic models

        Raises:
            peewee.DatabaseError: If there's an error accessing the database
        """
        try:
            trajectories = Trajectory.select().order_by(Trajectory.id)
            return {
                trajectory.id: self._to_model(trajectory) for trajectory in trajectories
            }
        except peewee.DatabaseError as e:
            logger.error(f"Failed to fetch all trajectories: {str(e)}")
            raise

    def get_trajectories_by_human_input(
        self, human_input_id: int
    ) -> List[TrajectoryModel]:
        """
        Retrieve all trajectory records associated with a specific human input.

        Args:
            human_input_id: The ID of the human input to get trajectories for

        Returns:
            List[TrajectoryModel]: List of trajectory Pydantic models associated with the human input

        Raises:
            peewee.DatabaseError: If there's an error accessing the database
        """
        try:
            trajectories = list(
                Trajectory.select()
                .where(Trajectory.human_input == human_input_id)
                .order_by(Trajectory.id)
            )
            return [self._to_model(trajectory) for trajectory in trajectories]
        except peewee.DatabaseError as e:
            logger.error(
                f"Failed to fetch trajectories for human input {human_input_id}: {str(e)}"
            )
            raise

    def get_parsed_trajectory(self, trajectory_id: int) -> Optional[TrajectoryModel]:
        """
        Get a trajectory record with JSON fields parsed into dictionaries.

        Args:
            trajectory_id: ID of the trajectory to retrieve

        Returns:
            Optional[TrajectoryModel]: The trajectory as a Pydantic model with parsed JSON fields,
                                      or None if not found
        """
        return self.get(trajectory_id)

    def get_session_usage_totals(self, session_id: int) -> Dict[str, Any]:
        """
        Calculate total usage metrics for a session by summing all trajectory records.

        Args:
            session_id: The ID of the session to calculate totals for

        Returns:
            Dict[str, Any]: Dictionary containing total cost, tokens, input tokens, and output tokens

        Raises:
            peewee.DatabaseError: If there's an error accessing the database
        """
        try:
            # Use SQL aggregation instead of Python computation
            query = (
                Trajectory
                .select(
                    peewee.fn.COALESCE(peewee.fn.SUM(Trajectory.current_cost), 0.0).alias('total_cost'),
                    peewee.fn.COALESCE(peewee.fn.SUM(Trajectory.input_tokens), 0).alias('total_input_tokens'),
                    peewee.fn.COALESCE(peewee.fn.SUM(Trajectory.output_tokens), 0).alias('total_output_tokens')
                )
                .where(
                    (Trajectory.session == session_id)
                    & (Trajectory.record_type == "model_usage")
                )
                .dicts()
                .get()
            )
            
            # Extract the results from the query
            totals = {
                "total_cost": float(query['total_cost']),
                "total_input_tokens": int(query['total_input_tokens']),
                "total_output_tokens": int(query['total_output_tokens'])
            }
            
            # Calculate total tokens from input and output tokens
            totals["total_tokens"] = (
                totals["total_input_tokens"] + totals["total_output_tokens"]
            )

            logger.debug(
                f"Calculated session {session_id} totals: "
                f"cost=${totals['total_cost']:.6f}, "
                f"tokens={totals['total_tokens']}, "
                f"input={totals['total_input_tokens']}, "
                f"output={totals['total_output_tokens']}"
            )

            return totals
        except peewee.DatabaseError as e:
            logger.error(f"Failed to calculate session usage totals: {str(e)}")
            raise

    def get_trajectories_by_session(self, session_id: int) -> List[TrajectoryModel]:
        """
        Retrieve all trajectory records associated with a specific session.

        Args:
            session_id: The ID of the session to get trajectories for

        Returns:
            List[TrajectoryModel]: List of trajectory Pydantic models associated with the session

        Raises:
            peewee.DatabaseError: If there's an error accessing the database
        """
        try:
            trajectories = list(
                Trajectory.select()
                .where(Trajectory.session == session_id)
                .order_by(Trajectory.created_at)
            )
            return [self._to_model(trajectory) for trajectory in trajectories]
        except peewee.DatabaseError as e:
            logger.error(
                f"Failed to fetch trajectories for session {session_id}: {str(e)}"
            )
            raise
