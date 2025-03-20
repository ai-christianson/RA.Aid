"""
Module to get usage statistics for sessions.

This module provides functions to retrieve session usage statistics
from the database.
"""

from typing import Dict, Any, Tuple

from ..database import DatabaseManager, ensure_migrations_applied
from ..database.repositories.session_repository import SessionRepositoryManager
from ..database.repositories.trajectory_repository import TrajectoryRepositoryManager


def create_empty_result(error_message=None):
    """Create a default result dictionary with zeros for all metrics."""
    result = {
        "total_cost": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0
    }
    
    if error_message:
        result["error"] = error_message
        
    return result


def get_latest_session_usage() -> Tuple[Dict[str, Any], int]:
    """
    Get usage statistics for the latest session.
    
    This function retrieves the latest session and calculates its total usage metrics.
    
    Returns:
        Tuple[Dict[str, Any], int]: A tuple containing:
            - Dictionary with session info and usage metrics
            - Status code (0 for success, 1 for error)
    """
    try:
        # Ensure database migrations are applied
        try:
            migration_result = ensure_migrations_applied()
            if not migration_result:
                return create_empty_result("Database migrations failed"), 1
        except Exception as e:
            return create_empty_result(f"Database migration error: {str(e)}"), 1
            
        # Initialize database connection using DatabaseManager context
        with DatabaseManager() as db:
            # Get the latest session
            with SessionRepositoryManager(db) as session_repo:
                latest_session = session_repo.get_latest_session()
                
                if latest_session is None:
                    return create_empty_result("No sessions found in database"), 1
                
                # Get usage totals for the session
                with TrajectoryRepositoryManager(db) as trajectory_repo:
                    usage_totals = trajectory_repo.get_session_usage_totals(latest_session.id)
                
                    # Create result object with session info and usage totals
                    result = {
                        "session_id": latest_session.id,
                        "session_start_time": latest_session.start_time.isoformat() if latest_session.start_time else None,
                        "session_display_name": latest_session.display_name,
                        **usage_totals  # Unpack usage totals directly
                    }
                    
                    return result, 0
    except Exception as e:
        return create_empty_result(str(e)), 1
