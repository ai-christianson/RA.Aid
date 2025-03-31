"""
Module to get usage statistics for sessions.

This module provides functions to retrieve session usage statistics
from the database.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

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


def get_latest_session_usage(project_dir: Optional[str] = None, db_path: Optional[str] = None) -> Tuple[Dict[str, Any], int]:
    """
    Get usage statistics for the latest session.
    
    This function retrieves the latest session and calculates its total usage metrics.
    
    Args:
        project_dir: Optional directory path where the .ra-aid folder is located.
                    Defaults to current working directory if not specified.
        db_path: Optional direct path to the database file. Takes precedence over project_dir if specified.
    
    Returns:
        Tuple[Dict[str, Any], int]: A tuple containing:
            - Dictionary with session info and usage metrics
            - Status code (0 for success, 1 for error)
    """
    try:
        # Determine the base directory for database operations
        base_dir = None
        if db_path:
            # If direct db_path is provided, extract its directory
            db_file = Path(db_path)
            if not db_file.exists():
                return create_empty_result(f"Database file not found: {db_path}"), 1
            base_dir = str(db_file.parent.parent)  # Go up one level from the db file
        elif project_dir:
            # If project_dir is provided, use it
            base_dir = project_dir
            # Check if .ra-aid directory exists in the specified project directory
            ra_aid_dir = Path(os.path.join(base_dir, ".ra-aid"))
            if not ra_aid_dir.exists():
                return create_empty_result(f"No .ra-aid directory found in {project_dir}"), 1
        
        # Ensure database migrations are applied
        try:
            # Handle the case where ensure_migrations_applied doesn't accept base_dir
            original_dir = None
            if base_dir:
                original_dir = os.getcwd()
                os.chdir(base_dir)
                
            migration_result = ensure_migrations_applied()
            
            # Restore original directory if we changed it
            if original_dir:
                os.chdir(original_dir)
                
            if not migration_result:
                return create_empty_result("Database migrations failed"), 1
        except Exception as e:
            # Restore original directory if we changed it
            if base_dir and original_dir:
                os.chdir(original_dir)
            return create_empty_result(f"Database migration error: {str(e)}"), 1
            
        # Initialize database connection using DatabaseManager context
        with DatabaseManager(base_dir=base_dir) as db:
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
