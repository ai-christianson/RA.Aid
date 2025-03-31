"""
Module to get usage statistics for all sessions.

This module provides functions to retrieve usage statistics for all sessions
from the database.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

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


def get_all_sessions_usage(project_dir: Optional[str] = None, db_path: Optional[str] = None) -> Tuple[List[Dict[str, Any]], int]:
    """
    Get usage statistics for all sessions.
    
    This function retrieves all sessions and calculates their usage metrics.
    
    Args:
        project_dir: Optional directory path where the .ra-aid folder is located.
                    Defaults to current working directory if not specified.
        db_path: Optional direct path to the database file. Takes precedence over project_dir if specified.
    
    Returns:
        Tuple[List[Dict[str, Any]], int]: A tuple containing:
            - List of dictionaries with session info and usage metrics
            - Status code (0 for success, 1 for error)
    """
    try:
        # Determine the base directory for database operations
        base_dir = None
        if db_path:
            # If direct db_path is provided, extract its directory
            db_file = Path(db_path)
            if not db_file.exists():
                return [create_empty_result(f"Database file not found: {db_path}")], 1
            base_dir = str(db_file.parent.parent)  # Go up one level from the db file
        elif project_dir:
            # If project_dir is provided, use it
            base_dir = project_dir
            # Check if .ra-aid directory exists in the specified project directory
            ra_aid_dir = Path(os.path.join(base_dir, ".ra-aid"))
            if not ra_aid_dir.exists():
                return [create_empty_result(f"No .ra-aid directory found in {project_dir}")], 1
        
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
                return [create_empty_result("Database migrations failed")], 1
        except Exception as e:
            # Restore original directory if we changed it
            if base_dir and original_dir:
                os.chdir(original_dir)
            return [create_empty_result(f"Database migration error: {str(e)}")], 1
            
        # Initialize database connection using DatabaseManager context
        with DatabaseManager(base_dir=base_dir) as db:
            # Get all session IDs
            with SessionRepositoryManager(db) as session_repo:
                session_ids = session_repo.get_all_session_ids()
                
                if not session_ids:
                    return [create_empty_result("No sessions found in database")], 1
                
                # Get usage totals for each session
                results = []
                with TrajectoryRepositoryManager(db) as trajectory_repo:
                    for session_id in session_ids:
                        # Get session details
                        session = session_repo.get(session_id)
                        if not session:
                            continue
                            
                        # Get usage totals for the session
                        usage_totals = trajectory_repo.get_session_usage_totals(session_id)
                        
                        # Create result object with session info and usage totals
                        result = {
                            "session_id": session.id,
                            "session_start_time": session.start_time.isoformat() if session.start_time else None,
                            "session_display_name": session.display_name,
                            **usage_totals  # Unpack usage totals directly
                        }
                        
                        results.append(result)
                    
                    # Calculate grand totals
                    grand_total = {
                        "session_id": "all",
                        "session_display_name": "All Sessions",
                        "total_cost": sum(r["total_cost"] for r in results),
                        "total_input_tokens": sum(r["total_input_tokens"] for r in results),
                        "total_output_tokens": sum(r["total_output_tokens"] for r in results),
                        "total_tokens": sum(r["total_tokens"] for r in results)
                    }
                    
                    # Add grand total to the beginning of the results
                    results.insert(0, grand_total)
                    
                    return results, 0
    except Exception as e:
        return [create_empty_result(str(e))], 1


def main():
    """
    Command-line entry point for getting usage statistics for all sessions.
    
    This function retrieves all sessions and calculates their usage metrics,
    then outputs the results as JSON to stdout.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Get usage statistics for all sessions")
    parser.add_argument("--project-dir", help="Directory containing the .ra-aid folder (defaults to current directory)")
    parser.add_argument("--db-path", help="Direct path to the database file (takes precedence over project-dir)")
    
    args = parser.parse_args()
    
    results, status_code = get_all_sessions_usage(
        project_dir=args.project_dir,
        db_path=args.db_path
    )
    print(json.dumps(results, indent=2))
    return status_code


if __name__ == "__main__":
    main()
