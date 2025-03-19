#!/usr/bin/env python3
"""
Script to get usage statistics for the latest session.

This script retrieves the latest session from the database and calculates
its total usage metrics (cost and tokens), then outputs the results as JSON.
"""

import json
import sys
import os

# Add the project root to the Python path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ra_aid.database import DatabaseManager, ensure_migrations_applied
from ra_aid.database.repositories.session_repository import SessionRepositoryManager
from ra_aid.database.repositories.trajectory_repository import TrajectoryRepositoryManager


def main():
    """
    Command-line entry point for getting usage statistics for the latest session.
    
    This function retrieves the latest session and calculates its total usage metrics,
    then outputs the results as JSON to stdout.
    """
    try:
        # Ensure database migrations are applied
        try:
            migration_result = ensure_migrations_applied()
            if not migration_result:
                print(json.dumps({
                    "error": "Database migrations failed",
                    "total_cost": 0.0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_tokens": 0
                }))
                return 1
        except Exception as e:
            print(json.dumps({
                "error": f"Database migration error: {str(e)}",
                "total_cost": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0
            }))
            return 1
            
        # Initialize database connection using DatabaseManager context
        with DatabaseManager() as db:
        
            # Get the latest session
            with SessionRepositoryManager(db) as session_repo:
                latest_session = session_repo.get_latest_session()
                
                if latest_session is None:
                    print(json.dumps({
                        "error": "No sessions found in database",
                        "total_cost": 0.0,
                        "total_input_tokens": 0,
                        "total_output_tokens": 0,
                        "total_tokens": 0
                    }))
                    return 1
                
                # Get usage totals for the session
                with TrajectoryRepositoryManager(db) as trajectory_repo:
                    usage_totals = trajectory_repo.get_session_usage_totals(latest_session.id)
                
                    # Create result object with session info and usage totals
                    result = {
                        "session_id": latest_session.id,
                        "session_start_time": latest_session.start_time.isoformat() if latest_session.start_time else None,
                        "session_display_name": latest_session.display_name,
                        "total_cost": usage_totals["total_cost"],
                        "total_input_tokens": usage_totals["total_input_tokens"],
                        "total_output_tokens": usage_totals["total_output_tokens"],
                        "total_tokens": usage_totals["total_tokens"]
                    }
                    
                    # Output as JSON
                    print(json.dumps(result, indent=2))
                    return 0
    except Exception as e:
        print(json.dumps({
            "error": str(e),
            "total_cost": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0
        }))
        return 1


if __name__ == "__main__":
    sys.exit(main())
