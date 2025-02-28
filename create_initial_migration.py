#!/usr/bin/env python3
"""
Create initial database migration script.

This script creates a baseline migration representing the current database schema.
It serves as the foundation for future schema changes.
"""

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ra_aid.database import DatabaseManager, create_new_migration
from ra_aid.logging_config import get_logger, setup_logging

# Set up logging
setup_logging(verbose=True)
logger = get_logger(__name__)


def create_initial_migration():
    """
    Create the initial migration for the current database schema.

    Returns:
        bool: True if migration was created successfully, False otherwise
    """
    try:
        with DatabaseManager() as db:
            # Create a descriptive name for the initial migration
            migration_name = "initial_schema"

            # Create the migration
            logger.info(f"Creating initial migration '{migration_name}'...")
            result = create_new_migration(migration_name, auto=True)

            if result:
                logger.info(f"Successfully created initial migration: {result}")
                print(f"✅ Initial migration created successfully: {result}")
                return True
            else:
                logger.error("Failed to create initial migration")
                print("❌ Failed to create initial migration")
                return False
    except Exception as e:
        logger.error(f"Error creating initial migration: {str(e)}")
        print(f"❌ Error creating initial migration: {str(e)}")
        return False


if __name__ == "__main__":
    print("Creating initial database migration...")
    success = create_initial_migration()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
