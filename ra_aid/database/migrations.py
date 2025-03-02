"""
Database migrations for ra_aid.

This module provides functionality for managing database schema migrations
using peewee-migrate. It includes tools for creating, checking, and applying
migrations automatically.
"""

import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from peewee_migrate import Router

from ra_aid.database.connection import DatabaseManager, get_db
from ra_aid.logging_config import get_logger

logger = get_logger(__name__)

# Constants
MIGRATIONS_DIRNAME = "migrations"
MIGRATIONS_TABLE = "migrationshistory"


class MigrationManager:
    """
    Manages database migrations for the ra_aid application.

    This class provides methods to initialize the migrator, check for
    pending migrations, apply migrations, and create new migrations.
    """

    def __init__(
        self, db_path: Optional[str] = None, migrations_dir: Optional[str] = None
    ):
        """
        Initialize the MigrationManager.

        Args:
            db_path: Optional path to the database file. If None, uses the default.
            migrations_dir: Optional path to the migrations directory. If None, uses source package migrations.
        """
        self.db = get_db()

        # Determine database path
        if db_path is None:
            # Get current working directory
            cwd = os.getcwd()
            ra_aid_dir = os.path.join(cwd, ".ra-aid")
            db_path = os.path.join(ra_aid_dir, "pk.db")

        self.db_path = db_path

        # Determine migrations directory
        if migrations_dir is None:
            # Use the source package migrations directory
            migrations_dir = self._get_source_package_migrations_dir()
            logger.debug(f"Using source package migrations directory: {migrations_dir}")
        else:
            # Use the specified migrations directory
            # Ensure the directory exists if a custom path is provided
            self._ensure_migrations_dir(migrations_dir)

        self.migrations_dir = migrations_dir

        # Initialize router
        self.router = self._init_router()

    def _get_source_package_migrations_dir(self) -> str:
        """
        Get the path to the migrations directory in the source package.

        Returns:
            str: Path to the source package migrations directory
        """
        try:
            # Get the base directory of the ra_aid package
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            source_migrations_dir = os.path.join(base_dir, MIGRATIONS_DIRNAME)
            
            if not os.path.exists(source_migrations_dir):
                error_msg = f"Source migrations directory not found: {source_migrations_dir}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
                
            logger.debug(f"Found source migrations directory: {source_migrations_dir}")
            return source_migrations_dir
        except Exception as e:
            error_msg = f"Failed to locate source migrations directory: {str(e)}"
            logger.error(error_msg)
            raise

    def _ensure_migrations_dir(self, migrations_dir: str) -> None:
        """
        Ensure that the migrations directory exists.

        Creates the directory if it doesn't exist.
        
        Args:
            migrations_dir: Path to the migrations directory
        """
        try:
            migrations_path = Path(migrations_dir)
            if not migrations_path.exists():
                logger.debug(f"Creating migrations directory at: {migrations_dir}")
                migrations_path.mkdir(parents=True, exist_ok=True)

                # Create __init__.py to make it a proper package
                init_file = migrations_path / "__init__.py"
                if not init_file.exists():
                    init_file.touch()

            logger.debug(f"Using migrations directory: {migrations_dir}")
        except Exception as e:
            logger.error(f"Failed to create migrations directory: {str(e)}")
            raise

    def _init_router(self) -> Router:
        """
        Initialize the peewee-migrate Router.

        Returns:
            Router: Configured peewee-migrate Router instance
        """
        try:
            router = Router(
                self.db, migrate_dir=self.migrations_dir, migrate_table=MIGRATIONS_TABLE
            )
            logger.debug(f"Initialized migration router with table: {MIGRATIONS_TABLE}")
            return router
        except Exception as e:
            logger.error(f"Failed to initialize migration router: {str(e)}")
            raise

    def check_migrations(self) -> Tuple[List[str], List[str]]:
        """
        Check for pending migrations.

        Returns:
            Tuple[List[str], List[str]]: A tuple containing (applied_migrations, pending_migrations)
        """
        try:
            # Get all migrations
            all_migrations = self.router.todo

            # Get applied migrations
            applied = self.router.done

            # Calculate pending migrations
            pending = [m for m in all_migrations if m not in applied]

            logger.debug(
                f"Found {len(applied)} applied migrations and {len(pending)} pending migrations"
            )
            return applied, pending
        except Exception as e:
            logger.error(f"Failed to check migrations: {str(e)}")
            return [], []

    def apply_migrations(self, fake: bool = False) -> bool:
        """
        Apply all pending migrations.

        Args:
            fake: If True, mark migrations as applied without running them

        Returns:
            bool: True if migrations were applied successfully, False otherwise
        """
        try:
            # Get pending migrations
            _, pending = self.check_migrations()

            if not pending:
                logger.info("No pending migrations to apply")
                return True

            logger.info(f"Applying {len(pending)} pending migrations...")

            # Apply migrations
            for migration in pending:
                try:
                    logger.info(f"Applying migration: {migration}")
                    self.router.run(migration, fake=fake)
                    logger.info(f"Successfully applied migration: {migration}")
                except Exception as e:
                    logger.error(f"Failed to apply migration {migration}: {str(e)}")
                    return False

            logger.info(f"Successfully applied {len(pending)} migrations")
            return True
        except Exception as e:
            logger.error(f"Failed to apply migrations: {str(e)}")
            return False

    def create_migration(self, name: str, auto: bool = True) -> Optional[str]:
        """
        Create a new migration.

        Args:
            name: Name of the migration
            auto: If True, automatically detect model changes

        Returns:
            Optional[str]: The name of the created migration, or None if creation failed
        """
        try:
            # Sanitize migration name
            safe_name = name.replace(" ", "_").lower()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            migration_name = f"{timestamp}_{safe_name}"

            logger.info(f"Creating new migration: {migration_name}")

            # Create migration
            self.router.create(migration_name, auto=auto)

            logger.info(f"Successfully created migration: {migration_name}")
            return migration_name
        except Exception as e:
            logger.error(f"Failed to create migration: {str(e)}")
            return None

    def get_migration_status(self) -> Dict[str, Any]:
        """
        Get the current migration status.

        Returns:
            Dict[str, Any]: A dictionary containing migration status information
        """
        applied, pending = self.check_migrations()

        return {
            "applied_count": len(applied),
            "pending_count": len(pending),
            "applied": applied,
            "pending": pending,
            "migrations_dir": self.migrations_dir,
            "db_path": self.db_path,
        }


def init_migrations(
    db_path: Optional[str] = None, migrations_dir: Optional[str] = None
) -> MigrationManager:
    """
    Initialize the migration manager.

    Args:
        db_path: Optional path to the database file
        migrations_dir: Optional path to the migrations directory

    Returns:
        MigrationManager: Initialized migration manager
    """
    return MigrationManager(db_path, migrations_dir)


def ensure_migrations_applied() -> bool:
    """
    Check for and apply any pending migrations.
    
    Creates the .ra-aid directory if it doesn't exist,
    but uses migrations directly from the source package.
    
    This function should be called during application startup to ensure
    the database schema is up to date.

    Returns:
        bool: True if migrations were applied successfully or none were pending
    """
    try:
        # Ensure .ra-aid directory exists for the database file
        cwd = os.getcwd()
        ra_aid_dir = os.path.join(cwd, ".ra-aid")
        os.makedirs(ra_aid_dir, exist_ok=True)
        
        # Use source package migrations directory
        import ra_aid
        package_dir = os.path.dirname(os.path.abspath(ra_aid.__file__))
        migrations_dir = os.path.join(package_dir, MIGRATIONS_DIRNAME)
        
        with DatabaseManager() as db:
            try:
                migration_manager = init_migrations(migrations_dir=migrations_dir)
                return migration_manager.apply_migrations()
            except Exception as e:
                logger.error(f"Failed to apply migrations: {str(e)}")
                return False
    except Exception as e:
        logger.error(f"Failed to ensure .ra-aid directory exists: {str(e)}")
        return False


def create_new_migration(name: str, auto: bool = True) -> Optional[str]:
    """
    Create a new migration with the given name.

    Args:
        name: Name of the migration
        auto: If True, automatically detect model changes

    Returns:
        Optional[str]: The name of the created migration, or None if creation failed
    """
    with DatabaseManager() as db:
        try:
            migration_manager = init_migrations()
            return migration_manager.create_migration(name, auto)
        except Exception as e:
            logger.error(f"Failed to create migration: {str(e)}")
            return None


def get_migration_status() -> Dict[str, Any]:
    """
    Get the current migration status.

    Returns:
        Dict[str, Any]: A dictionary containing migration status information
    """
    with DatabaseManager() as db:
        try:
            migration_manager = init_migrations()
            return migration_manager.get_migration_status()
        except Exception as e:
            logger.error(f"Failed to get migration status: {str(e)}")
            return {
                "error": str(e),
                "applied_count": 0,
                "pending_count": 0,
                "applied": [],
                "pending": [],
            }