"""
Tests for the database migrations module.
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from ra_aid.database.connection import DatabaseManager, db_var
from ra_aid.database.migrations import (
    MIGRATIONS_DIRNAME,
    MIGRATIONS_TABLE,
    MigrationManager,
    create_new_migration,
    ensure_migrations_applied,
    get_migration_status,
    init_migrations,
)


@pytest.fixture
def cleanup_db():
    """Reset the database contextvar and connection state after each test."""
    # Reset before the test
    db = db_var.get()
    if db is not None:
        try:
            if not db.is_closed():
                db.close()
        except Exception:
            # Ignore errors when closing the database
            pass
    db_var.set(None)

    # Run the test
    yield

    # Reset after the test
    db = db_var.get()
    if db is not None:
        try:
            if not db.is_closed():
                db.close()
        except Exception:
            # Ignore errors when closing the database
            pass
    db_var.set(None)


@pytest.fixture
def mock_logger():
    """Mock the logger to test for output messages."""
    with patch("ra_aid.database.migrations.logger") as mock:
        yield mock


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_migrations_dir(temp_dir):
    """Create a temporary migrations directory."""
    migrations_dir = os.path.join(temp_dir, MIGRATIONS_DIRNAME)
    os.makedirs(migrations_dir, exist_ok=True)
    # Create __init__.py to make it a proper package
    with open(os.path.join(migrations_dir, "__init__.py"), "w") as f:
        pass
    yield migrations_dir


@pytest.fixture
def mock_router():
    """Mock the peewee_migrate Router class."""
    with patch("ra_aid.database.migrations.Router") as mock:
        # Configure the mock router
        mock_instance = MagicMock()
        mock.return_value = mock_instance

        # Set up router properties
        mock_instance.todo = ["001_initial", "002_add_users"]
        mock_instance.done = ["001_initial"]

        yield mock_instance


class TestMigrationManager:
    """Tests for the MigrationManager class."""

    def test_init(self, cleanup_db, temp_dir, mock_logger):
        """Test MigrationManager initialization."""
        # Set up test paths
        db_path = os.path.join(temp_dir, "test.db")
        migrations_dir = os.path.join(temp_dir, MIGRATIONS_DIRNAME)

        # Initialize manager
        manager = MigrationManager(db_path=db_path, migrations_dir=migrations_dir)

        # Verify initialization
        assert manager.db_path == db_path
        assert manager.migrations_dir == migrations_dir
        assert os.path.exists(migrations_dir)
        assert os.path.exists(os.path.join(migrations_dir, "__init__.py"))

        # Verify router initialization was logged
        mock_logger.debug.assert_any_call(
            f"Using migrations directory: {migrations_dir}"
        )
        mock_logger.debug.assert_any_call(
            f"Initialized migration router with table: {MIGRATIONS_TABLE}"
        )

    def test_ensure_migrations_dir(self, cleanup_db, temp_dir, mock_logger):
        """Test _ensure_migrations_dir creates directory if it doesn't exist."""
        # Set up test paths
        db_path = os.path.join(temp_dir, "test.db")
        migrations_dir = os.path.join(temp_dir, "nonexistent_dir", MIGRATIONS_DIRNAME)

        # Initialize manager
        manager = MigrationManager(db_path=db_path, migrations_dir=migrations_dir)

        # Verify directory was created
        assert os.path.exists(migrations_dir)
        assert os.path.exists(os.path.join(migrations_dir, "__init__.py"))

        # Verify creation was logged
        mock_logger.debug.assert_any_call(
            f"Creating migrations directory at: {migrations_dir}"
        )

    def test_ensure_migrations_dir_error(self, cleanup_db, mock_logger):
        """Test _ensure_migrations_dir handles errors."""
        # Mock os.makedirs to raise an exception
        with patch(
            "pathlib.Path.mkdir", side_effect=PermissionError("Permission denied")
        ):
            # Set up test paths - use a path that would require elevated permissions
            db_path = "/root/test.db"
            migrations_dir = "/root/migrations"

            # Initialize manager should raise an exception
            with pytest.raises(Exception):
                manager = MigrationManager(
                    db_path=db_path, migrations_dir=migrations_dir
                )

            # Verify error was logged
            mock_logger.error.assert_called_with(
                "Failed to create migrations directory: [Errno 13] Permission denied: '/root/migrations'"
            )

    def test_init_router(self, cleanup_db, temp_dir, mock_router):
        """Test _init_router initializes the Router correctly."""
        # Set up test paths
        db_path = os.path.join(temp_dir, "test.db")
        migrations_dir = os.path.join(temp_dir, MIGRATIONS_DIRNAME)

        # Create the migrations directory
        os.makedirs(migrations_dir, exist_ok=True)

        # Initialize manager with mocked Router
        with patch("ra_aid.database.migrations.Router", return_value=mock_router):
            manager = MigrationManager(db_path=db_path, migrations_dir=migrations_dir)

            # Verify router was initialized
            assert manager.router == mock_router

    def test_check_migrations(self, cleanup_db, temp_dir, mock_router, mock_logger):
        """Test check_migrations returns correct migration lists."""
        # Set up test paths
        db_path = os.path.join(temp_dir, "test.db")
        migrations_dir = os.path.join(temp_dir, MIGRATIONS_DIRNAME)

        # Initialize manager with mocked Router
        with patch("ra_aid.database.migrations.Router", return_value=mock_router):
            manager = MigrationManager(db_path=db_path, migrations_dir=migrations_dir)

            # Call check_migrations
            applied, pending = manager.check_migrations()

            # Verify results
            assert applied == ["001_initial"]
            assert pending == ["002_add_users"]

            # Verify logging
            mock_logger.debug.assert_called_with(
                "Found 1 applied migrations and 1 pending migrations"
            )

    def test_check_migrations_error(self, cleanup_db, temp_dir, mock_logger):
        """Test check_migrations handles errors."""
        # Set up test paths
        db_path = os.path.join(temp_dir, "test.db")
        migrations_dir = os.path.join(temp_dir, MIGRATIONS_DIRNAME)

        # Create a mock router with a property that raises an exception
        mock_router = MagicMock()
        # Configure the todo property to raise an exception when accessed
        type(mock_router).todo = PropertyMock(side_effect=Exception("Test error"))
        mock_router.done = []

        # Initialize manager with the mocked Router
        with patch("ra_aid.database.migrations.Router", return_value=mock_router):
            manager = MigrationManager(db_path=db_path, migrations_dir=migrations_dir)

            # Directly call check_migrations on the manager with the mocked router
            applied, pending = manager.check_migrations()

            # Verify empty results are returned on error
            assert applied == []
            assert pending == []

            # Verify error was logged
            mock_logger.error.assert_called_with(
                "Failed to check migrations: Test error"
            )

    def test_apply_migrations(self, cleanup_db, temp_dir, mock_router, mock_logger):
        """Test apply_migrations applies pending migrations."""
        # Set up test paths
        db_path = os.path.join(temp_dir, "test.db")
        migrations_dir = os.path.join(temp_dir, MIGRATIONS_DIRNAME)

        # Initialize manager with mocked Router
        with patch("ra_aid.database.migrations.Router", return_value=mock_router):
            manager = MigrationManager(db_path=db_path, migrations_dir=migrations_dir)

            # Call apply_migrations
            result = manager.apply_migrations()

            # Verify result
            assert result is True

            # Verify migrations were applied
            mock_router.run.assert_called_once_with("002_add_users", fake=False)

            # Verify logging
            mock_logger.info.assert_any_call("Applying 1 pending migrations...")
            mock_logger.info.assert_any_call("Applying migration: 002_add_users")
            mock_logger.info.assert_any_call(
                "Successfully applied migration: 002_add_users"
            )
            mock_logger.info.assert_any_call("Successfully applied 1 migrations")

    def test_apply_migrations_no_pending(self, cleanup_db, temp_dir, mock_logger):
        """Test apply_migrations when no migrations are pending."""
        # Set up test paths
        db_path = os.path.join(temp_dir, "test.db")
        migrations_dir = os.path.join(temp_dir, MIGRATIONS_DIRNAME)

        # Create a mock router with no pending migrations
        mock_router = MagicMock()
        mock_router.todo = ["001_initial"]
        mock_router.done = ["001_initial"]

        # Initialize manager with mocked Router
        with patch("ra_aid.database.migrations.Router", return_value=mock_router):
            manager = MigrationManager(db_path=db_path, migrations_dir=migrations_dir)

            # Call apply_migrations
            result = manager.apply_migrations()

            # Verify result
            assert result is True

            # Verify no migrations were applied
            mock_router.run.assert_not_called()

            # Verify logging
            mock_logger.info.assert_called_with("No pending migrations to apply")

    def test_apply_migrations_error(self, cleanup_db, temp_dir, mock_logger):
        """Test apply_migrations handles errors during migration."""
        # Set up test paths
        db_path = os.path.join(temp_dir, "test.db")
        migrations_dir = os.path.join(temp_dir, MIGRATIONS_DIRNAME)

        # Create a mock router that raises an exception during run
        mock_router = MagicMock()
        mock_router.todo = ["001_initial", "002_add_users"]
        mock_router.done = ["001_initial"]
        mock_router.run.side_effect = Exception("Migration error")

        # Initialize manager with mocked Router
        with patch("ra_aid.database.migrations.Router", return_value=mock_router):
            manager = MigrationManager(db_path=db_path, migrations_dir=migrations_dir)

            # Call apply_migrations
            result = manager.apply_migrations()

            # Verify result
            assert result is False

            # Verify error was logged
            mock_logger.error.assert_called_with(
                "Failed to apply migration 002_add_users: Migration error"
            )

    def test_create_migration(self, cleanup_db, temp_dir, mock_router, mock_logger):
        """Test create_migration creates a new migration."""
        # Set up test paths
        db_path = os.path.join(temp_dir, "test.db")
        migrations_dir = os.path.join(temp_dir, MIGRATIONS_DIRNAME)

        # Initialize manager with mocked Router
        with patch("ra_aid.database.migrations.Router", return_value=mock_router):
            manager = MigrationManager(db_path=db_path, migrations_dir=migrations_dir)

            # Call create_migration
            result = manager.create_migration("add_users", auto=True)

            # Verify result contains timestamp and name
            assert result is not None
            assert "add_users" in result

            # Verify migration was created
            mock_router.create.assert_called_once()

            # Verify logging
            mock_logger.info.assert_any_call(f"Creating new migration: {result}")
            mock_logger.info.assert_any_call(
                f"Successfully created migration: {result}"
            )

    def test_create_migration_error(self, cleanup_db, temp_dir, mock_logger):
        """Test create_migration handles errors."""
        # Set up test paths
        db_path = os.path.join(temp_dir, "test.db")
        migrations_dir = os.path.join(temp_dir, MIGRATIONS_DIRNAME)

        # Create a mock router that raises an exception during create
        mock_router = MagicMock()
        mock_router.create.side_effect = Exception("Creation error")

        # Initialize manager with mocked Router
        with patch("ra_aid.database.migrations.Router", return_value=mock_router):
            manager = MigrationManager(db_path=db_path, migrations_dir=migrations_dir)

            # Call create_migration
            result = manager.create_migration("add_users", auto=True)

            # Verify result is None on error
            assert result is None

            # Verify error was logged
            mock_logger.error.assert_called_with(
                "Failed to create migration: Creation error"
            )

    def test_get_migration_status(self, cleanup_db, temp_dir, mock_router):
        """Test get_migration_status returns correct status information."""
        # Set up test paths
        db_path = os.path.join(temp_dir, "test.db")
        migrations_dir = os.path.join(temp_dir, MIGRATIONS_DIRNAME)

        # Initialize manager with mocked Router
        with patch("ra_aid.database.migrations.Router", return_value=mock_router):
            manager = MigrationManager(db_path=db_path, migrations_dir=migrations_dir)

            # Call get_migration_status
            status = manager.get_migration_status()

            # Verify status information
            assert status["applied_count"] == 1
            assert status["pending_count"] == 1
            assert status["applied"] == ["001_initial"]
            assert status["pending"] == ["002_add_users"]
            assert status["migrations_dir"] == migrations_dir
            assert status["db_path"] == db_path


class TestMigrationFunctions:
    """Tests for the migration utility functions."""

    def test_init_migrations(self, cleanup_db, temp_dir):
        """Test init_migrations returns a MigrationManager instance."""
        # Set up test paths
        db_path = os.path.join(temp_dir, "test.db")
        migrations_dir = os.path.join(temp_dir, MIGRATIONS_DIRNAME)

        # Call init_migrations
        with patch("ra_aid.database.migrations.MigrationManager") as mock_manager:
            mock_manager.return_value = MagicMock()

            manager = init_migrations(db_path=db_path, migrations_dir=migrations_dir)

            # Verify MigrationManager was initialized with correct parameters
            mock_manager.assert_called_once_with(db_path, migrations_dir)
            assert manager == mock_manager.return_value

    def test_ensure_migrations_applied(self, cleanup_db, mock_logger):
        """Test ensure_migrations_applied applies pending migrations."""
        # Mock MigrationManager
        mock_manager = MagicMock()
        mock_manager.apply_migrations.return_value = True

        # Call ensure_migrations_applied
        with patch(
            "ra_aid.database.migrations.init_migrations", return_value=mock_manager
        ):
            result = ensure_migrations_applied()

            # Verify result
            assert result is True

            # Verify migrations were applied
            mock_manager.apply_migrations.assert_called_once()

    def test_ensure_migrations_applied_error(self, cleanup_db, mock_logger):
        """Test ensure_migrations_applied handles errors."""
        # Call ensure_migrations_applied with an exception
        with patch(
            "ra_aid.database.migrations.init_migrations",
            side_effect=Exception("Test error"),
        ):
            result = ensure_migrations_applied()

            # Verify result is False on error
            assert result is False

            # Verify error was logged
            mock_logger.error.assert_called_with(
                "Failed to apply migrations: Test error"
            )

    def test_create_new_migration(self, cleanup_db, mock_logger):
        """Test create_new_migration creates a new migration."""
        # Mock MigrationManager
        mock_manager = MagicMock()
        mock_manager.create_migration.return_value = "20250226_123456_test_migration"

        # Call create_new_migration
        with patch(
            "ra_aid.database.migrations.init_migrations", return_value=mock_manager
        ):
            result = create_new_migration("test_migration", auto=True)

            # Verify result
            assert result == "20250226_123456_test_migration"

            # Verify migration was created
            mock_manager.create_migration.assert_called_once_with(
                "test_migration", True
            )

    def test_create_new_migration_error(self, cleanup_db, mock_logger):
        """Test create_new_migration handles errors."""
        # Call create_new_migration with an exception
        with patch(
            "ra_aid.database.migrations.init_migrations",
            side_effect=Exception("Test error"),
        ):
            result = create_new_migration("test_migration", auto=True)

            # Verify result is None on error
            assert result is None

            # Verify error was logged
            mock_logger.error.assert_called_with(
                "Failed to create migration: Test error"
            )

    def test_get_migration_status(self, cleanup_db, mock_logger):
        """Test get_migration_status returns correct status information."""
        # Mock MigrationManager
        mock_manager = MagicMock()
        mock_manager.get_migration_status.return_value = {
            "applied_count": 2,
            "pending_count": 1,
            "applied": ["001_initial", "002_add_users"],
            "pending": ["003_add_profiles"],
            "migrations_dir": "/test/migrations",
            "db_path": "/test/db.sqlite",
        }

        # Call get_migration_status
        with patch(
            "ra_aid.database.migrations.init_migrations", return_value=mock_manager
        ):
            status = get_migration_status()

            # Verify status information
            assert status["applied_count"] == 2
            assert status["pending_count"] == 1
            assert status["applied"] == ["001_initial", "002_add_users"]
            assert status["pending"] == ["003_add_profiles"]
            assert status["migrations_dir"] == "/test/migrations"
            assert status["db_path"] == "/test/db.sqlite"

            # Verify migration status was retrieved
            mock_manager.get_migration_status.assert_called_once()

    def test_get_migration_status_error(self, cleanup_db, mock_logger):
        """Test get_migration_status handles errors."""
        # Call get_migration_status with an exception
        with patch(
            "ra_aid.database.migrations.init_migrations",
            side_effect=Exception("Test error"),
        ):
            status = get_migration_status()

            # Verify default status on error
            assert status["error"] == "Test error"
            assert status["applied_count"] == 0
            assert status["pending_count"] == 0
            assert status["applied"] == []
            assert status["pending"] == []

            # Verify error was logged
            mock_logger.error.assert_called_with(
                "Failed to get migration status: Test error"
            )


class TestIntegration:
    """Integration tests for the migrations module."""

    def test_in_memory_migrations(self, cleanup_db):
        """Test migrations with in-memory database."""
        # Initialize in-memory database
        with DatabaseManager(in_memory=True) as db:
            # Create a temporary migrations directory
            with tempfile.TemporaryDirectory() as temp_dir:
                migrations_dir = os.path.join(temp_dir, MIGRATIONS_DIRNAME)
                os.makedirs(migrations_dir, exist_ok=True)

                # Create __init__.py to make it a proper package
                with open(os.path.join(migrations_dir, "__init__.py"), "w") as f:
                    pass

                # Initialize migration manager
                manager = MigrationManager(
                    db_path=":memory:", migrations_dir=migrations_dir
                )

                # Create a test migration
                migration_name = manager.create_migration("test_migration", auto=False)

                # Write a simple migration file
                migration_path = os.path.join(migrations_dir, f"{migration_name}.py")
                with open(migration_path, "w") as f:
                    f.write("""
def migrate(migrator, database, fake=False, **kwargs):
    migrator.create_table('test_table', (
        ('id', 'INTEGER', {'primary_key': True}),
        ('name', 'STRING', {'null': False}),
    ))

def rollback(migrator, database, fake=False, **kwargs):
    migrator.drop_table('test_table')
""")

                # Check migrations
                applied, pending = manager.check_migrations()
                assert len(applied) == 0
                assert len(pending) == 1
                assert (
                    migration_name in pending[0]
                )  # Instead of exact equality, check if name is contained

                # Apply migrations
                result = manager.apply_migrations()
                assert result is True

                # Check migrations again
                applied, pending = manager.check_migrations()
                assert len(applied) == 1
                assert len(pending) == 0
                assert (
                    migration_name in applied[0]
                )  # Instead of exact equality, check if name is contained

                # Verify migration status
                status = manager.get_migration_status()
                assert status["applied_count"] == 1
                assert status["pending_count"] == 0
                # Use substring check for applied migrations
                assert len(status["applied"]) == 1
                assert migration_name in status["applied"][0]
                assert status["pending"] == []
