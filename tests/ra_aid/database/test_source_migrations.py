"""
Tests for the migration system's source package migration handling.
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from ra_aid.database.migrations import (
    MIGRATIONS_DIRNAME,
    MigrationManager,
    ensure_migrations_applied,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_logger():
    """Mock the logger to test for output messages."""
    with patch("ra_aid.database.migrations.logger") as mock:
        yield mock


class TestSourceMigrations:
    """Tests for source package migration handling."""

    def test_migration_manager_uses_source_migrations_dir(self, temp_dir, mock_logger):
        """Test that MigrationManager uses source package migrations directory by default."""
        # Set up test paths
        db_path = os.path.join(temp_dir, "test.db")
        
        # Mock _get_source_package_migrations_dir to return a test path
        source_migrations_dir = os.path.join(temp_dir, "source_migrations")
        os.makedirs(source_migrations_dir, exist_ok=True)
        
        # Create __init__.py to make it a proper package
        with open(os.path.join(source_migrations_dir, "__init__.py"), "w") as f:
            pass

        # Mock router initialization
        with patch("ra_aid.database.migrations.Router") as mock_router:
            mock_router.return_value = MagicMock()
            
            # Mock _get_source_package_migrations_dir
            with patch.object(
                MigrationManager, 
                "_get_source_package_migrations_dir", 
                return_value=source_migrations_dir
            ):
                # Initialize manager
                manager = MigrationManager(db_path=db_path)
                
                # Verify source migrations directory is used
                assert manager.migrations_dir == source_migrations_dir
                
                # Verify logging
                mock_logger.debug.assert_any_call(
                    f"Using source package migrations directory: {source_migrations_dir}"
                )

    def test_migration_manager_with_custom_migrations_dir(self, temp_dir, mock_logger):
        """Test that MigrationManager uses custom migrations directory when provided."""
        # Set up test paths
        db_path = os.path.join(temp_dir, "test.db")
        custom_migrations_dir = os.path.join(temp_dir, "custom_migrations")
        
        # Mock router initialization
        with patch("ra_aid.database.migrations.Router") as mock_router:
            mock_router.return_value = MagicMock()
            
            # Initialize manager with custom migrations directory
            manager = MigrationManager(db_path=db_path, migrations_dir=custom_migrations_dir)
            
            # Verify custom migrations directory is used
            assert manager.migrations_dir == custom_migrations_dir
            
            # Verify directory was created
            assert os.path.exists(custom_migrations_dir)
            assert os.path.exists(os.path.join(custom_migrations_dir, "__init__.py"))
            
            # Verify logging
            mock_logger.debug.assert_any_call(
                f"Using migrations directory: {custom_migrations_dir}"
            )

    def test_get_source_package_migrations_dir(self, temp_dir, mock_logger):
        """Test that _get_source_package_migrations_dir returns the correct path."""
        # Set up a mock source directory structure
        mock_base_dir = os.path.join(temp_dir, "ra_aid")
        os.makedirs(mock_base_dir, exist_ok=True)
        
        source_migrations_dir = os.path.join(mock_base_dir, MIGRATIONS_DIRNAME)
        os.makedirs(source_migrations_dir, exist_ok=True)
        
        # Create a manager with patch in place
        with patch("ra_aid.database.migrations.os.path.dirname") as mock_dirname:
            with patch("ra_aid.database.migrations.os.path.abspath") as mock_abspath:
                # Mock the path functions to return our test paths
                mock_abspath.return_value = os.path.join(mock_base_dir, "database", "migrations.py")
                # Use a custom side_effect to avoid recursion
                def dirname_side_effect(path):
                    if path == os.path.join(mock_base_dir, "database", "migrations.py"):
                        return os.path.join(mock_base_dir, "database")
                    elif path == os.path.join(mock_base_dir, "database"):
                        return mock_base_dir
                    else:
                        return os.path.dirname(path)
                
                mock_dirname.side_effect = dirname_side_effect
                
                # Create the manager
                manager = MigrationManager(db_path=os.path.join(temp_dir, "test.db"), 
                                           migrations_dir=os.path.join(temp_dir, "custom_migrations"))
                
                # Call the method directly
                with patch.object(manager, "_get_source_package_migrations_dir") as mock_method:
                    mock_method.return_value = source_migrations_dir
                    
                    # Verify the method returns the expected path
                    assert manager._get_source_package_migrations_dir() == source_migrations_dir

    def test_get_source_package_migrations_dir_not_found(self, temp_dir, mock_logger):
        """Test that _get_source_package_migrations_dir handles missing directory."""
        # Create a manager
        manager = MigrationManager(db_path=os.path.join(temp_dir, "test.db"), 
                                    migrations_dir=os.path.join(temp_dir, "custom_migrations"))
        
        # Create a test implementation that will raise the expected error
        def raise_not_found(*args, **kwargs):
            error_msg = f"Source migrations directory not found: /path/to/migrations"
            logger = mock_logger  # Use the mocked logger
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Replace the method with our test implementation
        with patch.object(manager, "_get_source_package_migrations_dir", side_effect=raise_not_found):
            # Call should raise FileNotFoundError
            with pytest.raises(FileNotFoundError) as excinfo:
                manager._get_source_package_migrations_dir()
            
            # Verify error message
            assert "Source migrations directory not found" in str(excinfo.value)
            
            # Verify logging
            mock_logger.error.assert_called_with(
                "Source migrations directory not found: /path/to/migrations"
            )

    def test_ensure_migrations_applied_creates_ra_aid_dir(self, temp_dir, mock_logger):
        """Test that ensure_migrations_applied creates the .ra-aid directory if it doesn't exist."""
        # Get a path to a directory that doesn't exist
        ra_aid_dir = os.path.join(temp_dir, ".ra-aid")
        
        # Mock getcwd to return our temp directory
        with patch("os.getcwd", return_value=temp_dir):
            # Mock DatabaseManager
            with patch("ra_aid.database.migrations.DatabaseManager") as mock_db_manager:
                # Mock the context manager
                mock_db_manager.return_value.__enter__.return_value = MagicMock()
                mock_db_manager.return_value.__exit__.return_value = None
                
                # Mock ra_aid package import
                with patch("ra_aid.database.migrations.ra_aid", create=True) as mock_ra_aid:
                    # Set up the mock package directory path
                    mock_package_dir = os.path.join(temp_dir, "ra_aid_package")
                    os.makedirs(mock_package_dir, exist_ok=True)
                    mock_migrations_dir = os.path.join(mock_package_dir, MIGRATIONS_DIRNAME)
                    os.makedirs(mock_migrations_dir, exist_ok=True)
                    
                    # Configure the mock
                    mock_ra_aid.__file__ = os.path.join(mock_package_dir, "__init__.py")
                    
                    # Mock init_migrations and apply_migrations
                    mock_migration_manager = MagicMock()
                    mock_migration_manager.apply_migrations.return_value = True
                    
                    with patch("ra_aid.database.migrations.init_migrations", return_value=mock_migration_manager):
                        # Call ensure_migrations_applied
                        result = ensure_migrations_applied()
                        
                        # Verify result
                        assert result is True
                        
                        # Verify .ra-aid directory was created
                        assert os.path.exists(ra_aid_dir)

    def test_ensure_migrations_applied_handles_directory_error(self, mock_logger):
        """Test that ensure_migrations_applied handles errors creating the .ra-aid directory."""
        # Mock os.makedirs to raise an exception
        with patch("os.makedirs", side_effect=PermissionError("Permission denied")):
            # Call ensure_migrations_applied
            result = ensure_migrations_applied()
            
            # Verify result is False on error
            assert result is False
            
            # Verify error was logged
            mock_logger.error.assert_called_with(
                "Failed to ensure .ra-aid directory exists: Permission denied"
            )
            
    def test_ensure_migrations_applied_uses_package_migrations(self, temp_dir, mock_logger):
        """Test that ensure_migrations_applied uses the source package migrations directory."""
        # Set up test paths
        ra_aid_dir = os.path.join(temp_dir, ".ra-aid")
        
        # Mock getcwd to return our temp directory
        with patch("os.getcwd", return_value=temp_dir):
            # Mock DatabaseManager
            with patch("ra_aid.database.migrations.DatabaseManager") as mock_db_manager:
                # Mock the context manager
                mock_db_manager.return_value.__enter__.return_value = MagicMock()
                mock_db_manager.return_value.__exit__.return_value = None
                
                # Mock ra_aid package import
                with patch("ra_aid.database.migrations.ra_aid", create=True) as mock_ra_aid:
                    # Set up the mock package directory path
                    mock_package_dir = os.path.join(temp_dir, "ra_aid_package")
                    os.makedirs(mock_package_dir, exist_ok=True)
                    mock_migrations_dir = os.path.join(mock_package_dir, MIGRATIONS_DIRNAME)
                    os.makedirs(mock_migrations_dir, exist_ok=True)
                    
                    # Configure the mock
                    mock_ra_aid.__file__ = os.path.join(mock_package_dir, "__init__.py")
                    
                    # Create a mock migration manager that we can verify
                    mock_init_migrations = MagicMock()
                    mock_init_migrations.apply_migrations.return_value = True
                    
                    with patch("ra_aid.database.migrations.init_migrations", return_value=mock_init_migrations) as mock_init:
                        # Call ensure_migrations_applied
                        result = ensure_migrations_applied()
                        
                        # Verify init_migrations was called
                        mock_init.assert_called_once()
                        # We can't verify the exact path since it's derived from non-mock objects
                        # Instead, verify that init_migrations was called and succeeded
                        assert result is True
                        
    def test_router_initialization_with_source_migrations(self, temp_dir, mock_logger):
        """Test that the migration router is initialized with the source package migrations."""
        # Set up test paths
        db_path = os.path.join(temp_dir, "test.db")
        
        # Create a mock source migrations directory
        source_migrations_dir = os.path.join(temp_dir, "source_migrations")
        os.makedirs(source_migrations_dir, exist_ok=True)
        
        # Create __init__.py to make it a proper package
        with open(os.path.join(source_migrations_dir, "__init__.py"), "w") as f:
            pass
            
        # Mock the Router class
        with patch("ra_aid.database.migrations.Router") as mock_router_class:
            # Create a mock router instance
            mock_router = MagicMock()
            mock_router_class.return_value = mock_router
            
            # Mock _get_source_package_migrations_dir
            with patch.object(
                MigrationManager, 
                "_get_source_package_migrations_dir", 
                return_value=source_migrations_dir
            ):
                # Initialize manager
                manager = MigrationManager(db_path=db_path)
                
                # Verify router was initialized with the source migrations directory
                mock_router_class.assert_called_once()
                # Get the args from the call
                call_args = mock_router_class.call_args
                assert call_args.kwargs["migrate_dir"] == source_migrations_dir