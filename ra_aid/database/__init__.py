"""
Database package for ra_aid.

This package provides database functionality for the ra_aid application,
including connection management, models, utility functions, and migrations.
"""

from ra_aid.database.connection import (
    init_db,
    get_db,
    close_db,
    DatabaseManager
)
from ra_aid.database.models import BaseModel
from ra_aid.database.utils import get_model_count, truncate_table, ensure_tables_created
from ra_aid.database.migrations import (
    init_migrations,
    ensure_migrations_applied,
    create_new_migration,
    get_migration_status,
    MigrationManager
)

__all__ = [
    'init_db',
    'get_db',
    'close_db',
    'DatabaseManager',
    'BaseModel',
    'get_model_count',
    'truncate_table',
    'ensure_tables_created',
    'init_migrations',
    'ensure_migrations_applied',
    'create_new_migration',
    'get_migration_status',
    'MigrationManager',
]
