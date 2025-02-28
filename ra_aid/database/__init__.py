"""
Database package for ra_aid.

This package provides database functionality for the ra_aid application,
including connection management, models, utility functions, and migrations.
"""

from ra_aid.database.connection import DatabaseManager, close_db, get_db, init_db
from ra_aid.database.migrations import (
    MigrationManager,
    create_new_migration,
    ensure_migrations_applied,
    get_migration_status,
    init_migrations,
)
from ra_aid.database.models import BaseModel
from ra_aid.database.utils import ensure_tables_created, get_model_count, truncate_table

__all__ = [
    "init_db",
    "get_db",
    "close_db",
    "DatabaseManager",
    "BaseModel",
    "get_model_count",
    "truncate_table",
    "ensure_tables_created",
    "init_migrations",
    "ensure_migrations_applied",
    "create_new_migration",
    "get_migration_status",
    "MigrationManager",
]
