"""
Database package for ra_aid.

This package provides database functionality for the ra_aid application,
including connection management, models, and utility functions.
"""

from ra_aid.database.connection import (
    init_db,
    get_db,
    close_db,
    DatabaseManager
)
from ra_aid.database.models import BaseModel
from ra_aid.database.utils import get_model_count, truncate_table, ensure_tables_created

__all__ = [
    'init_db',
    'get_db',
    'close_db',
    'DatabaseManager',
    'BaseModel',
    'get_model_count',
    'truncate_table',
    'ensure_tables_created',
]
