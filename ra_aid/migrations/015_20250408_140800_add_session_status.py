# 015_20250408_140800_add_session_status.py
import peewee as pw
from peewee_migrate import Migrator
from ra_aid.logging_config import get_logger # Optional: for logging if needed

logger = get_logger(__name__) # Optional

def migrate(migrator: Migrator, database: pw.Database, fake=False, **kwargs):
    """Write your migrations here."""
    logger.info("Adding status field to session table")
    # Ensure the field doesn't already exist before adding
    # This is a safety measure, though peewee-migrate might handle it.
    # A more robust check might involve inspecting the table directly if needed.
    try:
        # Check if the column already exists - basic check using peewee introspection
        # Note: Introspection capabilities might vary between DB backends
        introspector = migrator.router.model_introspector
        columns = introspector.get_columns('session')
        if 'status' not in [c.name for c in columns]:
             migrator.add_fields(
                 'session',
                 status=pw.CharField(max_length=20, default='pending', index=True, null=False)
             )
             logger.info("Successfully added status field to session table")
        else:
             logger.warning("Column 'status' already exists in 'session' table. Skipping add.")
    except Exception as e:
        logger.error(f"Error checking or adding 'status' column: {e}")
        # Depending on requirements, you might want to raise the exception
        # or just log it and potentially continue if it's non-critical.
        # For now, just log it.


def rollback(migrator: Migrator, database: pw.Database, fake=False, **kwargs):
    """Write your rollback migrations here."""
    logger.info("Removing status field from session table")
    try:
         # Check if the column exists before trying to remove it
         introspector = migrator.router.model_introspector
         columns = introspector.get_columns('session')
         if 'status' in [c.name for c in columns]:
             migrator.remove_fields('session', 'status')
             logger.info("Successfully removed status field from session table")
         else:
             logger.warning("Column 'status' does not exist in 'session' table. Skipping remove.")
    except Exception as e:
        logger.error(f"Error checking or removing 'status' column: {e}")
        # Log error during rollback
