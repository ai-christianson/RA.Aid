"""Peewee migrations -- 013_20250311_191701_add_session_fk_to_trajectory.py.

Some examples (model - class or model name)::

    > Model = migrator.orm['table_name']            # Return model in current state by name
    > Model = migrator.ModelClass                   # Return model in current state by name

    > migrator.sql(sql)                             # Run custom SQL
    > migrator.run(func, *args, **kwargs)           # Run python function with the given args
    > migrator.create_model(Model)                  # Create a model (could be used as decorator)
    > migrator.remove_model(model, cascade=True)    # Remove a model
    > migrator.add_fields(model, **fields)          # Add fields to a model
    > migrator.change_fields(model, **fields)       # Change fields
    > migrator.remove_fields(model, *field_names, cascade=True)
    > migrator.rename_field(model, old_field_name, new_field_name)
    > migrator.rename_table(model, new_table_name)
    > migrator.add_index(model, *col_names, unique=False)
    > migrator.add_not_null(model, *field_names)
    > migrator.add_default(model, field_name, default)
    > migrator.add_constraint(model, name, sql)
    > migrator.drop_index(model, *col_names)
    > migrator.drop_not_null(model, *field_names)
    > migrator.drop_constraints(model, *constraints)

"""

from contextlib import suppress

import peewee as pw
from peewee_migrate import Migrator


with suppress(ImportError):
    import playhouse.postgres_ext as pw_pext


def migrate(migrator: Migrator, database: pw.Database, *, fake=False):
    """Add session foreign key to Trajectory table."""
    
    # Get the Session model from migrator.orm
    Session = migrator.orm['session']
    
    # Check if the column already exists
    try:
        database.execute_sql("SELECT session_id FROM trajectory LIMIT 1")
        # If we reach here, the column exists
        return
    except pw.OperationalError:
        # Column doesn't exist, safe to add
        pass
    
    # Add the session_id foreign key column
    migrator.add_fields(
        'trajectory', 
        session=pw.ForeignKeyField(
            Session, 
            null=True, 
            field='id',
            on_delete='CASCADE'
        )
    )


def rollback(migrator: Migrator, database: pw.Database, *, fake=False):
    """Remove session foreign key from Trajectory table."""
    
    migrator.remove_fields('trajectory', 'session')