"""Peewee migrations -- 008_20250311_191232_add_session_model.py.

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
    """Create the session table for storing application session information."""
    
    table_exists = False
    # Check if the table already exists
    try:
        database.execute_sql("SELECT id FROM session LIMIT 1")
        # If we reach here, the table exists
        table_exists = True
    except pw.OperationalError:
        # Table doesn't exist, safe to create
        pass
    
    # Create the Session model - this registers it in migrator.orm as 'Session'
    @migrator.create_model
    class Session(pw.Model):
        id = pw.AutoField()
        created_at = pw.DateTimeField()
        updated_at = pw.DateTimeField()
        start_time = pw.DateTimeField()
        command_line = pw.TextField(null=True)
        program_version = pw.TextField(null=True)
        machine_info = pw.TextField(null=True)
        
        class Meta:
            table_name = "session"
    
    # FIX: Explicitly register the model under the lowercase table name key
    # This ensures that later migrations can access it via either:
    # - migrator.orm['Session'] (class name)
    # - migrator.orm['session'] (table name)
    if 'Session' in migrator.orm:
        migrator.orm['session'] = migrator.orm['Session']
    
    # Only return after model registration is complete
    if table_exists:
        return


def rollback(migrator: Migrator, database: pw.Database, *, fake=False):
    """Remove the session table."""
    
    migrator.remove_model('session')