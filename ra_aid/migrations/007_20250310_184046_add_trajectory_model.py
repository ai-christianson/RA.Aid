"""Peewee migrations -- 007_20250310_184046_add_trajectory_model.py.

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
    """Create the trajectory table for storing agent action trajectories."""
    
    # Check if the table already exists
    try:
        database.execute_sql("SELECT id FROM trajectory LIMIT 1")
        # If we reach here, the table exists
        return
    except pw.OperationalError:
        # Table doesn't exist, safe to create
        pass
    
    @migrator.create_model
    class Trajectory(pw.Model):
        id = pw.AutoField()
        created_at = pw.DateTimeField()
        updated_at = pw.DateTimeField()
        tool_name = pw.TextField(null=True)  # JSON-encoded parameters
        tool_parameters = pw.TextField(null=True)  # JSON-encoded parameters
        tool_result = pw.TextField(null=True)  # JSON-encoded result
        step_data = pw.TextField(null=True)  # JSON-encoded UI rendering data
        record_type = pw.TextField(null=True)  # Type of trajectory record
        cost = pw.FloatField(null=True)  # Placeholder for cost tracking
        tokens = pw.IntegerField(null=True)  # Placeholder for token usage tracking
        is_error = pw.BooleanField(default=False)  # Flag indicating if this record represents an error
        error_message = pw.TextField(null=True)  # The error message
        error_type = pw.TextField(null=True)  # The type/class of the error
        error_details = pw.TextField(null=True)  # Additional error details like stack traces or context
        # We'll add the human_input foreign key in a separate step for safety
        
        class Meta:
            table_name = "trajectory"
    
    # Check if HumanInput model exists before adding the foreign key
    try:
        HumanInput = migrator.orm['human_input']
        
        # Only add the foreign key if the human_input_id column doesn't already exist
        try:
            database.execute_sql("SELECT human_input_id FROM trajectory LIMIT 1")
        except pw.OperationalError:
            # Column doesn't exist, safe to add
            migrator.add_fields(
                'trajectory',
                human_input=pw.ForeignKeyField(
                    HumanInput,
                    null=True,
                    field='id',
                    on_delete='SET NULL'
                )
            )
    except KeyError:
        # HumanInput doesn't exist, we'll skip adding the foreign key
        pass
    

def rollback(migrator: Migrator, database: pw.Database, *, fake=False):
    """Remove the trajectory table."""
    
    # First remove any foreign key fields
    try:
        migrator.remove_fields('trajectory', 'human_input')
    except pw.OperationalError:
        # Field might not exist, that's fine
        pass
    
    # Then remove the model
    migrator.remove_model('trajectory')