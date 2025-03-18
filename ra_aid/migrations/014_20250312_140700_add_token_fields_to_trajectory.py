"""Peewee migrations -- 014_20250312_140700_add_token_fields_to_trajectory.py.

This migration adds input_tokens, output_tokens, and current_cost fields to the Trajectory model.
These fields allow tracking token usage separately for input (prompt tokens) and output 
(completion tokens) to provide more detailed cost and usage analytics.

This migration also removes the legacy cost and tokens fields that are no longer needed.

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


# Define the Trajectory model class to represent the database schema after migration
class TrajectoryModel(pw.Model):
    id = pw.AutoField()
    created_at = pw.DateTimeField()
    updated_at = pw.DateTimeField()
    tool_name = pw.TextField(null=True)  # JSON-encoded parameters
    tool_parameters = pw.TextField(null=True)  # JSON-encoded parameters
    tool_result = pw.TextField(null=True)  # JSON-encoded result
    step_data = pw.TextField(null=True)  # JSON-encoded UI rendering data
    record_type = pw.TextField(null=True)  # Type of trajectory record
    # New fields for detailed token and cost tracking
    input_tokens = pw.IntegerField(null=True)  # Track input/prompt tokens
    output_tokens = pw.IntegerField(null=True)  # Track output/completion tokens
    current_cost = pw.FloatField(null=True)  # Cost of the current operation
    # Error tracking fields
    is_error = pw.BooleanField(default=False)  # Flag indicating if this record represents an error
    error_message = pw.TextField(null=True)  # The error message
    error_type = pw.TextField(null=True)  # The type/class of the error
    error_details = pw.TextField(null=True)  # Additional error details like stack traces or context
    # Foreign keys are handled separately by the migrator
    
    class Meta:
        table_name = "trajectory"


def migrate(migrator: Migrator, database: pw.Database, *, fake=False):
    """
    Add input_tokens, output_tokens, and current_cost fields to the Trajectory model.
    Remove legacy cost and tokens fields.
    
    These fields provide a more detailed breakdown of token usage:
    - input_tokens: Number of tokens in the prompt/input
    - output_tokens: Number of tokens in the completion/output
    - current_cost: Cost of the current operation
    
    This allows for more accurate cost tracking and usage analytics since different models
    may charge different rates for input vs. output tokens.
    """
    
    # Check if the table exists before trying to modify it
    try:
        database.execute_sql("SELECT id FROM trajectory LIMIT 1")
    except pw.OperationalError:
        # Table doesn't exist, nothing to do
        return
    
    # Add the new fields to the trajectory table if they don't exist
    try:
        database.execute_sql("SELECT input_tokens FROM trajectory LIMIT 1")
    except pw.OperationalError:
        migrator.add_fields(
            TrajectoryModel,
            input_tokens=pw.IntegerField(null=True),  # Track input/prompt tokens
        )
    
    try:
        database.execute_sql("SELECT output_tokens FROM trajectory LIMIT 1")
    except pw.OperationalError:
        migrator.add_fields(
            TrajectoryModel,
            output_tokens=pw.IntegerField(null=True),  # Track output/completion tokens
        )
    
    try:
        database.execute_sql("SELECT current_cost FROM trajectory LIMIT 1")
    except pw.OperationalError:
        migrator.add_fields(
            TrajectoryModel,
            current_cost=pw.FloatField(null=True),  # Cost of the current operation
        )
    
    # Check if the table exists before trying to remove fields
    try:
        database.execute_sql("SELECT id FROM trajectory LIMIT 1")
        # Only attempt to remove fields if the table exists
        try:
            migrator.remove_fields(TrajectoryModel, 'cost')
        except Exception as e:
            # print(f"Error removing cost field: {e}")
            pass
        
        try:
            migrator.remove_fields(TrajectoryModel, 'tokens')
        except Exception as e:
            # print(f"Error removing tokens field: {e}")
            pass
            
    except pw.OperationalError:
        # Table doesn't exist, nothing to remove
        # print("Trajectory table doesn't exist, skipping field removal")
        pass


def rollback(migrator: Migrator, database: pw.Database, *, fake=False):
    """Remove the new fields and restore the legacy fields."""
    
    migrator.remove_fields(TrajectoryModel, 'input_tokens', 'output_tokens', 'current_cost')
    
    migrator.add_fields(
        TrajectoryModel,
        cost=pw.FloatField(null=True),
        tokens=pw.IntegerField(null=True)
    )
