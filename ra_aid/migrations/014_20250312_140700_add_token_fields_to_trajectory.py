"""Peewee migrations -- 014_20250312_140700_add_token_fields_to_trajectory.py.

This migration adds input_tokens and output_tokens fields to the Trajectory model.
These fields allow tracking token usage separately for input (prompt tokens) and output 
(completion tokens) to provide more detailed cost and usage analytics.

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
    """
    Add input_tokens and output_tokens fields to the Trajectory model.
    
    These fields provide a more detailed breakdown of token usage:
    - input_tokens: Number of tokens in the prompt/input (previously only tracked as total tokens)
    - output_tokens: Number of tokens in the completion/output (previously only tracked as total tokens)
    
    This allows for more accurate cost tracking and usage analytics since different models
    may charge different rates for input vs. output tokens.
    """
    
    # Check if the table exists before trying to modify it
    try:
        database.execute_sql("SELECT id FROM trajectory LIMIT 1")
    except pw.OperationalError:
        # Table doesn't exist, nothing to do
        return
    
    # Check if the columns already exist
    try:
        database.execute_sql("SELECT input_tokens, output_tokens FROM trajectory LIMIT 1")
        # If we reach here, the columns exist
        return
    except pw.OperationalError:
        # Columns don't exist, safe to add
        pass
    
    # Add the new fields to the trajectory table
    migrator.add_fields(
        'trajectory',
        input_tokens=pw.IntegerField(null=True),  # Track input/prompt tokens
        output_tokens=pw.IntegerField(null=True)  # Track output/completion tokens
    )


def rollback(migrator: Migrator, database: pw.Database, *, fake=False):
    """Remove the input_tokens and output_tokens fields from the Trajectory model."""
    
    migrator.remove_fields('trajectory', 'input_tokens', 'output_tokens')
