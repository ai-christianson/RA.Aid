"""Peewee migrations -- 005_20250302_201611_add_human_input_reference.py.

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
    """Add human_input_id foreign key field to KeyFact and KeySnippet tables."""
    
    # Add human_input_id field to KeyFact model
    migrator.add_fields(
        'key_fact', 
        human_input_id=pw.ForeignKeyField(
            'human_input', 
            null=True, 
            field='id',
            on_delete='SET NULL'
        )
    )
    
    # Add human_input_id field to KeySnippet model
    migrator.add_fields(
        'key_snippet', 
        human_input_id=pw.ForeignKeyField(
            'human_input', 
            null=True, 
            field='id',
            on_delete='SET NULL'
        )
    )


def rollback(migrator: Migrator, database: pw.Database, *, fake=False):
    """Remove human_input_id field from KeyFact and KeySnippet tables."""
    
    migrator.remove_fields('key_fact', 'human_input_id')
    migrator.remove_fields('key_snippet', 'human_input_id')