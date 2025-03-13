"""Peewee migrations -- 017_remove_extra_cost_token_fields.py.

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
    """Write your migrations here."""
    
    # Check if the fields exist in the trajectory table before trying to remove them
    try:
        # Try to get the trajectory model from the ORM
        Trajectory = migrator.orm.get('trajectory', None)
        if Trajectory:
            # Check if the cost and tokens fields exist
            if hasattr(Trajectory, 'cost'):
                migrator.remove_fields('trajectory', 'cost')
            if hasattr(Trajectory, 'tokens'):
                migrator.remove_fields('trajectory', 'tokens')
    except Exception as e:
        # Log the error but continue with the migration
        print(f"Error checking trajectory fields: {e}")


def rollback(migrator: Migrator, database: pw.Database, *, fake=False):
    """Write your rollback migrations here."""
    
    # Add back the cost and tokens fields if needed
    try:
        migrator.add_fields(
            'trajectory',
            cost=pw.FloatField(null=True),
            tokens=pw.IntegerField(null=True)
        )
    except Exception as e:
        # Log the error but continue with the rollback
        print(f"Error adding back trajectory fields: {e}")
