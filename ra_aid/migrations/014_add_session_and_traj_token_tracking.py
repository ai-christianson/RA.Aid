"""Peewee migrations -- 014_add_session_and_traj_token_tracking.py.

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
    
    # Check if the columns already exist in the session table
    try:
        # Try to execute a query that selects the columns
        database.execute_sql("SELECT total_output_tokens, total_tokens, total_input_tokens, total_cost FROM session LIMIT 1")
        print("Session token tracking columns already exist, skipping...")
    except Exception as e:
        # If the query fails, the columns don't exist, so add them
        print(f"Adding session token tracking columns: {e}")
        migrator.add_fields(
            'session',
            total_output_tokens=pw.IntegerField(default=0),
            total_tokens=pw.IntegerField(default=0),
            total_input_tokens=pw.IntegerField(default=0),
            total_cost=pw.FloatField(default=0.0))

    # Check if the session foreign key already exists in human_input
    try:
        database.execute_sql("SELECT session_id FROM human_input LIMIT 1")
        print("Human input session foreign key already exists, skipping...")
    except Exception as e:
        print(f"Adding human_input session foreign key: {e}")
        migrator.add_fields(
            'human_input',
            session=pw.ForeignKeyField(column_name='session_id', field='id', model=migrator.orm['session'], null=True))

    # Check if the foreign keys already exist in key_fact
    try:
        database.execute_sql("SELECT human_input_id, session_id FROM key_fact LIMIT 1")
        print("Key fact foreign keys already exist, skipping...")
    except Exception as e:
        print(f"Adding key_fact foreign keys: {e}")
        migrator.add_fields(
            'key_fact',
            human_input=pw.ForeignKeyField(column_name='human_input_id', field='id', model=migrator.orm['human_input'], null=True),
            session=pw.ForeignKeyField(column_name='session_id', field='id', model=migrator.orm['session'], null=True))

    # Check if the foreign keys already exist in key_snippet
    try:
        database.execute_sql("SELECT human_input_id, session_id FROM key_snippet LIMIT 1")
        print("Key snippet foreign keys already exist, skipping...")
    except Exception as e:
        print(f"Adding key_snippet foreign keys: {e}")
        migrator.add_fields(
            'key_snippet',
            human_input=pw.ForeignKeyField(column_name='human_input_id', field='id', model=migrator.orm['human_input'], null=True),
            session=pw.ForeignKeyField(column_name='session_id', field='id', model=migrator.orm['session'], null=True))

    # Check if basemodel table already exists
    try:
        database.execute_sql("SELECT id FROM basemodel LIMIT 1")
        print("BaseModel table already exists, skipping...")
    except Exception as e:
        print(f"Creating BaseModel table: {e}")
        @migrator.create_model
        class BaseModel(pw.Model):
            id = pw.AutoField()
            created_at = pw.DateTimeField()
            updated_at = pw.DateTimeField()

            class Meta:
                table_name = "basemodel"

    # Check if research_note table already exists
    try:
        database.execute_sql("SELECT id FROM research_note LIMIT 1")
        print("ResearchNote table already exists, skipping...")
    except Exception as e:
        print(f"Creating ResearchNote table: {e}")
        @migrator.create_model
        class ResearchNote(pw.Model):
            id = pw.AutoField()
            created_at = pw.DateTimeField()
            updated_at = pw.DateTimeField()
            content = pw.TextField()
            human_input = pw.ForeignKeyField(column_name='human_input_id', field='id', model=migrator.orm['human_input'], null=True)
            session = pw.ForeignKeyField(column_name='session_id', field='id', model=migrator.orm['session'], null=True)

            class Meta:
                table_name = "research_note"

    # Check if trajectory table already exists
    try:
        database.execute_sql("SELECT id FROM trajectory LIMIT 1")
        print("Trajectory table already exists, skipping...")
    except Exception as e:
        print(f"Creating Trajectory table: {e}")
        @migrator.create_model
        class Trajectory(pw.Model):
            id = pw.AutoField()
            created_at = pw.DateTimeField()
            updated_at = pw.DateTimeField()
            human_input = pw.ForeignKeyField(column_name='human_input_id', field='id', model=migrator.orm['human_input'], null=True)
            tool_name = pw.TextField(null=True)
            tool_parameters = pw.TextField(null=True)
            tool_result = pw.TextField(null=True)
            step_data = pw.TextField(null=True)
            record_type = pw.TextField(null=True)
            cost = pw.FloatField(null=True)
            tokens = pw.IntegerField(null=True)
            current_cost = pw.FloatField(null=True)
            current_tokens = pw.IntegerField(null=True)
            total_cost = pw.FloatField(null=True)
            total_tokens = pw.IntegerField(null=True)
            input_tokens = pw.IntegerField(null=True)
            output_tokens = pw.IntegerField(null=True)
            is_error = pw.BooleanField(default=False)
            error_message = pw.TextField(null=True)
            error_type = pw.TextField(null=True)
            error_details = pw.TextField(null=True)
            session = pw.ForeignKeyField(column_name='session_id', field='id', model=migrator.orm['session'], null=True)

            class Meta:
                table_name = "trajectory"


def rollback(migrator: Migrator, database: pw.Database, *, fake=False):
    """Write your rollback migrations here."""
    
    migrator.remove_fields('session', 'total_output_tokens', 'total_tokens', 'total_input_tokens', 'total_cost')

    migrator.remove_fields('key_snippet', 'human_input', 'session')

    migrator.remove_fields('key_fact', 'human_input', 'session')

    migrator.remove_fields('human_input', 'session')

    migrator.remove_model('trajectory')

    migrator.remove_model('research_note')

    migrator.remove_model('basemodel')
