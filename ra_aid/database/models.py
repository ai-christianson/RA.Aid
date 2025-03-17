"""
Database models for ra_aid.

This module defines the base model class that all models will inherit from.
"""

import datetime
from typing import Any, Type, TypeVar

import peewee

from ra_aid.database.connection import get_db
from ra_aid.logging_config import get_logger

T = TypeVar("T", bound="BaseModel")
logger = get_logger(__name__)

# Create a database proxy that will be initialized later
database_proxy = peewee.DatabaseProxy()


def initialize_database():
    """
    Initialize the database proxy with a real database connection.

    This function should be called before any database operations
    to ensure the proxy points to a real database connection.

    Returns:
        peewee.SqliteDatabase: The initialized database connection
    """
    db = get_db()
    # Check if proxy is already initialized by checking the obj attribute directly
    if getattr(database_proxy, "obj", None) is None:
        logger.debug("Initializing database proxy")
        database_proxy.initialize(db)
    else:
        logger.debug("Database proxy already initialized")

    # Create tables if they don't exist yet
    # We need to import models here for table creation
    # to avoid circular imports
    # Note: This import needs to be here, not at the top level
    try:
        from ra_aid.database.models import (
            KeyFact,
            KeySnippet,
            HumanInput,
            ResearchNote,
            Trajectory,
            Session,
        )

        db.create_tables(
            [KeyFact, KeySnippet, HumanInput, ResearchNote, Trajectory, Session],
            safe=True,
        )
        logger.debug("Ensured database tables exist")
    except Exception as e:
        logger.error(f"Error creating tables: {str(e)}")

    return db


class BaseModel(peewee.Model):
    """
    Base model class for all ra_aid models.

    All models should inherit from this class to ensure consistent
    behavior and database connection.
    """

    created_at = peewee.DateTimeField(default=datetime.datetime.now)
    updated_at = peewee.DateTimeField(default=datetime.datetime.now)

    class Meta:
        database = database_proxy

    def save(self, *args: Any, **kwargs: Any) -> int:
        """
        Override save to update the updated_at field.

        Args:
            *args: Arguments to pass to the parent save method
            **kwargs: Keyword arguments to pass to the parent save method

        Returns:
            int: The primary key of the saved instance
        """
        self.updated_at = datetime.datetime.now()
        return super().save(*args, **kwargs)

    @classmethod
    def get_or_create(cls: Type[T], **kwargs: Any) -> tuple[T, bool]:
        """
        Get an instance or create it if it doesn't exist.

        Args:
            **kwargs: Fields to use for lookup and creation

        Returns:
            tuple: (instance, created) where created is a boolean indicating
                  whether a new instance was created
        """
        try:
            return super().get_or_create(**kwargs)
        except peewee.DatabaseError as e:
            # Log the error with logger
            logger.error(f"Failed in get_or_create: {str(e)}")
            raise


class Session(BaseModel):
    """
    Model representing a session stored in the database.

    Sessions track information about each program run, providing a way to group
    related records like human inputs, trajectories, and key facts.

    Each session record captures details about when the program was started,
    what command line arguments were used, and environment information.
    """

    start_time = peewee.DateTimeField(default=datetime.datetime.now)
    command_line = peewee.TextField(null=True)
    program_version = peewee.TextField(null=True)
    machine_info = peewee.TextField(
        null=True, help_text="JSON-encoded machine information"
    )

    class Meta:
        table_name = "session"


class HumanInput(BaseModel):
    """
    Model representing human input stored in the database.

    Human inputs are text inputs provided by users through various interfaces
    such as CLI, chat, or HIL (human-in-the-loop). This model tracks these inputs
    along with their source for analysis and reference.
    """

    content = peewee.TextField()
    source = peewee.TextField(help_text="'cli', 'chat', or 'hil'")
    session = peewee.ForeignKeyField(Session, backref="human_inputs", null=True)
    # created_at and updated_at are inherited from BaseModel

    class Meta:
        table_name = "human_input"


class KeyFact(BaseModel):
    """
    Model representing a key fact stored in the database.

    Key facts are important information about the project or current task
    that need to be referenced later.
    """

    content = peewee.TextField()
    human_input = peewee.ForeignKeyField(HumanInput, backref="key_facts", null=True)
    session = peewee.ForeignKeyField(Session, backref="key_facts", null=True)
    # created_at and updated_at are inherited from BaseModel

    class Meta:
        table_name = "key_fact"


class KeySnippet(BaseModel):
    """
    Model representing a key code snippet stored in the database.

    Key snippets are important code fragments from the project that need to be
    referenced later. Each snippet includes its file location, line number,
    the code content itself, and an optional description of its significance.
    """

    filepath = peewee.TextField()
    line_number = peewee.IntegerField()
    snippet = peewee.TextField()
    description = peewee.TextField(null=True)
    human_input = peewee.ForeignKeyField(HumanInput, backref="key_snippets", null=True)
    session = peewee.ForeignKeyField(Session, backref="key_snippets", null=True)
    # created_at and updated_at are inherited from BaseModel

    class Meta:
        table_name = "key_snippet"


class ResearchNote(BaseModel):
    """
    Model representing a research note stored in the database.

    Research notes are detailed information compiled from research activities
    that need to be preserved for future reference. These notes contain valuable
    context and findings about topics relevant to the project.
    """

    content = peewee.TextField()
    human_input = peewee.ForeignKeyField(
        HumanInput, backref="research_notes", null=True
    )
    session = peewee.ForeignKeyField(Session, backref="research_notes", null=True)
    # created_at and updated_at are inherited from BaseModel

    class Meta:
        table_name = "research_note"


class Trajectory(BaseModel):
    """
    Model representing an agent trajectory stored in the database.

    Trajectories track the sequence of actions taken by agents, including
    tool executions and their results. This enables analysis of agent behavior,
    debugging of issues, and reconstruction of the decision-making process.

    Each trajectory record captures details about a single tool execution:
    - Which tool was used
    - What parameters were passed to the tool
    - What result was returned by the tool
    - UI rendering data for displaying the tool execution
    - Cost and token usage metrics for tracking resource utilization
    - Detailed token usage breakdown (input_tokens and output_tokens)
    - Error information (when a tool execution fails)
    """

    human_input = peewee.ForeignKeyField(HumanInput, backref="trajectories", null=True)
    tool_name = peewee.TextField(null=True)
    tool_parameters = peewee.TextField(null=True, help_text="JSON-encoded parameters")
    tool_result = peewee.TextField(null=True, help_text="JSON-encoded result")
    step_data = peewee.TextField(null=True, help_text="JSON-encoded UI rendering data")
    record_type = peewee.TextField(null=True, help_text="Type of trajectory record")
    current_cost = peewee.FloatField(
        null=True, help_text="Cost of the last LLM message"
    )
    input_tokens = peewee.IntegerField(
        null=True, help_text="Prompt/input token usage for last message"
    )
    output_tokens = peewee.IntegerField(
        null=True, help_text="Completion/output token usage for last message"
    )
    is_error = peewee.BooleanField(
        default=False, help_text="Flag indicating if this record represents an error"
    )
    error_message = peewee.TextField(null=True, help_text="The error message")
    error_type = peewee.TextField(null=True, help_text="The type/class of the error")
    error_details = peewee.TextField(
        null=True, help_text="Additional error details like stack traces or context"
    )
    session = peewee.ForeignKeyField(Session, backref="trajectories", null=True)
    # created_at and updated_at are inherited from BaseModel


    class Meta:
        table_name = "trajectory"
