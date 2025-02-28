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


class BaseModel(peewee.Model):
    """
    Base model class for all ra_aid models.

    All models should inherit from this class to ensure consistent
    behavior and database connection.
    """

    created_at = peewee.DateTimeField(default=datetime.datetime.now)
    updated_at = peewee.DateTimeField(default=datetime.datetime.now)

    class Meta:
        database = get_db()

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
