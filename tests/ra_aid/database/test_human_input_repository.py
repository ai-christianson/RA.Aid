"""
Tests for the human input repository.

This module provides tests for the HumanInputRepository class,
ensuring it correctly interfaces with the database and returns
appropriate Pydantic models.
"""

import unittest
from typing import List, Dict, Any

import pytest
from peewee import SqliteDatabase

from ra_aid.database.models import HumanInput, Session, database_proxy
from ra_aid.database.pydantic_models import HumanInputModel, SessionModel
from ra_aid.database.repositories.human_input_repository import HumanInputRepository
from ra_aid.database.repositories.session_repository import SessionRepository


@pytest.fixture
def test_db():
    """Fixture for creating a test database."""
    # Create an in-memory SQLite database for testing
    test_db = SqliteDatabase(':memory:')
    
    # Register the models with the test database
    with test_db.bind_ctx([HumanInput, Session]):
        # Create the tables
        test_db.create_tables([HumanInput, Session])
        
        # Return the test database for use in the tests
        yield test_db
        
        # Drop the tables after the tests
        test_db.drop_tables([HumanInput, Session])


class TestHumanInputRepository(unittest.TestCase):
    """Test case for the HumanInputRepository class."""
    
    def setUp(self):
        """Set up the test case with a test database and repositories."""
        # Create an in-memory database for testing
        self.db = SqliteDatabase(':memory:')
        
        # Register the models with the test database
        self.models = [HumanInput, Session]
        self.db.bind(self.models)
        
        # Create the tables
        self.db.create_tables(self.models)
        
        # Create repository instances for testing
        self.repository = HumanInputRepository(self.db)
        self.session_repository = SessionRepository(self.db)
        
        # Bind the test database to the repository model
        database_proxy.initialize(self.db)
    
    def tearDown(self):
        """Clean up after the test case."""
        # Close the database connection
        self.db.close()
    
    def test_create(self):
        """Test creating a human input record."""
        # Create a session first
        session_model = self.session_repository.create_session()
        
        # Create a human input
        content = "Test human input"
        source = "cli"
        human_input = self.repository.create(content=content, source=source)
        
        # Verify the human input was created
        self.assertIsInstance(human_input, HumanInputModel)
        self.assertEqual(human_input.content, content)
        self.assertEqual(human_input.source, source)
    
    def test_get(self):
        """Test retrieving a human input record by ID."""
        # Create a session first
        session_model = self.session_repository.create_session()
        
        # Create a human input
        content = "Test human input"
        source = "chat"
        created_input = self.repository.create(content=content, source=source)
        
        # Get the human input by ID
        retrieved_input = self.repository.get(created_input.id)
        
        # Verify the human input was retrieved correctly
        self.assertIsInstance(retrieved_input, HumanInputModel)
        self.assertEqual(retrieved_input.id, created_input.id)
        self.assertEqual(retrieved_input.content, content)
        self.assertEqual(retrieved_input.source, source)
    
    def test_update(self):
        """Test updating a human input record."""
        # Create a session first
        session_model = self.session_repository.create_session()
        
        # Create a human input
        content = "Original content"
        source = "cli"
        created_input = self.repository.create(content=content, source=source)
        
        # Update the human input
        new_content = "Updated content"
        updated_input = self.repository.update(created_input.id, content=new_content)
        
        # Verify the human input was updated correctly
        self.assertIsInstance(updated_input, HumanInputModel)
        self.assertEqual(updated_input.id, created_input.id)
        self.assertEqual(updated_input.content, new_content)
        self.assertEqual(updated_input.source, source)
    
    def test_get_all(self):
        """Test retrieving all human input records."""
        # Create a session first
        session_model = self.session_repository.create_session()
        
        # Create multiple human inputs
        self.repository.create(content="Input 1", source="cli")
        self.repository.create(content="Input 2", source="chat")
        self.repository.create(content="Input 3", source="hil")
        
        # Get all human inputs
        all_inputs = self.repository.get_all()
        
        # Verify all human inputs were retrieved
        self.assertEqual(len(all_inputs), 3)
        self.assertIsInstance(all_inputs[0], HumanInputModel)
        
        # Verify the inputs are ordered by created_at in descending order
        self.assertEqual(all_inputs[0].content, "Input 3")
        self.assertEqual(all_inputs[1].content, "Input 2")
        self.assertEqual(all_inputs[2].content, "Input 1")
    
    def test_get_recent(self):
        """Test retrieving the most recent human input records."""
        # Create a session first
        session_model = self.session_repository.create_session()
        
        # Create multiple human inputs
        self.repository.create(content="Input 1", source="cli")
        self.repository.create(content="Input 2", source="chat")
        self.repository.create(content="Input 3", source="hil")
        self.repository.create(content="Input 4", source="cli")
        self.repository.create(content="Input 5", source="chat")
        
        # Get recent human inputs with a limit of 3
        recent_inputs = self.repository.get_recent(limit=3)
        
        # Verify only the 3 most recent inputs were retrieved
        self.assertEqual(len(recent_inputs), 3)
        self.assertIsInstance(recent_inputs[0], HumanInputModel)
        self.assertEqual(recent_inputs[0].content, "Input 5")
        self.assertEqual(recent_inputs[1].content, "Input 4")
        self.assertEqual(recent_inputs[2].content, "Input 3")
    
    def test_get_by_source(self):
        """Test retrieving human input records by source."""
        # Create a session first
        session_model = self.session_repository.create_session()
        
        # Create human inputs with different sources
        self.repository.create(content="CLI Input 1", source="cli")
        self.repository.create(content="Chat Input 1", source="chat")
        self.repository.create(content="HIL Input", source="hil")
        self.repository.create(content="CLI Input 2", source="cli")
        self.repository.create(content="Chat Input 2", source="chat")
        
        # Get human inputs for the 'cli' source
        cli_inputs = self.repository.get_by_source("cli")
        
        # Verify only cli inputs were retrieved
        self.assertEqual(len(cli_inputs), 2)
        self.assertIsInstance(cli_inputs[0], HumanInputModel)
        self.assertEqual(cli_inputs[0].content, "CLI Input 2")
        self.assertEqual(cli_inputs[1].content, "CLI Input 1")
    
    def test_get_most_recent_id(self):
        """Test retrieving the ID of the most recent human input record."""
        # Create a session first
        session_model = self.session_repository.create_session()
        
        # Create multiple human inputs
        self.repository.create(content="Input 1", source="cli")
        input2 = self.repository.create(content="Input 2", source="chat")
        
        # Get the most recent ID
        most_recent_id = self.repository.get_most_recent_id()
        
        # Verify the correct ID was retrieved
        self.assertEqual(most_recent_id, input2.id)