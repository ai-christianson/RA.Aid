"""
Tests for the KeyFactRepository class.
"""

import pytest

from ra_aid.database.connection import DatabaseManager, db_var
from ra_aid.database.models import KeyFact
from ra_aid.database.repositories.key_fact_repository import KeyFactRepository


@pytest.fixture
def cleanup_db():
    """Reset the database contextvar and connection state after each test."""
    # Reset before the test
    db = db_var.get()
    if db is not None:
        try:
            if not db.is_closed():
                db.close()
        except Exception:
            # Ignore errors when closing the database
            pass
    db_var.set(None)

    # Run the test
    yield

    # Reset after the test
    db = db_var.get()
    if db is not None:
        try:
            if not db.is_closed():
                db.close()
        except Exception:
            # Ignore errors when closing the database
            pass
    db_var.set(None)


@pytest.fixture
def setup_db(cleanup_db):
    """Set up an in-memory database with the KeyFact table."""
    # Initialize an in-memory database connection
    with DatabaseManager(in_memory=True) as db:
        # Create the KeyFact table
        with db.atomic():
            db.create_tables([KeyFact], safe=True)
        
        yield db

        # Clean up
        with db.atomic():
            KeyFact.drop_table(safe=True)


def test_create_key_fact(setup_db):
    """Test creating a key fact."""
    # Set up repository
    repo = KeyFactRepository()
    
    # Create a key fact
    content = "Test key fact"
    fact = repo.create(content)
    
    # Verify the fact was created correctly
    assert fact.id is not None
    assert fact.content == content
    
    # Verify we can retrieve it from the database
    fact_from_db = KeyFact.get_by_id(fact.id)
    assert fact_from_db.content == content


def test_get_key_fact(setup_db):
    """Test retrieving a key fact by ID."""
    # Set up repository
    repo = KeyFactRepository()
    
    # Create a key fact
    content = "Test key fact"
    fact = repo.create(content)
    
    # Retrieve the fact by ID
    retrieved_fact = repo.get(fact.id)
    
    # Verify the retrieved fact matches the original
    assert retrieved_fact is not None
    assert retrieved_fact.id == fact.id
    assert retrieved_fact.content == content
    
    # Try to retrieve a non-existent fact
    non_existent_fact = repo.get(999)
    assert non_existent_fact is None


def test_update_key_fact(setup_db):
    """Test updating a key fact."""
    # Set up repository
    repo = KeyFactRepository()
    
    # Create a key fact
    original_content = "Original content"
    fact = repo.create(original_content)
    
    # Update the fact
    new_content = "Updated content"
    updated_fact = repo.update(fact.id, new_content)
    
    # Verify the fact was updated correctly
    assert updated_fact is not None
    assert updated_fact.id == fact.id
    assert updated_fact.content == new_content
    
    # Verify we can retrieve the updated content from the database
    fact_from_db = KeyFact.get_by_id(fact.id)
    assert fact_from_db.content == new_content
    
    # Try to update a non-existent fact
    non_existent_update = repo.update(999, "This shouldn't work")
    assert non_existent_update is None


def test_delete_key_fact(setup_db):
    """Test deleting a key fact."""
    # Set up repository
    repo = KeyFactRepository()
    
    # Create a key fact
    content = "Test key fact to delete"
    fact = repo.create(content)
    
    # Verify the fact exists
    assert KeyFact.get_or_none(KeyFact.id == fact.id) is not None
    
    # Delete the fact
    delete_result = repo.delete(fact.id)
    
    # Verify the delete operation was successful
    assert delete_result is True
    
    # Verify the fact no longer exists in the database
    assert KeyFact.get_or_none(KeyFact.id == fact.id) is None
    
    # Try to delete a non-existent fact
    non_existent_delete = repo.delete(999)
    assert non_existent_delete is False


def test_get_all_key_facts(setup_db):
    """Test retrieving all key facts."""
    # Set up repository
    repo = KeyFactRepository()
    
    # Create some key facts
    contents = ["Fact 1", "Fact 2", "Fact 3"]
    for content in contents:
        repo.create(content)
    
    # Retrieve all facts
    all_facts = repo.get_all()
    
    # Verify we got the correct number of facts
    assert len(all_facts) == len(contents)
    
    # Verify the content of each fact
    fact_contents = [fact.content for fact in all_facts]
    for content in contents:
        assert content in fact_contents


def test_get_facts_dict(setup_db):
    """Test retrieving key facts as a dictionary."""
    # Set up repository
    repo = KeyFactRepository()
    
    # Create some key facts
    facts = []
    contents = ["Fact 1", "Fact 2", "Fact 3"]
    for content in contents:
        facts.append(repo.create(content))
    
    # Retrieve facts as dictionary
    facts_dict = repo.get_facts_dict()
    
    # Verify we got the correct number of facts
    assert len(facts_dict) == len(contents)
    
    # Verify each fact is in the dictionary with the correct content
    for fact in facts:
        assert fact.id in facts_dict
        assert facts_dict[fact.id] == fact.content