"""
Tests for the KeyFactRepository class.
"""

import pytest
from unittest.mock import patch

import peewee

from ra_aid.database.connection import DatabaseManager, db_var
from ra_aid.database.models import KeyFact, BaseModel
from ra_aid.database.repositories.key_fact_repository import (
    KeyFactRepository, 
    KeyFactRepositoryManager,
    get_key_fact_repository,
    key_fact_repo_var
)


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
def cleanup_repo():
    """Reset the repository contextvar after each test."""
    # Reset before the test
    key_fact_repo_var.set(None)
    
    # Run the test
    yield
    
    # Reset after the test
    key_fact_repo_var.set(None)


@pytest.fixture
def setup_db(cleanup_db):
    """Set up an in-memory database with the KeyFact table and patch the BaseModel.Meta.database."""
    # Initialize an in-memory database connection
    with DatabaseManager(in_memory=True) as db:
        # Patch the BaseModel.Meta.database to use our in-memory database
        # This ensures that model operations like KeyFact.create() use our test database
        with patch.object(BaseModel._meta, 'database', db):
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
    repo = KeyFactRepository(db=setup_db)
    
    # Create a key fact
    content = "Test key fact"
    fact = repo.create(content)
    
    # Verify the fact was created correctly
    assert fact.id is not None
    assert fact.content == content
    
    # Verify we can retrieve it from the database using the repository
    fact_from_db = repo.get(fact.id)
    assert fact_from_db.content == content


def test_get_key_fact(setup_db):
    """Test retrieving a key fact by ID."""
    # Set up repository
    repo = KeyFactRepository(db=setup_db)
    
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
    repo = KeyFactRepository(db=setup_db)
    
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
    
    # Verify we can retrieve the updated content from the database using the repository
    fact_from_db = repo.get(fact.id)
    assert fact_from_db.content == new_content
    
    # Try to update a non-existent fact
    non_existent_update = repo.update(999, "This shouldn't work")
    assert non_existent_update is None


def test_delete_key_fact(setup_db):
    """Test deleting a key fact."""
    # Set up repository
    repo = KeyFactRepository(db=setup_db)
    
    # Create a key fact
    content = "Test key fact to delete"
    fact = repo.create(content)
    
    # Verify the fact exists using the repository
    assert repo.get(fact.id) is not None
    
    # Delete the fact
    delete_result = repo.delete(fact.id)
    
    # Verify the delete operation was successful
    assert delete_result is True
    
    # Verify the fact no longer exists in the database using the repository
    assert repo.get(fact.id) is None
    
    # Try to delete a non-existent fact
    non_existent_delete = repo.delete(999)
    assert non_existent_delete is False


def test_get_all_key_facts(setup_db):
    """Test retrieving all key facts."""
    # Set up repository
    repo = KeyFactRepository(db=setup_db)
    
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
    repo = KeyFactRepository(db=setup_db)
    
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


def test_repository_init_without_db():
    """Test that KeyFactRepository raises an error when initialized without a db parameter."""
    # Attempt to create a repository without a database connection
    with pytest.raises(ValueError) as excinfo:
        KeyFactRepository(db=None)
    
    # Verify the correct error message
    assert "Database connection is required" in str(excinfo.value)


def test_key_fact_repository_manager(setup_db, cleanup_repo):
    """Test the KeyFactRepositoryManager context manager."""
    # Use the context manager to create a repository
    with KeyFactRepositoryManager(setup_db) as repo:
        # Verify the repository was created correctly
        assert isinstance(repo, KeyFactRepository)
        assert repo.db is setup_db
        
        # Verify we can use the repository
        content = "Test fact via context manager"
        fact = repo.create(content)
        assert fact.id is not None
        assert fact.content == content
        
        # Verify we can get the repository using get_key_fact_repository
        repo_from_var = get_key_fact_repository()
        assert repo_from_var is repo
    
    # Verify the repository was removed from the context var
    with pytest.raises(RuntimeError) as excinfo:
        get_key_fact_repository()
    
    assert "No KeyFactRepository available" in str(excinfo.value)


def test_get_key_fact_repository_when_not_set(cleanup_repo):
    """Test that get_key_fact_repository raises an error when no repository is in context."""
    # Attempt to get the repository when none exists
    with pytest.raises(RuntimeError) as excinfo:
        get_key_fact_repository()
    
    # Verify the correct error message
    assert "No KeyFactRepository available" in str(excinfo.value)