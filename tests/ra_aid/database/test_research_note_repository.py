"""
Tests for the ResearchNoteRepository class.
"""

import pytest
from unittest.mock import patch

import peewee

from ra_aid.database.connection import DatabaseManager, db_var
from ra_aid.database.models import ResearchNote, BaseModel
from ra_aid.database.repositories.research_note_repository import (
    ResearchNoteRepository, 
    ResearchNoteRepositoryManager,
    get_research_note_repository,
    research_note_repo_var
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
    research_note_repo_var.set(None)
    
    # Run the test
    yield
    
    # Reset after the test
    research_note_repo_var.set(None)


@pytest.fixture
def setup_db(cleanup_db):
    """Set up an in-memory database with the ResearchNote table and patch the BaseModel.Meta.database."""
    # Initialize an in-memory database connection
    with DatabaseManager(in_memory=True) as db:
        # Patch the BaseModel.Meta.database to use our in-memory database
        # This ensures that model operations like ResearchNote.create() use our test database
        with patch.object(BaseModel._meta, 'database', db):
            # Create the ResearchNote table
            with db.atomic():
                db.create_tables([ResearchNote], safe=True)
            
            yield db
            
            # Clean up
            with db.atomic():
                ResearchNote.drop_table(safe=True)


def test_create_research_note(setup_db):
    """Test creating a research note."""
    # Set up repository
    repo = ResearchNoteRepository(db=setup_db)
    
    # Create a research note
    content = "Test research note"
    note = repo.create(content)
    
    # Verify the note was created correctly
    assert note.id is not None
    assert note.content == content
    
    # Verify we can retrieve it from the database using the repository
    note_from_db = repo.get(note.id)
    assert note_from_db.content == content


def test_get_research_note(setup_db):
    """Test retrieving a research note by ID."""
    # Set up repository
    repo = ResearchNoteRepository(db=setup_db)
    
    # Create a research note
    content = "Test research note"
    note = repo.create(content)
    
    # Retrieve the note by ID
    retrieved_note = repo.get(note.id)
    
    # Verify the retrieved note matches the original
    assert retrieved_note is not None
    assert retrieved_note.id == note.id
    assert retrieved_note.content == content
    
    # Try to retrieve a non-existent note
    non_existent_note = repo.get(999)
    assert non_existent_note is None


def test_update_research_note(setup_db):
    """Test updating a research note."""
    # Set up repository
    repo = ResearchNoteRepository(db=setup_db)
    
    # Create a research note
    original_content = "Original content"
    note = repo.create(original_content)
    
    # Update the note
    new_content = "Updated content"
    updated_note = repo.update(note.id, new_content)
    
    # Verify the note was updated correctly
    assert updated_note is not None
    assert updated_note.id == note.id
    assert updated_note.content == new_content
    
    # Verify we can retrieve the updated content from the database using the repository
    note_from_db = repo.get(note.id)
    assert note_from_db.content == new_content
    
    # Try to update a non-existent note
    non_existent_update = repo.update(999, "This shouldn't work")
    assert non_existent_update is None


def test_delete_research_note(setup_db):
    """Test deleting a research note."""
    # Set up repository
    repo = ResearchNoteRepository(db=setup_db)
    
    # Create a research note
    content = "Test research note to delete"
    note = repo.create(content)
    
    # Verify the note exists using the repository
    assert repo.get(note.id) is not None
    
    # Delete the note
    delete_result = repo.delete(note.id)
    
    # Verify the delete operation was successful
    assert delete_result is True
    
    # Verify the note no longer exists in the database using the repository
    assert repo.get(note.id) is None
    
    # Try to delete a non-existent note
    non_existent_delete = repo.delete(999)
    assert non_existent_delete is False


def test_get_all_research_notes(setup_db):
    """Test retrieving all research notes."""
    # Set up repository
    repo = ResearchNoteRepository(db=setup_db)
    
    # Create some research notes
    contents = ["Note 1", "Note 2", "Note 3"]
    for content in contents:
        repo.create(content)
    
    # Retrieve all notes
    all_notes = repo.get_all()
    
    # Verify we got the correct number of notes
    assert len(all_notes) == len(contents)
    
    # Verify the content of each note
    note_contents = [note.content for note in all_notes]
    for content in contents:
        assert content in note_contents


def test_get_notes_dict(setup_db):
    """Test retrieving research notes as a dictionary."""
    # Set up repository
    repo = ResearchNoteRepository(db=setup_db)
    
    # Create some research notes
    notes = []
    contents = ["Note 1", "Note 2", "Note 3"]
    for content in contents:
        notes.append(repo.create(content))
    
    # Retrieve notes as dictionary
    notes_dict = repo.get_notes_dict()
    
    # Verify we got the correct number of notes
    assert len(notes_dict) == len(contents)
    
    # Verify each note is in the dictionary with the correct content
    for note in notes:
        assert note.id in notes_dict
        assert notes_dict[note.id] == note.content


def test_repository_init_without_db():
    """Test that ResearchNoteRepository raises an error when initialized without a db parameter."""
    # Attempt to create a repository without a database connection
    with pytest.raises(ValueError) as excinfo:
        ResearchNoteRepository(db=None)
    
    # Verify the correct error message
    assert "Database connection is required" in str(excinfo.value)


def test_research_note_repository_manager(setup_db, cleanup_repo):
    """Test the ResearchNoteRepositoryManager context manager."""
    # Use the context manager to create a repository
    with ResearchNoteRepositoryManager(setup_db) as repo:
        # Verify the repository was created correctly
        assert isinstance(repo, ResearchNoteRepository)
        assert repo.db is setup_db
        
        # Verify we can use the repository
        content = "Test note via context manager"
        note = repo.create(content)
        assert note.id is not None
        assert note.content == content
        
        # Verify we can get the repository using get_research_note_repository
        repo_from_var = get_research_note_repository()
        assert repo_from_var is repo
    
    # Verify the repository was removed from the context var
    with pytest.raises(RuntimeError) as excinfo:
        get_research_note_repository()
    
    assert "No ResearchNoteRepository available" in str(excinfo.value)


def test_get_research_note_repository_when_not_set(cleanup_repo):
    """Test that get_research_note_repository raises an error when no repository is in context."""
    # Attempt to get the repository when none exists
    with pytest.raises(RuntimeError) as excinfo:
        get_research_note_repository()
    
    # Verify the correct error message
    assert "No ResearchNoteRepository available" in str(excinfo.value)