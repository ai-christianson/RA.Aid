"""
Tests for the KeySnippetRepository class.
"""

import pytest

from ra_aid.database.connection import DatabaseManager, db_var
from ra_aid.database.models import KeySnippet
from ra_aid.database.repositories.key_snippet_repository import KeySnippetRepository


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
    """Set up an in-memory database with the KeySnippet table."""
    # Initialize an in-memory database connection
    with DatabaseManager(in_memory=True) as db:
        # Create the KeySnippet table
        with db.atomic():
            db.create_tables([KeySnippet], safe=True)
        
        yield db

        # Clean up
        with db.atomic():
            KeySnippet.drop_table(safe=True)


def test_create_key_snippet(setup_db):
    """Test creating a key snippet."""
    # Set up repository with the in-memory database
    repo = KeySnippetRepository(db=setup_db)
    
    # Create a key snippet
    filepath = "test_file.py"
    line_number = 42
    snippet = "def test_function():"
    description = "Test function definition"
    
    key_snippet = repo.create(
        filepath=filepath,
        line_number=line_number,
        snippet=snippet,
        description=description
    )
    
    # Verify the snippet was created correctly
    assert key_snippet.id is not None
    assert key_snippet.filepath == filepath
    assert key_snippet.line_number == line_number
    assert key_snippet.snippet == snippet
    assert key_snippet.description == description
    
    # Verify we can retrieve it from the database
    snippet_from_db = KeySnippet.get_by_id(key_snippet.id)
    assert snippet_from_db.filepath == filepath
    assert snippet_from_db.line_number == line_number
    assert snippet_from_db.snippet == snippet
    assert snippet_from_db.description == description


def test_get_key_snippet(setup_db):
    """Test retrieving a key snippet by ID."""
    # Set up repository with the in-memory database
    repo = KeySnippetRepository(db=setup_db)
    
    # Create a key snippet
    filepath = "test_file.py"
    line_number = 42
    snippet = "def test_function():"
    description = "Test function definition"
    
    key_snippet = repo.create(
        filepath=filepath,
        line_number=line_number,
        snippet=snippet,
        description=description
    )
    
    # Retrieve the snippet by ID
    retrieved_snippet = repo.get(key_snippet.id)
    
    # Verify the retrieved snippet matches the original
    assert retrieved_snippet is not None
    assert retrieved_snippet.id == key_snippet.id
    assert retrieved_snippet.filepath == filepath
    assert retrieved_snippet.line_number == line_number
    assert retrieved_snippet.snippet == snippet
    assert retrieved_snippet.description == description
    
    # Try to retrieve a non-existent snippet
    non_existent_snippet = repo.get(999)
    assert non_existent_snippet is None


def test_update_key_snippet(setup_db):
    """Test updating a key snippet."""
    # Set up repository with the in-memory database
    repo = KeySnippetRepository(db=setup_db)
    
    # Create a key snippet
    original_filepath = "original_file.py"
    original_line_number = 10
    original_snippet = "def original_function():"
    original_description = "Original function definition"
    
    key_snippet = repo.create(
        filepath=original_filepath,
        line_number=original_line_number,
        snippet=original_snippet,
        description=original_description
    )
    
    # Update the snippet
    new_filepath = "updated_file.py"
    new_line_number = 20
    new_snippet = "def updated_function():"
    new_description = "Updated function definition"
    
    updated_snippet = repo.update(
        key_snippet.id,
        filepath=new_filepath,
        line_number=new_line_number,
        snippet=new_snippet,
        description=new_description
    )
    
    # Verify the snippet was updated correctly
    assert updated_snippet is not None
    assert updated_snippet.id == key_snippet.id
    assert updated_snippet.filepath == new_filepath
    assert updated_snippet.line_number == new_line_number
    assert updated_snippet.snippet == new_snippet
    assert updated_snippet.description == new_description
    
    # Verify we can retrieve the updated content from the database
    snippet_from_db = KeySnippet.get_by_id(key_snippet.id)
    assert snippet_from_db.filepath == new_filepath
    assert snippet_from_db.line_number == new_line_number
    assert snippet_from_db.snippet == new_snippet
    assert snippet_from_db.description == new_description
    
    # Try to update a non-existent snippet
    non_existent_update = repo.update(
        999, 
        filepath="nonexistent.py",
        line_number=999,
        snippet="This shouldn't work",
        description="This shouldn't work"
    )
    assert non_existent_update is None


def test_delete_key_snippet(setup_db):
    """Test deleting a key snippet."""
    # Set up repository with the in-memory database
    repo = KeySnippetRepository(db=setup_db)
    
    # Create a key snippet
    filepath = "file_to_delete.py"
    line_number = 30
    snippet = "def function_to_delete():"
    description = "Function to delete"
    
    key_snippet = repo.create(
        filepath=filepath,
        line_number=line_number,
        snippet=snippet,
        description=description
    )
    
    # Verify the snippet exists
    assert KeySnippet.get_or_none(KeySnippet.id == key_snippet.id) is not None
    
    # Delete the snippet
    delete_result = repo.delete(key_snippet.id)
    
    # Verify the delete operation was successful
    assert delete_result is True
    
    # Verify the snippet no longer exists in the database
    assert KeySnippet.get_or_none(KeySnippet.id == key_snippet.id) is None
    
    # Try to delete a non-existent snippet
    non_existent_delete = repo.delete(999)
    assert non_existent_delete is False


def test_get_all_key_snippets(setup_db):
    """Test retrieving all key snippets."""
    # Set up repository with the in-memory database
    repo = KeySnippetRepository(db=setup_db)
    
    # Create some key snippets
    snippets_data = [
        {
            "filepath": "file1.py",
            "line_number": 10,
            "snippet": "def function1():",
            "description": "Function 1"
        },
        {
            "filepath": "file2.py",
            "line_number": 20,
            "snippet": "def function2():",
            "description": "Function 2"
        },
        {
            "filepath": "file3.py",
            "line_number": 30,
            "snippet": "def function3():",
            "description": "Function 3"
        }
    ]
    
    for data in snippets_data:
        repo.create(**data)
    
    # Retrieve all snippets
    all_snippets = repo.get_all()
    
    # Verify we got the correct number of snippets
    assert len(all_snippets) == len(snippets_data)
    
    # Verify the content of each snippet
    for i, snippet in enumerate(all_snippets):
        assert snippet.filepath == snippets_data[i]["filepath"]
        assert snippet.line_number == snippets_data[i]["line_number"]
        assert snippet.snippet == snippets_data[i]["snippet"]
        assert snippet.description == snippets_data[i]["description"]


def test_get_snippets_dict(setup_db):
    """Test retrieving key snippets as a dictionary."""
    # Set up repository with the in-memory database
    repo = KeySnippetRepository(db=setup_db)
    
    # Create some key snippets
    snippets = []
    snippets_data = [
        {
            "filepath": "file1.py",
            "line_number": 10,
            "snippet": "def function1():",
            "description": "Function 1"
        },
        {
            "filepath": "file2.py",
            "line_number": 20,
            "snippet": "def function2():",
            "description": "Function 2"
        },
        {
            "filepath": "file3.py",
            "line_number": 30,
            "snippet": "def function3():",
            "description": "Function 3"
        }
    ]
    
    for data in snippets_data:
        snippets.append(repo.create(**data))
    
    # Retrieve snippets as dictionary
    snippets_dict = repo.get_snippets_dict()
    
    # Verify we got the correct number of snippets
    assert len(snippets_dict) == len(snippets_data)
    
    # Verify each snippet is in the dictionary with the correct content
    for i, snippet in enumerate(snippets):
        assert snippet.id in snippets_dict
        assert snippets_dict[snippet.id]["filepath"] == snippets_data[i]["filepath"]
        assert snippets_dict[snippet.id]["line_number"] == snippets_data[i]["line_number"]
        assert snippets_dict[snippet.id]["snippet"] == snippets_data[i]["snippet"]
        assert snippets_dict[snippet.id]["description"] == snippets_data[i]["description"]