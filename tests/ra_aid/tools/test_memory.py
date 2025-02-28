import pytest

from ra_aid.tools.memory import (
    _global_memory,
    delete_key_facts,
    delete_key_snippets,
    delete_tasks,
    deregister_related_files,
    emit_key_facts,
    emit_key_snippet,
    emit_related_files,
    emit_task,
    get_memory_value,
    get_related_files,
    get_work_log,
    log_work_event,
    reset_work_log,
    swap_task_order,
)


@pytest.fixture
def reset_memory():
    """Reset global memory before each test"""
    _global_memory["key_facts"] = {}
    _global_memory["key_fact_id_counter"] = 0
    _global_memory["key_snippets"] = {}
    _global_memory["key_snippet_id_counter"] = 0
    _global_memory["research_notes"] = []
    _global_memory["plans"] = []
    _global_memory["tasks"] = {}
    _global_memory["task_id_counter"] = 0
    _global_memory["related_files"] = {}
    _global_memory["related_file_id_counter"] = 0
    _global_memory["work_log"] = []
    yield
    # Clean up after test
    _global_memory["key_facts"] = {}
    _global_memory["key_fact_id_counter"] = 0
    _global_memory["key_snippets"] = {}
    _global_memory["key_snippet_id_counter"] = 0
    _global_memory["research_notes"] = []
    _global_memory["plans"] = []
    _global_memory["tasks"] = {}
    _global_memory["task_id_counter"] = 0
    _global_memory["related_files"] = {}
    _global_memory["related_file_id_counter"] = 0
    _global_memory["work_log"] = []


def test_emit_key_facts_single_fact(reset_memory):
    """Test emitting a single key fact using emit_key_facts"""
    # Test with single fact
    result = emit_key_facts.invoke({"facts": ["First fact"]})
    assert result == "Facts stored."
    assert _global_memory["key_facts"][0] == "First fact"
    assert _global_memory["key_fact_id_counter"] == 1


def test_delete_key_facts_single_fact(reset_memory):
    """Test deleting a single key fact using delete_key_facts"""
    # Add a fact
    emit_key_facts.invoke({"facts": ["Test fact"]})

    # Delete the fact
    result = delete_key_facts.invoke({"fact_ids": [0]})
    assert result == "Facts deleted."
    assert 0 not in _global_memory["key_facts"]


def test_delete_key_facts_invalid(reset_memory):
    """Test deleting non-existent facts returns empty list"""
    # Try to delete non-existent fact
    result = delete_key_facts.invoke({"fact_ids": [999]})
    assert result == "Facts deleted."

    # Add and delete a fact, then try to delete it again
    emit_key_facts.invoke({"facts": ["Test fact"]})
    delete_key_facts.invoke({"fact_ids": [0]})
    result = delete_key_facts.invoke({"fact_ids": [0]})
    assert result == "Facts deleted."


def test_get_memory_value_key_facts(reset_memory):
    """Test get_memory_value with key facts dictionary"""
    # Empty key facts should return empty string
    assert get_memory_value("key_facts") == ""

    # Add some facts
    emit_key_facts.invoke({"facts": ["First fact", "Second fact"]})

    # Should return markdown formatted list
    expected = "## 🔑 Key Fact #0\n\nFirst fact\n\n## 🔑 Key Fact #1\n\nSecond fact"
    assert get_memory_value("key_facts") == expected


def test_get_memory_value_other_types(reset_memory):
    """Test get_memory_value remains compatible with other memory types"""
    # Add some research notes
    _global_memory["research_notes"].append("Note 1")
    _global_memory["research_notes"].append("Note 2")

    assert get_memory_value("research_notes") == "Note 1\nNote 2"

    # Test with empty list
    assert get_memory_value("plans") == ""

    # Test with non-existent key
    assert get_memory_value("nonexistent") == ""


def test_log_work_event(reset_memory):
    """Test logging work events with timestamps"""
    # Log some events
    log_work_event("Started task")
    log_work_event("Made progress")
    log_work_event("Completed task")

    # Verify events are stored
    assert len(_global_memory["work_log"]) == 3

    # Check event structure
    event = _global_memory["work_log"][0]
    assert isinstance(event["timestamp"], str)
    assert event["event"] == "Started task"

    # Verify order
    assert _global_memory["work_log"][1]["event"] == "Made progress"
    assert _global_memory["work_log"][2]["event"] == "Completed task"


def test_get_work_log(reset_memory):
    """Test work log formatting with heading-based markdown"""
    # Test empty log
    assert get_work_log() == "No work log entries"

    # Add some events
    log_work_event("First event")
    log_work_event("Second event")

    # Get formatted log
    log = get_work_log()

    assert "First event" in log
    assert "Second event" in log


def test_reset_work_log(reset_memory):
    """Test resetting the work log"""
    # Add some events
    log_work_event("Test event")
    assert len(_global_memory["work_log"]) == 1

    # Reset log
    reset_work_log()

    # Verify log is empty
    assert len(_global_memory["work_log"]) == 0
    assert get_memory_value("work_log") == ""


def test_empty_work_log(reset_memory):
    """Test empty work log behavior"""
    # Fresh work log should return empty string
    assert get_memory_value("work_log") == ""


def test_emit_key_facts(reset_memory):
    """Test emitting multiple key facts at once"""
    # Test emitting multiple facts
    facts = ["First fact", "Second fact", "Third fact"]
    result = emit_key_facts.invoke({"facts": facts})

    # Verify return message
    assert result == "Facts stored."

    # Verify facts stored in memory with correct IDs
    assert _global_memory["key_facts"][0] == "First fact"
    assert _global_memory["key_facts"][1] == "Second fact"
    assert _global_memory["key_facts"][2] == "Third fact"

    # Verify counter incremented correctly
    assert _global_memory["key_fact_id_counter"] == 3


def test_delete_key_facts(reset_memory):
    """Test deleting multiple key facts"""
    # Add some test facts
    emit_key_facts.invoke({"facts": ["First fact", "Second fact", "Third fact"]})

    # Test deleting mix of existing and non-existing IDs
    result = delete_key_facts.invoke({"fact_ids": [0, 1, 999]})

    # Verify success message
    assert result == "Facts deleted."

    # Verify correct facts removed from memory
    assert 0 not in _global_memory["key_facts"]
    assert 1 not in _global_memory["key_facts"]
    assert 2 in _global_memory["key_facts"]  # ID 2 should remain
    assert _global_memory["key_facts"][2] == "Third fact"


def test_emit_key_snippet(reset_memory):
    """Test emitting a single code snippet"""
    # Test snippet with description
    snippet = {
        "filepath": "test.py",
        "line_number": 10,
        "snippet": "def test():\n    pass",
        "description": "Test function",
    }

    # Emit snippet
    result = emit_key_snippet.invoke({"snippet_info": snippet})

    # Verify return message
    assert result == "Snippet #0 stored."

    # Verify snippet stored correctly
    assert _global_memory["key_snippets"][0] == snippet

    # Verify counter incremented correctly
    assert _global_memory["key_snippet_id_counter"] == 1

    # Test snippet without description
    snippet2 = {
        "filepath": "main.py",
        "line_number": 20,
        "snippet": "print('hello')",
        "description": None,
    }

    # Emit second snippet
    result = emit_key_snippet.invoke({"snippet_info": snippet2})

    # Verify return message
    assert result == "Snippet #1 stored."

    # Verify snippet stored correctly
    assert _global_memory["key_snippets"][1] == snippet2

    # Verify counter incremented correctly
    assert _global_memory["key_snippet_id_counter"] == 2


def test_delete_key_snippets(reset_memory):
    """Test deleting multiple code snippets"""
    # Add test snippets
    snippets = [
        {
            "filepath": "test1.py",
            "line_number": 1,
            "snippet": "code1",
            "description": None,
        },
        {
            "filepath": "test2.py",
            "line_number": 2,
            "snippet": "code2",
            "description": None,
        },
        {
            "filepath": "test3.py",
            "line_number": 3,
            "snippet": "code3",
            "description": None,
        },
    ]
    # Add snippets one by one
    for snippet in snippets:
        emit_key_snippet.invoke({"snippet_info": snippet})

    # Test deleting mix of valid and invalid IDs
    result = delete_key_snippets.invoke({"snippet_ids": [0, 1, 999]})

    # Verify success message
    assert result == "Snippets deleted."

    # Verify correct snippets removed
    assert 0 not in _global_memory["key_snippets"]
    assert 1 not in _global_memory["key_snippets"]
    assert 2 in _global_memory["key_snippets"]
    assert _global_memory["key_snippets"][2]["filepath"] == "test3.py"


def test_delete_key_snippets_empty(reset_memory):
    """Test deleting snippets with empty ID list"""
    # Add a test snippet
    snippet = {
        "filepath": "test.py",
        "line_number": 1,
        "snippet": "code",
        "description": None,
    }
    emit_key_snippet.invoke({"snippet_info": snippet})

    # Test with empty list
    result = delete_key_snippets.invoke({"snippet_ids": []})
    assert result == "Snippets deleted."

    # Verify snippet still exists
    assert 0 in _global_memory["key_snippets"]


def test_emit_related_files_basic(reset_memory, tmp_path):
    """Test basic adding of files with ID tracking"""
    # Create test files
    test_file = tmp_path / "test.py"
    test_file.write_text("# Test file")
    main_file = tmp_path / "main.py"
    main_file.write_text("# Main file")
    utils_file = tmp_path / "utils.py"
    utils_file.write_text("# Utils file")

    # Test adding single file
    result = emit_related_files.invoke({"files": [str(test_file)]})
    assert result == "Files noted."
    assert _global_memory["related_files"][0] == str(test_file)

    # Test adding multiple files
    result = emit_related_files.invoke({"files": [str(main_file), str(utils_file)]})
    assert result == "Files noted."
    # Verify both files exist in related_files
    values = list(_global_memory["related_files"].values())
    assert str(main_file) in values
    assert str(utils_file) in values


def test_get_related_files_empty(reset_memory):
    """Test getting related files when none added"""
    assert get_related_files() == []


def test_emit_related_files_duplicates(reset_memory, tmp_path):
    """Test that duplicate files return existing IDs with proper formatting"""
    # Create test files
    test_file = tmp_path / "test.py"
    test_file.write_text("# Test file")
    main_file = tmp_path / "main.py"
    main_file.write_text("# Main file")
    new_file = tmp_path / "new.py"
    new_file.write_text("# New file")

    # Add initial files
    result = emit_related_files.invoke({"files": [str(test_file), str(main_file)]})
    assert result == "Files noted."
    _first_id = 0  # ID of test.py

    # Try adding duplicates
    result = emit_related_files.invoke({"files": [str(test_file)]})
    assert result == "Files noted."
    assert len(_global_memory["related_files"]) == 2  # Count should not increase

    # Try mix of new and duplicate files
    result = emit_related_files.invoke({"files": [str(test_file), str(new_file)]})
    assert result == "Files noted."
    assert len(_global_memory["related_files"]) == 3


def test_related_files_id_tracking(reset_memory, tmp_path):
    """Test ID assignment and counter functionality for related files"""
    # Create test files
    file1 = tmp_path / "file1.py"
    file1.write_text("# File 1")
    file2 = tmp_path / "file2.py"
    file2.write_text("# File 2")

    # Add first file
    result = emit_related_files.invoke({"files": [str(file1)]})
    assert result == "Files noted."
    assert _global_memory["related_file_id_counter"] == 1

    # Add second file
    result = emit_related_files.invoke({"files": [str(file2)]})
    assert result == "Files noted."
    assert _global_memory["related_file_id_counter"] == 2

    # Verify all files stored correctly
    assert _global_memory["related_files"][0] == str(file1)
    assert _global_memory["related_files"][1] == str(file2)


def test_deregister_related_files(reset_memory, tmp_path):
    """Test deleting related files"""
    # Create test files
    file1 = tmp_path / "file1.py"
    file1.write_text("# File 1")
    file2 = tmp_path / "file2.py"
    file2.write_text("# File 2")
    file3 = tmp_path / "file3.py"
    file3.write_text("# File 3")

    # Add test files
    emit_related_files.invoke({"files": [str(file1), str(file2), str(file3)]})

    # Delete middle file
    result = deregister_related_files.invoke({"file_ids": [1]})
    assert result == "Files noted."
    assert 1 not in _global_memory["related_files"]
    assert len(_global_memory["related_files"]) == 2

    # Delete multiple files including non-existent ID
    result = deregister_related_files.invoke({"file_ids": [0, 2, 999]})
    assert result == "Files noted."
    assert len(_global_memory["related_files"]) == 0

    # Counter should remain unchanged after deletions
    assert _global_memory["related_file_id_counter"] == 3


def test_related_files_duplicates(reset_memory, tmp_path):
    """Test duplicate file handling returns same ID"""
    # Create test file
    test_file = tmp_path / "test.py"
    test_file.write_text("# Test file")

    # Add initial file
    result1 = emit_related_files.invoke({"files": [str(test_file)]})
    assert result1 == "Files noted."

    # Add same file again
    result2 = emit_related_files.invoke({"files": [str(test_file)]})
    assert result2 == "Files noted."

    # Verify only one entry exists
    assert len(_global_memory["related_files"]) == 1
    assert _global_memory["related_file_id_counter"] == 1


def test_emit_related_files_with_directory(reset_memory, tmp_path):
    """Test that directories and non-existent paths are rejected while valid files are added"""
    # Create test directory and file
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    test_file = tmp_path / "test_file.txt"
    test_file.write_text("test content")
    nonexistent = tmp_path / "does_not_exist.txt"

    # Try to emit directory, nonexistent path, and valid file
    result = emit_related_files.invoke(
        {"files": [str(test_dir), str(nonexistent), str(test_file)]}
    )

    # Verify result is the standard message
    assert result == "Files noted."

    # Verify directory and nonexistent not added but valid file was
    assert len(_global_memory["related_files"]) == 1
    values = list(_global_memory["related_files"].values())
    assert str(test_file) in values
    assert str(test_dir) not in values
    assert str(nonexistent) not in values


def test_related_files_formatting(reset_memory, tmp_path):
    """Test related files output string formatting"""
    # Create test files
    file1 = tmp_path / "file1.py"
    file1.write_text("# File 1")
    file2 = tmp_path / "file2.py"
    file2.write_text("# File 2")

    # Add some files
    emit_related_files.invoke({"files": [str(file1), str(file2)]})

    # Get formatted output
    output = get_memory_value("related_files")
    # Expect just the IDs on separate lines
    expected = "0\n1"
    assert output == expected

    # Test empty case
    _global_memory["related_files"] = {}
    assert get_memory_value("related_files") == ""


def test_emit_related_files_path_normalization(reset_memory, tmp_path):
    """Test that emit_related_files fails to detect duplicates with non-normalized paths"""
    # Create a test file
    test_file = tmp_path / "file.txt"
    test_file.write_text("test content")

    # Change to the temp directory so relative paths work
    import os

    original_dir = os.getcwd()
    os.chdir(tmp_path)

    try:
        # Add file with absolute path
        result1 = emit_related_files.invoke({"files": ["file.txt"]})
        assert result1 == "Files noted."

        # Add same file with relative path - should get same ID due to path normalization
        result2 = emit_related_files.invoke({"files": ["./file.txt"]})
        assert result2 == "Files noted."

        # Verify only one normalized path entry exists
        assert len(_global_memory["related_files"]) == 1
        assert os.path.abspath("file.txt") in _global_memory["related_files"].values()
    finally:
        # Restore original directory
        os.chdir(original_dir)


def test_key_snippets_integration(reset_memory, tmp_path):
    """Integration test for key snippets functionality"""
    # Create test files
    file1 = tmp_path / "file1.py"
    file1.write_text("def func1():\n    pass")
    file2 = tmp_path / "file2.py"
    file2.write_text("def func2():\n    return True")
    file3 = tmp_path / "file3.py"
    file3.write_text("class TestClass:\n    pass")

    # Initial snippets to add
    snippets = [
        {
            "filepath": str(file1),
            "line_number": 10,
            "snippet": "def func1():\n    pass",
            "description": "First function",
        },
        {
            "filepath": str(file2),
            "line_number": 20,
            "snippet": "def func2():\n    return True",
            "description": "Second function",
        },
        {
            "filepath": str(file3),
            "line_number": 30,
            "snippet": "class TestClass:\n    pass",
            "description": "Test class",
        },
    ]

    # Add all snippets one by one
    for i, snippet in enumerate(snippets):
        result = emit_key_snippet.invoke({"snippet_info": snippet})
        assert result == f"Snippet #{i} stored."
    assert _global_memory["key_snippet_id_counter"] == 3
    # Verify related files were tracked with IDs
    assert len(_global_memory["related_files"]) == 3
    # Check files are stored with proper IDs
    file_values = _global_memory["related_files"].values()
    assert str(file1) in file_values
    assert str(file2) in file_values
    assert str(file3) in file_values

    # Verify all snippets were stored correctly
    assert len(_global_memory["key_snippets"]) == 3
    assert _global_memory["key_snippets"][0] == snippets[0]
    assert _global_memory["key_snippets"][1] == snippets[1]
    assert _global_memory["key_snippets"][2] == snippets[2]

    # Delete some but not all snippets (0 and 2)
    result = delete_key_snippets.invoke({"snippet_ids": [0, 2]})
    assert result == "Snippets deleted."

    # Verify remaining snippet is intact
    assert len(_global_memory["key_snippets"]) == 1
    assert 1 in _global_memory["key_snippets"]
    assert _global_memory["key_snippets"][1] == snippets[1]

    # Counter should remain unchanged after deletions
    assert _global_memory["key_snippet_id_counter"] == 3

    # Add new snippet to verify counter continues correctly
    file4 = tmp_path / "file4.py"
    file4.write_text("def func4():\n    return False")
    new_snippet = {
        "filepath": str(file4),
        "line_number": 40,
        "snippet": "def func4():\n    return False",
        "description": "Fourth function",
    }
    result = emit_key_snippet.invoke({"snippet_info": new_snippet})
    assert result == "Snippet #3 stored."
    assert _global_memory["key_snippet_id_counter"] == 4
    # Verify new file was added to related files
    file_values = _global_memory["related_files"].values()
    assert str(file4) in file_values
    assert len(_global_memory["related_files"]) == 4

    # Delete remaining snippets
    result = delete_key_snippets.invoke({"snippet_ids": [1, 3]})
    assert result == "Snippets deleted."

    # Verify all snippets are gone
    assert len(_global_memory["key_snippets"]) == 0

    # Counter should still maintain its value
    assert _global_memory["key_snippet_id_counter"] == 4


def test_emit_task_with_id(reset_memory):
    """Test emitting tasks with ID tracking"""
    # Test adding a single task
    task = "Implement new feature"
    result = emit_task.invoke({"task": task})

    # Verify return message includes task ID
    assert result == "Task #0 stored."

    # Verify task stored correctly with ID
    assert _global_memory["tasks"][0] == task

    # Verify counter incremented
    assert _global_memory["task_id_counter"] == 1

    # Add another task to verify counter continues correctly
    task2 = "Fix bug"
    result = emit_task.invoke({"task": task2})
    assert result == "Task #1 stored."
    assert _global_memory["tasks"][1] == task2
    assert _global_memory["task_id_counter"] == 2


def test_delete_tasks(reset_memory):
    """Test deleting tasks"""
    # Add some test tasks
    tasks = ["Task 1", "Task 2", "Task 3"]
    for task in tasks:
        emit_task.invoke({"task": task})

    # Test deleting single task
    result = delete_tasks.invoke({"task_ids": [1]})
    assert result == "Tasks deleted."
    assert 1 not in _global_memory["tasks"]
    assert len(_global_memory["tasks"]) == 2

    # Test deleting multiple tasks including non-existent ID
    result = delete_tasks.invoke({"task_ids": [0, 2, 999]})
    assert result == "Tasks deleted."
    assert len(_global_memory["tasks"]) == 0

    # Test deleting from empty tasks dict
    result = delete_tasks.invoke({"task_ids": [0]})
    assert result == "Tasks deleted."

    # Counter should remain unchanged after deletions
    assert _global_memory["task_id_counter"] == 3


def test_swap_task_order_valid_ids(reset_memory):
    """Test basic task swapping functionality"""
    # Add test tasks
    tasks = ["Task 1", "Task 2", "Task 3"]
    for task in tasks:
        emit_task.invoke({"task": task})

    # Swap tasks 0 and 2
    result = swap_task_order.invoke({"id1": 0, "id2": 2})
    assert result == "Tasks deleted."

    # Verify tasks were swapped
    assert _global_memory["tasks"][0] == "Task 3"
    assert _global_memory["tasks"][2] == "Task 1"
    assert _global_memory["tasks"][1] == "Task 2"  # Unchanged


def test_swap_task_order_invalid_ids(reset_memory):
    """Test error handling for invalid task IDs"""
    # Add a test task
    emit_task.invoke({"task": "Task 1"})

    # Try to swap with non-existent ID
    result = swap_task_order.invoke({"id1": 0, "id2": 999})
    assert result == "Invalid task ID(s)"

    # Verify original task unchanged
    assert _global_memory["tasks"][0] == "Task 1"


def test_swap_task_order_same_id(reset_memory):
    """Test handling of attempt to swap a task with itself"""
    # Add test task
    emit_task.invoke({"task": "Task 1"})

    # Try to swap task with itself
    result = swap_task_order.invoke({"id1": 0, "id2": 0})
    assert result == "Cannot swap task with itself"

    # Verify task unchanged
    assert _global_memory["tasks"][0] == "Task 1"


def test_swap_task_order_empty_tasks(reset_memory):
    """Test swapping behavior with empty tasks dictionary"""
    result = swap_task_order.invoke({"id1": 0, "id2": 1})
    assert result == "Invalid task ID(s)"


def test_swap_task_order_after_delete(reset_memory):
    """Test swapping after deleting a task"""
    # Add test tasks
    tasks = ["Task 1", "Task 2", "Task 3"]
    for task in tasks:
        emit_task.invoke({"task": task})

    # Delete middle task
    delete_tasks.invoke({"task_ids": [1]})

    # Try to swap with deleted task
    result = swap_task_order.invoke({"id1": 0, "id2": 1})
    assert result == "Invalid task ID(s)"

    # Try to swap remaining valid tasks
    result = swap_task_order.invoke({"id1": 0, "id2": 2})
    assert result == "Tasks deleted."

    # Verify swap worked
    assert _global_memory["tasks"][0] == "Task 3"
    assert _global_memory["tasks"][2] == "Task 1"


def test_emit_related_files_binary_filtering(reset_memory, tmp_path, monkeypatch):
    """Test that binary files are filtered out when adding related files"""
    # Create test text files
    text_file1 = tmp_path / "text1.txt"
    text_file1.write_text("Text file 1 content")
    text_file2 = tmp_path / "text2.txt"
    text_file2.write_text("Text file 2 content")

    # Create test "binary" files
    binary_file1 = tmp_path / "binary1.bin"
    binary_file1.write_text("Binary file 1 content")
    binary_file2 = tmp_path / "binary2.bin"
    binary_file2.write_text("Binary file 2 content")

    # Mock the is_binary_file function to identify our "binary" files
    def mock_is_binary_file(filepath):
        return ".bin" in str(filepath)

    # Apply the mock
    import ra_aid.tools.memory

    monkeypatch.setattr(ra_aid.tools.memory, "is_binary_file", mock_is_binary_file)

    # Call emit_related_files with mix of text and binary files
    result = emit_related_files.invoke(
        {
            "files": [
                str(text_file1),
                str(binary_file1),
                str(text_file2),
                str(binary_file2),
            ]
        }
    )

    # Verify the result message mentions skipped binary files
    assert "Files noted." in result
    assert "Binary files skipped:" in result
    assert f"'{binary_file1}'" in result
    assert f"'{binary_file2}'" in result

    # Verify only text files were added to related_files
    assert len(_global_memory["related_files"]) == 2
    file_values = list(_global_memory["related_files"].values())
    assert str(text_file1) in file_values
    assert str(text_file2) in file_values
    assert str(binary_file1) not in file_values
    assert str(binary_file2) not in file_values

    # Verify counter is correct (only incremented for text files)
    assert _global_memory["related_file_id_counter"] == 2


def test_is_binary_file_with_ascii(reset_memory, monkeypatch):
    """Test that ASCII files are correctly identified as text files"""
    import os

    import ra_aid.tools.memory

    # Path to the mock ASCII file
    ascii_file_path = os.path.join(
        os.path.dirname(__file__), "..", "mocks", "ascii.txt"
    )

    # Test with magic library if available
    if ra_aid.tools.memory.magic:
        # Test real implementation with ASCII file
        is_binary = ra_aid.tools.memory.is_binary_file(ascii_file_path)
        assert not is_binary, "ASCII file should not be identified as binary"

    # Test fallback implementation
    # Mock magic to be None to force fallback implementation
    monkeypatch.setattr(ra_aid.tools.memory, "magic", None)

    # Test fallback with ASCII file
    is_binary = ra_aid.tools.memory.is_binary_file(ascii_file_path)
    assert (
        not is_binary
    ), "ASCII file should not be identified as binary with fallback method"


def test_is_binary_file_with_null_bytes(reset_memory, tmp_path, monkeypatch):
    """Test that files with null bytes are correctly identified as binary"""
    import ra_aid.tools.memory

    # Create a file with null bytes (binary content)
    binary_file = tmp_path / "binary_with_nulls.bin"
    with open(binary_file, "wb") as f:
        f.write(b"Some text with \x00 null \x00 bytes")

    # Test with magic library if available
    if ra_aid.tools.memory.magic:
        # Test real implementation with binary file
        is_binary = ra_aid.tools.memory.is_binary_file(str(binary_file))
        assert is_binary, "File with null bytes should be identified as binary"

    # Test fallback implementation
    # Mock magic to be None to force fallback implementation
    monkeypatch.setattr(ra_aid.tools.memory, "magic", None)

    # Test fallback with binary file
    is_binary = ra_aid.tools.memory.is_binary_file(str(binary_file))
    assert (
        is_binary
    ), "File with null bytes should be identified as binary with fallback method"
