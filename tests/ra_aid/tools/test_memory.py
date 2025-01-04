import pytest
from ra_aid.tools.memory import (
    _global_memory,
    get_memory_value,
    emit_key_facts,
    delete_key_facts,
    emit_key_snippets,
    delete_key_snippets,
    emit_related_files,
    get_related_files,
    deregister_related_files,
    emit_task,
    delete_tasks,
    swap_task_order,
    log_work_event,
    reset_work_log,
    get_work_log
)

@pytest.fixture
def reset_memory():
    """Reset global memory before each test"""
    _global_memory['key_facts'] = {}
    _global_memory['key_fact_id_counter'] = 0
    _global_memory['key_snippets'] = {}
    _global_memory['key_snippet_id_counter'] = 0
    _global_memory['research_notes'] = []
    _global_memory['plans'] = []
    _global_memory['tasks'] = {}
    _global_memory['task_id_counter'] = 0
    _global_memory['related_files'] = {}
    _global_memory['related_file_id_counter'] = 0
    _global_memory['work_log'] = []
    yield
    # Clean up after test
    _global_memory['key_facts'] = {}
    _global_memory['key_fact_id_counter'] = 0
    _global_memory['key_snippets'] = {}
    _global_memory['key_snippet_id_counter'] = 0
    _global_memory['research_notes'] = []
    _global_memory['plans'] = []
    _global_memory['tasks'] = {}
    _global_memory['task_id_counter'] = 0
    _global_memory['related_files'] = {}
    _global_memory['related_file_id_counter'] = 0
    _global_memory['work_log'] = []

def test_emit_key_facts_single_fact(reset_memory):
    """Test emitting a single key fact using emit_key_facts"""
    # Test with single fact
    result = emit_key_facts.invoke({"facts": ["First fact"]})
    assert result == "Facts stored."
    assert _global_memory['key_facts'][0] == "First fact"
    assert _global_memory['key_fact_id_counter'] == 1

def test_delete_key_facts_single_fact(reset_memory):
    """Test deleting a single key fact using delete_key_facts"""
    # Add a fact
    emit_key_facts.invoke({"facts": ["Test fact"]})
    
    # Delete the fact
    result = delete_key_facts.invoke({"fact_ids": [0]})
    assert result == "Facts deleted."
    assert 0 not in _global_memory['key_facts']

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
    assert get_memory_value('key_facts') == ""
    
    # Add some facts
    emit_key_facts.invoke({"facts": ["First fact", "Second fact"]})
    
    # Should return markdown formatted list
    expected = "## 🔑 Key Fact #0\n\nFirst fact\n\n## 🔑 Key Fact #1\n\nSecond fact"
    assert get_memory_value('key_facts') == expected

def test_get_memory_value_other_types(reset_memory):
    """Test get_memory_value remains compatible with other memory types"""
    # Add some research notes
    _global_memory['research_notes'].append("Note 1")
    _global_memory['research_notes'].append("Note 2")
    
    assert get_memory_value('research_notes') == "Note 1\nNote 2"
    
    # Test with empty list
    assert get_memory_value('plans') == ""
    
    # Test with non-existent key
    assert get_memory_value('nonexistent') == ""

def test_log_work_event(reset_memory):
    """Test logging work events with timestamps"""
    # Log some events
    log_work_event("Started task")
    log_work_event("Made progress")
    log_work_event("Completed task")
    
    # Verify events are stored
    assert len(_global_memory['work_log']) == 3
    
    # Check event structure
    event = _global_memory['work_log'][0]
    assert isinstance(event['timestamp'], str)
    assert event['event'] == "Started task"
    
    # Verify order
    assert _global_memory['work_log'][1]['event'] == "Made progress"
    assert _global_memory['work_log'][2]['event'] == "Completed task"

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
    assert len(_global_memory['work_log']) == 1
    
    # Reset log
    reset_work_log()
    
    # Verify log is empty
    assert len(_global_memory['work_log']) == 0
    assert get_memory_value('work_log') == ""

def test_empty_work_log(reset_memory):
    """Test empty work log behavior"""
    # Fresh work log should return empty string
    assert get_memory_value('work_log') == ""

def test_emit_key_facts(reset_memory):
    """Test emitting multiple key facts at once"""
    # Test emitting multiple facts
    facts = ["First fact", "Second fact", "Third fact"]
    result = emit_key_facts.invoke({"facts": facts})
    
    # Verify return message
    assert result == "Facts stored."
    
    # Verify facts stored in memory with correct IDs
    assert _global_memory['key_facts'][0] == "First fact"
    assert _global_memory['key_facts'][1] == "Second fact"
    assert _global_memory['key_facts'][2] == "Third fact"
    
    # Verify counter incremented correctly
    assert _global_memory['key_fact_id_counter'] == 3

def test_delete_key_facts(reset_memory):
    """Test deleting multiple key facts"""
    # Add some test facts
    emit_key_facts.invoke({"facts": ["First fact", "Second fact", "Third fact"]})
    
    # Test deleting mix of existing and non-existing IDs
    result = delete_key_facts.invoke({"fact_ids": [0, 1, 999]})
    
    # Verify success message
    assert result == "Facts deleted."
    
    # Verify correct facts removed from memory
    assert 0 not in _global_memory['key_facts']
    assert 1 not in _global_memory['key_facts']
    assert 2 in _global_memory['key_facts']  # ID 2 should remain
    assert _global_memory['key_facts'][2] == "Third fact"

def test_emit_key_snippets(reset_memory):
    """Test emitting multiple code snippets at once"""
    # Test snippets with and without descriptions
    snippets = [
        {
            "snippet": "def test():\n    pass",
            "source": "test.py:10",
            "relevance": 0.8,
            "filepath": "test.py",
            "line_number": 10,
            "description": "Test function"
        },
        {
            "snippet": "print('hello')",
            "source": "main.py:20",
            "relevance": 0.6,
            "filepath": "main.py",
            "line_number": 20,
            "description": None
        }
    ]
    
    # Emit snippets
    result = emit_key_snippets.invoke({"snippets": snippets})
    
    # Verify return message
    assert result == "Snippets stored."
    
    # Verify snippets stored correctly
    assert _global_memory['key_snippets'][0] == snippets[0]
    assert _global_memory['key_snippets'][1] == snippets[1]
    
    # Verify counter incremented correctly
    assert _global_memory['key_snippet_id_counter'] == 2

def test_delete_key_snippets(reset_memory):
    """Test deleting multiple code snippets"""
    # Add test snippets
    snippets = [
        {
            "snippet": "code1",
            "source": "test1.py:1",
            "relevance": 0.7,
            "filepath": "test1.py",
            "line_number": 1,
            "description": None
        },
        {
            "snippet": "code2",
            "source": "test2.py:2",
            "relevance": 0.8,
            "filepath": "test2.py",
            "line_number": 2,
            "description": None
        },
        {
            "snippet": "code3",
            "source": "test3.py:3",
            "relevance": 0.9,
            "filepath": "test3.py",
            "line_number": 3,
            "description": None
        }
    ]
    emit_key_snippets.invoke({"snippets": snippets})
    
    # Test deleting mix of valid and invalid IDs
    result = delete_key_snippets.invoke({"snippet_ids": [0, 1, 999]})
    
    # Verify success message
    assert result == "Snippets deleted."
    
    # Verify correct snippets removed
    assert 0 not in _global_memory['key_snippets']
    assert 1 not in _global_memory['key_snippets']
    assert 2 in _global_memory['key_snippets']
    assert _global_memory['key_snippets'][2]['filepath'] == "test3.py"

def test_delete_key_snippets_empty(reset_memory):
    """Test deleting snippets from empty memory"""
    # Add a single snippet
    snippet = {
        "snippet": "test code",
        "source": "test.py:1",
        "relevance": 0.5,
        "filepath": "test.py",
        "line_number": 1,
        "description": None
    }
    emit_key_snippets.invoke({"snippets": [snippet]})
    
    # Delete the snippet
    delete_key_snippets.invoke({"snippet_ids": [0]})
    
    # Try to delete again - should succeed silently
    result = delete_key_snippets.invoke({"snippet_ids": [0]})
    assert result == "Snippets deleted."
    
    # Memory should be empty
    assert len(_global_memory['key_snippets']) == 0

def test_emit_related_files_basic(reset_memory):
    """Test basic adding of files with ID tracking"""
    # Test adding single file
    result = emit_related_files.invoke({"files": ["test.py"]})
    assert result == "File ID #0: test.py"
    assert _global_memory['related_files'][0] == "test.py"
    
    # Test adding multiple files
    result = emit_related_files.invoke({"files": ["main.py", "utils.py"]})
    assert result == "File ID #1: main.py\nFile ID #2: utils.py"
    # Verify both files exist in related_files
    values = list(_global_memory['related_files'].values())
    assert "main.py" in values
    assert "utils.py" in values

def test_get_related_files_empty(reset_memory):
    """Test getting related files when none added"""
    assert get_related_files() == []

def test_emit_related_files_duplicates(reset_memory):
    """Test that duplicate files return existing IDs with proper formatting"""
    # Add initial files
    result = emit_related_files.invoke({"files": ["test.py", "main.py"]})
    assert result == "File ID #0: test.py\nFile ID #1: main.py"
    first_id = 0  # ID of test.py
    
    # Try adding duplicates
    result = emit_related_files.invoke({"files": ["test.py"]})
    assert result == "File ID #0: test.py"  # Should return same ID
    assert len(_global_memory['related_files']) == 2  # Count should not increase
    
    # Try mix of new and duplicate files
    result = emit_related_files.invoke({"files": ["test.py", "new.py"]})
    assert result == "File ID #0: test.py\nFile ID #2: new.py"
    assert len(_global_memory['related_files']) == 3

def test_related_files_id_tracking(reset_memory):
    """Test ID assignment and counter functionality for related files"""
    # Add first file
    result = emit_related_files.invoke({"files": ["file1.py"]})
    assert result == "File ID #0: file1.py"
    assert _global_memory['related_file_id_counter'] == 1
    
    # Add second file
    result = emit_related_files.invoke({"files": ["file2.py"]})
    assert result == "File ID #1: file2.py"
    assert _global_memory['related_file_id_counter'] == 2
    
    # Verify all files stored correctly
    assert _global_memory['related_files'][0] == "file1.py"
    assert _global_memory['related_files'][1] == "file2.py"

def test_deregister_related_files(reset_memory):
    """Test deleting related files"""
    # Add test files
    emit_related_files.invoke({"files": ["file1.py", "file2.py", "file3.py"]})
    
    # Delete middle file
    result = deregister_related_files.invoke({"file_ids": [1]})
    assert result == "File references removed."
    assert 1 not in _global_memory['related_files']
    assert len(_global_memory['related_files']) == 2
    
    # Delete multiple files including non-existent ID
    result = deregister_related_files.invoke({"file_ids": [0, 2, 999]})
    assert result == "File references removed."
    assert len(_global_memory['related_files']) == 0
    
    # Counter should remain unchanged after deletions
    assert _global_memory['related_file_id_counter'] == 3

def test_related_files_duplicates(reset_memory):
    """Test duplicate file handling returns same ID"""
    # Add initial file
    result1 = emit_related_files.invoke({"files": ["test.py"]})
    assert result1 == "File ID #0: test.py"
    
    # Add same file again
    result2 = emit_related_files.invoke({"files": ["test.py"]})
    assert result2 == "File ID #0: test.py"
    
    # Verify only one entry exists
    assert len(_global_memory['related_files']) == 1
    assert _global_memory['related_file_id_counter'] == 1

def test_related_files_formatting(reset_memory):
    """Test related files output string formatting"""
    # Add some files
    emit_related_files.invoke({"files": ["file1.py", "file2.py"]})
    
    # Get formatted output
    output = get_memory_value('related_files')
    # Expect just the IDs on separate lines
    expected = "0\n1"
    assert output == expected
    
    # Test empty case
    _global_memory['related_files'] = {}
    assert get_memory_value('related_files') == ""

def test_key_snippets_integration(reset_memory):
    """Test full integration of key snippets functionality"""
    # Test snippets with various descriptions and content
    snippets = [
        {
            "snippet": "def first_function():\n    return 'first'",
            "source": "file1.py:10",
            "relevance": 0.9,
            "filepath": "file1.py",
            "line_number": 10,
            "description": "First function"
        },
        {
            "snippet": "def second_function():\n    return 'second'",
            "source": "file2.py:20",
            "relevance": 0.8,
            "filepath": "file2.py",
            "line_number": 20,
            "description": "Second function"
        },
        {
            "snippet": "class TestClass:\n    pass",
            "source": "file3.py:30",
            "relevance": 0.7,
            "filepath": "file3.py",
            "line_number": 30,
            "description": "Test class"
        }
    ]
    
    # Emit snippets
    result = emit_key_snippets.invoke({"snippets": snippets})
    assert result == "Snippets stored."
    
    # Verify snippets stored correctly
    assert len(_global_memory['key_snippets']) == 3
    assert _global_memory['key_snippets'][0] == snippets[0]
    assert _global_memory['key_snippets'][1] == snippets[1]
    assert _global_memory['key_snippets'][2] == snippets[2]
    
    # Delete middle snippet
    result = delete_key_snippets.invoke({"snippet_ids": [1]})
    assert result == "Snippets deleted."
    
    # Verify correct snippet was deleted
    assert len(_global_memory['key_snippets']) == 2
    assert _global_memory['key_snippets'][0] == snippets[0]
    assert _global_memory['key_snippets'][2] == snippets[2]
    
    # Get memory value and verify formatting
    memory_value = get_memory_value('key_snippets')
    assert "file1.py" in memory_value
    assert "first_function" in memory_value
    assert "file3.py" in memory_value
    assert "TestClass" in memory_value
    assert "file2.py" not in memory_value  # Deleted snippet
    assert "second_function" not in memory_value  # Deleted snippet

def test_emit_task_with_id(reset_memory):
    """Test emitting tasks with ID tracking"""
    # Test adding a single task
    task = "Implement new feature"
    result = emit_task.invoke({"task": task})
    
    # Verify return message includes task ID
    assert result == "Task #0 stored."
    
    # Verify task stored correctly with ID
    assert _global_memory['tasks'][0] == task
    
    # Verify counter incremented
    assert _global_memory['task_id_counter'] == 1
    
    # Add another task to verify counter continues correctly
    task2 = "Fix bug"
    result = emit_task.invoke({"task": task2})
    assert result == "Task #1 stored."
    assert _global_memory['tasks'][1] == task2
    assert _global_memory['task_id_counter'] == 2

def test_delete_tasks(reset_memory):
    """Test deleting tasks"""
    # Add some test tasks
    tasks = ["Task 1", "Task 2", "Task 3"]
    for task in tasks:
        emit_task.invoke({"task": task})
    
    # Test deleting single task
    result = delete_tasks.invoke({"task_ids": [1]})
    assert result == "Tasks deleted."
    assert 1 not in _global_memory['tasks']
    assert len(_global_memory['tasks']) == 2
    
    # Test deleting multiple tasks including non-existent ID
    result = delete_tasks.invoke({"task_ids": [0, 2, 999]})
    assert result == "Tasks deleted."
    assert len(_global_memory['tasks']) == 0
    
    # Test deleting from empty tasks dict
    result = delete_tasks.invoke({"task_ids": [0]})
    assert result == "Tasks deleted."
    
    # Counter should remain unchanged after deletions
    assert _global_memory['task_id_counter'] == 3

def test_swap_task_order_valid_ids(reset_memory):
    """Test basic task swapping functionality"""
    # Add test tasks
    tasks = ["Task 1", "Task 2", "Task 3"]
    for task in tasks:
        emit_task.invoke({"task": task})
    
    # Swap tasks 0 and 2
    result = swap_task_order.invoke({"id1": 0, "id2": 2})
    assert result == "Tasks swapped."
    
    # Verify tasks were swapped
    assert _global_memory['tasks'][0] == "Task 3"
    assert _global_memory['tasks'][2] == "Task 1"
    assert _global_memory['tasks'][1] == "Task 2"  # Unchanged

def test_swap_task_order_invalid_ids(reset_memory):
    """Test error handling for invalid task IDs"""
    # Add a test task
    emit_task.invoke({"task": "Task 1"})
    
    # Try to swap with non-existent ID
    result = swap_task_order.invoke({"id1": 0, "id2": 999})
    assert result == "Invalid task ID(s)"
    
    # Verify original task unchanged
    assert _global_memory['tasks'][0] == "Task 1"

def test_swap_task_order_same_id(reset_memory):
    """Test handling of attempt to swap a task with itself"""
    # Add test task
    emit_task.invoke({"task": "Task 1"})
    
    # Try to swap task with itself
    result = swap_task_order.invoke({"id1": 0, "id2": 0})
    assert result == "Cannot swap task with itself"
    
    # Verify task unchanged
    assert _global_memory['tasks'][0] == "Task 1"

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
    assert result == "Tasks swapped."
    
    # Verify swap worked
    assert _global_memory['tasks'][0] == "Task 3"
    assert _global_memory['tasks'][2] == "Task 1"
