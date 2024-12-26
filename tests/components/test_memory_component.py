import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from components.memory import initialize_memory, get_memory, _global_memory

def test_initialize_memory():
    """Test memory initialization."""
    initialize_memory()
    
    memory = get_memory()
    assert isinstance(memory, dict)
    assert 'related_files' in memory
    assert 'implementation_requested' in memory
    assert 'config' in memory
    assert 'research_notes' in memory
    assert 'plans' in memory
    assert 'tasks' in memory
    assert 'key_facts' in memory
    assert 'key_snippets' in memory
    
    # Check default values
    assert memory['related_files'] == {}
    assert memory['implementation_requested'] == False
    assert memory['config'] == {}
    assert memory['research_notes'] == []
    assert memory['plans'] == []
    assert memory['tasks'] == {}
    assert memory['key_facts'] == {}
    assert memory['key_snippets'] == {}

def test_get_memory():
    """Test memory retrieval."""
    initialize_memory()
    
    # Add test data
    _global_memory['test_key'] = 'test_value'
    _global_memory['related_files']['test.py'] = 'test content'
    _global_memory['research_notes'].append('test note')
    
    memory = get_memory()
    assert memory['test_key'] == 'test_value'
    assert memory['related_files']['test.py'] == 'test content'
    assert memory['research_notes'] == ['test note']

def test_memory_persistence():
    """Test memory persistence between operations."""
    initialize_memory()
    
    # First operation
    _global_memory['operation1'] = 'completed'
    memory1 = get_memory()
    assert memory1['operation1'] == 'completed'
    
    # Second operation
    _global_memory['operation2'] = 'in_progress'
    memory2 = get_memory()
    assert memory2['operation1'] == 'completed'
    assert memory2['operation2'] == 'in_progress'

def test_memory_reset():
    """Test memory reset functionality."""
    # Set initial state
    _global_memory['test_key'] = 'test_value'
    _global_memory['related_files']['test.py'] = 'test content'
    
    # Reset memory
    initialize_memory()
    
    memory = get_memory()
    assert 'test_key' not in memory
    assert memory['related_files'] == {}
    assert memory['implementation_requested'] == False 