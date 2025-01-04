"""Mock module for component imports."""
from unittest.mock import MagicMock

# Mock the component functions
research_component = MagicMock()
planning_component = MagicMock()
implementation_component = MagicMock()

# Set default return values
research_component.return_value = {
    'key_facts': ['fact1', 'fact2'],
    'key_snippets': ['snippet1', 'snippet2'],
    'success': True
}

planning_component.return_value = {
    'plans': ['plan1', 'plan2'],
    'tasks': ['task1', 'task2'],
    'success': True
}

implementation_component.return_value = {
    'implemented_tasks': ['task1', 'task2'],
    'success': True
} 