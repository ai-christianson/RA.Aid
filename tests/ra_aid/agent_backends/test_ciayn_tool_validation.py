import pytest
from ra_aid.agent_backends.ciayn_agent import validate_function_call_pattern

def test_fuzzy_find_validation():
    # This is the exact function call from the error message
    function_call = 'fuzzy_find_project_files(search_term="bullet physics", repo_path=".", threshold=60, max_results=10, include_paths=None, exclude_patterns=None)'
    
    # The validate_function_call_pattern should return False for valid function calls
    # (False means "not invalid" in this function's logic)
    assert validate_function_call_pattern(function_call) is False, "The fuzzy_find_project_files call should be considered valid"

def test_validate_function_call_pattern_with_none_args():
    # Test with None as arguments for various parameter types
    valid_calls = [
        'some_function(arg1="test", arg2=None)',
        'some_function(arg1=None, arg2=123)',
        'some_function(arg1=None, arg2=None, arg3="text")',
        'fuzzy_find_project_files(search_term="bullet physics", repo_path=".", threshold=60, max_results=10, include_paths=None, exclude_patterns=None)',
    ]
    
    for call in valid_calls:
        assert validate_function_call_pattern(call) is False, f"Call should be valid: {call}"

def test_validate_function_call_pattern_invalid_syntax():
    # Test with invalid syntax
    invalid_calls = [
        'some_function(arg1="test"',  # Missing closing parenthesis
        'some_function arg1="test")',  # Missing opening parenthesis
        'some_function("test") another_function()',  # Multiple function calls
        '= some_function(arg1="test")',  # Invalid start
    ]
    
    for call in invalid_calls:
        assert validate_function_call_pattern(call) is True, f"Call should be invalid: {call}"
