"""Functions for extracting and validating tool function signatures and documentation.

This module provides utilities for:
- Extracting function signatures and docstrings via reflection
- Formatting tool information for agent consumption
"""

import inspect

__all__ = ["get_function_info"]


def get_function_info(func):
    """Return a well-formatted string containing the function signature and docstring.

    Uses Python's inspect module to extract and format function metadata in a way
    that is easily readable by both humans and language models.

    Args:
        func: The function to analyze

    Returns:
        str: Formatted string containing function name, signature and docstring
    """
    signature = inspect.signature(func)
    docstring = inspect.getdoc(func)
    if docstring is None:
        docstring = "No docstring provided"
    full_signature = f"{func.__name__}{signature}"
    info = f"""{full_signature}
\"\"\"
{docstring}
\"\"\""""
    return info
