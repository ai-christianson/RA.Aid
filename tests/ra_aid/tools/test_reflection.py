from ra_aid.tools.reflection import get_function_info


# Sample functions for testing get_function_info
def simple_func():
    """A simple function with no parameters."""
    pass


def typed_func(a: int, b: str = "default") -> bool:
    """A function with type hints and default value.

    Args:
        a: An integer parameter
        b: A string parameter with default

    Returns:
        bool: Always returns True
    """
    return True


def complex_func(pos1, pos2, *args, kw1="default", **kwargs):
    """A function with complex signature."""
    pass


def no_docstring_func(x):
    pass


class TestGetFunctionInfo:
    def test_simple_function_info(self):
        """Test info extraction for simple function."""
        info = get_function_info(simple_func)
        assert "simple_func()" in info
        assert "A simple function with no parameters" in info

    def test_typed_function_info(self):
        """Test info extraction for function with type hints."""
        info = get_function_info(typed_func)
        assert "typed_func" in info
        assert "a: int" in info
        assert "b: str = 'default'" in info
        assert "-> bool" in info
        assert "Args:" in info
        assert "Returns:" in info

    def test_complex_function_info(self):
        """Test info extraction for function with complex signature."""
        info = get_function_info(complex_func)
        assert "complex_func" in info
        assert "pos1" in info
        assert "pos2" in info
        assert "*args" in info
        assert "**kwargs" in info
        assert "kw1='default'" in info
        assert "A function with complex signature" in info

    def test_no_docstring_function(self):
        """Test handling of functions without docstrings."""
        info = get_function_info(no_docstring_func)
        assert "no_docstring_func" in info
        assert "No docstring provided" in info
