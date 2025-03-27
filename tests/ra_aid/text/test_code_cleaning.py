import unittest
import ast
import os
from ra_aid.text import fix_triple_quote_contents


class TestFixTripleQuoteContents(unittest.TestCase):
    """Test the fix_triple_quote_contents function from the code_cleaning module."""

    @classmethod
    def setUpClass(cls):
        """Set up paths to test examples."""
        # Path to the test examples directory
        cls.examples_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "code_cleaning", "examples"
        )
    
    def _load_example_file(self, example_name):
        """Load an example from a file."""
        file_path = os.path.join(self.examples_dir, f"{example_name}.py")
        with open(file_path, "r") as f:
            return f.read()

    def _test_example(self, example_name):
        """Generic test function for testing a code example."""
        code = self._load_example_file(example_name)
        
        # Check that original code is invalid
        with self.assertRaises(SyntaxError):
            ast.parse(code)
        
        # Fix the code
        fixed_code = fix_triple_quote_contents(code)
        
        # Check that fixed code is valid
        try:
            ast.parse(fixed_code)
            is_valid = True
        except SyntaxError as e:
            is_valid = False
            print(f"SyntaxError with fixed code: {e}")
        
        self.assertTrue(is_valid, f"Example '{example_name}' fixed code is not valid Python: {fixed_code}")
        return fixed_code

    def test_complex_nested_triple_quotes(self):
        """Test fixing a more complex case of nested triple quotes."""
        self._test_example("complex_nested_triple_quotes")
        
    def test_already_escaped_triple_quotes_preserved(self):
        """Test that already escaped triple quotes are preserved without modification."""
        code = self._load_example_file("complex_nested_triple_quotes_already_escaped")
        
        # Store the original code for comparison
        original_code = code
        
        # Apply the fix function
        fixed_code = fix_triple_quote_contents(code)
        
        # Verify that the code was not modified (it was already correctly escaped)
        self.assertEqual(fixed_code, original_code, 
                        "Already escaped triple quotes should not be modified")
        
        # Verify that the fixed code can be parsed as valid Python
        try:
            ast.parse(fixed_code)
        except SyntaxError as e:
            self.fail(f"Already escaped code should be valid Python, but got: {e}")
            
    def test_ascii_generator(self):
        """Test fixing a docstring with unclosed triple quotes in an ASCII art generator function."""
        # This test should also handle the case of matching the closing quote style with the opening
        fixed_code = self._test_example("ascii_generator")
        print(fixed_code)
        
        # Verify that the fixed code maintains function docstring structure
        self.assertIn('def generate_ascii_art(text: str) -> str:', fixed_code)
        self.assertIn('"""', fixed_code)  # Should have triple quotes
        self.assertIn('Returns:', fixed_code)  # Important docstring content preserved


if __name__ == "__main__":
    unittest.main()
