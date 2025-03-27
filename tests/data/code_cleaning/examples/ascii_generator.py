put_file_complete_contents(filename="example.py", content="""
"""
Converts text to ASCII art using the pyfiglet library.

Installation:
  pip install pyfiglet

Usage:
  python ascii_art_generator.py "Your Text"
  python ascii_art_generator.py  (for interactive input)
"""

import sys

try:
    import pyfiglet
except ImportError:
    print("Error: The 'pyfiglet' library is required to run this script.")
    print("Please install it using pip:")
    print("  pip install pyfiglet")
    sys.exit(1)

def generate_ascii_art(text: str) -> str:
    """
    Generates ASCII art representation of the input text using pyfiglet.

    Args:
        text: The string to convert.

    Returns:
        The ASCII art string.
    """
    ascii_art = pyfiglet.figlet_format(text)
    return ascii_art

if __name__ == '__main__':
    if len(sys.argv) > 1:
        input_text = sys.argv[1]
    else:
        input_text = input("Enter the text to convert: ")

    art = generate_ascii_art(input_text)
    print(art)
""")