import re
import io
import tokenize
import ast
from typing import List, Tuple

def fix_triple_quote_contents(text: str) -> str:
    """
    Fixes potentially incorrect escaping of nested triple quotes within Python code.

    Assumes that outermost triple quotes start on the first line and end on the last line
    of a string literal. Escapes any unescaped inner triple quotes of matching style.

    Args:
        text: A string containing Python code, potentially with unescaped
              nested triple quotes.

    Returns:
        A string with the nested triple quotes properly escaped.
    """
    if not text:
        return text

    lines = text.splitlines()
    result = []
    
    # Find the quote style from the first line
    first_line = lines[0]
    last_line = lines[-1]
    
    # Determine which quote style is used (single or double)
    quote_style = None
    if '"""' in first_line:
        quote_style = '"""'
    elif "'''" in first_line:
        quote_style = "'''"
    
    # If we can't find a quote style, or if we only have one line, return original
    if not quote_style or len(lines) <= 1:
        return text
    
    # Check if the same quote style appears in the last line
    if quote_style not in last_line:
        return text
    
    # Keep first line as is
    result.append(first_line)
    
    # Process all middle lines - escape any matching triple quotes
    for i in range(1, len(lines)-1):
        line = lines[i]
        
        # Find all unescaped instances of the quote style
        index = 0
        while index < len(line):
            found_index = line.find(quote_style, index)
            if found_index == -1:
                break
                
            # Check if it's already escaped
            if found_index > 0 and line[found_index-1] == '\\':
                # Count backslashes to handle cases like \\\"\"\"
                backslash_count = 0
                check_index = found_index - 1
                while check_index >= 0 and line[check_index] == '\\':
                    backslash_count += 1
                    check_index -= 1
                
                # If odd number of backslashes, it's already escaped
                if backslash_count % 2 == 1:
                    index = found_index + 3
                    continue
            
            # Found an unescaped triple quote - escape it
            line = line[:found_index] + '\\' + line[found_index:]
            index = found_index + 4  # Skip past the newly escaped triple quote
        
        result.append(line)
    
    # Keep last line as is
    result.append(last_line)
    
    return '\n'.join(result)