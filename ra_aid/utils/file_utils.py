"""Utility functions for file operations."""

import os

try:
    import magic
except ImportError:
    magic = None


def is_binary_file(filepath):
    """Check if a file is binary using magic library if available."""
    # First check if file is empty
    if os.path.getsize(filepath) == 0:
        return False  # Empty files are not binary

    if magic:
        try:
            mime = magic.from_file(filepath, mime=True)
            file_type = magic.from_file(filepath)

            # If MIME type starts with 'text/', it's likely a text file
            if mime.startswith("text/"):
                return False

            # Also consider 'application/x-python' and similar script types as text
            if any(mime.startswith(prefix) for prefix in ['application/x-python', 'application/javascript']):
                return False

            # Check for common text file descriptors
            text_indicators = ["text", "script", "xml", "json", "yaml", "markdown", "HTML"]
            if any(indicator.lower() in file_type.lower() for indicator in text_indicators):
                return False

            # If none of the text indicators are present, assume it's binary
            return True
        except Exception:
            return _is_binary_fallback(filepath)
    else:
        return _is_binary_fallback(filepath)


def _is_binary_fallback(filepath):
    """Fallback method to detect binary files without using magic."""
    try:
        # First check if file is empty
        if os.path.getsize(filepath) == 0:
            return False  # Empty files are not binary

        with open(filepath, "r", encoding="utf-8") as f:
            chunk = f.read(1024)

            # Check for null bytes which indicate binary content
            if "\0" in chunk:
                return True

            # If we can read it as text without errors, it's probably not binary
            return False
    except UnicodeDecodeError:
        # If we can't decode as UTF-8, it's likely binary
        return True