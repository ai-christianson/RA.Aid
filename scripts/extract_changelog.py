#!/usr/bin/env python3
"""
Extract changelog entries for a specific version from CHANGELOG.md.

Usage:
    python extract_changelog.py VERSION
"""

import re
import sys
from pathlib import Path


def extract_version_content(content: str, version: str) -> str:
    """Extract content for specified version from changelog text."""
    # Escape version for regex pattern
    version_escaped = re.escape(version)
    pattern = rf"## \[{version_escaped}\].*?(?=## \[|$)"

    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(f"Version {version} not found in changelog")

    return match.group(0).strip()


def main():
    """Main entry point for the script."""
    if len(sys.argv) != 2:
        print("Usage: python extract_changelog.py VERSION", file=sys.stderr)
        sys.exit(1)

    version = sys.argv[1]
    changelog_path = Path(__file__).parent.parent / "CHANGELOG.md"

    try:
        content = changelog_path.read_text()
    except FileNotFoundError:
        print(f"Error: Could not find {changelog_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading changelog: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        version_content = extract_version_content(content, version)
        print(version_content)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
