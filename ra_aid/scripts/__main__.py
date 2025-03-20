"""Command-line entry point for session usage statistics."""

import json
import sys
from ra_aid.scripts.last_session_usage import get_latest_session_usage

def main():
    """
    Command-line entry point for getting usage statistics for the latest session.
    
    This function retrieves the latest session and calculates its total usage metrics,
    then outputs the results as JSON to stdout.
    """
    result, status_code = get_latest_session_usage()
    print(json.dumps(result, indent=2))
    return status_code

# This is the entry point when running as a module
if __name__ == "__main__":
    sys.exit(main())

# This function is called by runpy when executing the module
def run_module():
    """Entry point for runpy module execution."""
    return main()
