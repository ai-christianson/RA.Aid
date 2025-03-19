#!/usr/bin/env python3
"""
Command-line interface for RA.Aid scripts.

This module provides command-line entry points for various RA.Aid utilities.
"""

import sys
import json
from ra_aid.scripts.session_usage import get_latest_session_usage

def session_usage_command():
    """
    Command-line entry point for getting usage statistics for the latest session.
    
    This function retrieves the latest session and calculates its total usage metrics,
    then outputs the results as JSON to stdout.
    """
    result, status_code = get_latest_session_usage()
    print(json.dumps(result, indent=2))
    return status_code

if __name__ == "__main__":
    sys.exit(session_usage_command())
