#!/usr/bin/env python3
"""
Command-line interface for RA.Aid scripts.

This module provides command-line entry points for various RA.Aid utilities.
"""

import sys
import json
import argparse
from ra_aid.scripts.last_session_usage import get_latest_session_usage
from ra_aid.scripts.all_sessions_usage import get_all_sessions_usage

def session_usage_command():
    """
    Command-line entry point for getting usage statistics for the latest session.
    
    This function retrieves the latest session and calculates its total usage metrics,
    then outputs the results as JSON to stdout.
    """
    result, status_code = get_latest_session_usage()
    print(json.dumps(result, indent=2))
    return status_code

def all_sessions_usage_command():
    """
    Command-line entry point for getting usage statistics for all sessions.
    
    This function retrieves all sessions and calculates their usage metrics,
    then outputs the results as JSON to stdout.
    """
    results, status_code = get_all_sessions_usage()
    print(json.dumps(results, indent=2))
    return status_code

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="RA.Aid utility scripts")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Latest session command
    latest_parser = subparsers.add_parser("latest", help="Get usage statistics for the latest session")
    
    # All sessions command
    all_parser = subparsers.add_parser("all", help="Get usage statistics for all sessions")
    
    args = parser.parse_args()
    
    if args.command == "latest" or not args.command:
        return session_usage_command()
    elif args.command == "all":
        return all_sessions_usage_command()
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
