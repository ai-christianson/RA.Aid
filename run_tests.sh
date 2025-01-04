#!/bin/bash

# Set up error handling
set -e

# Print startup message
echo "Running RA.Aid WebUI Tests..."

# Check for .env file
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    echo "Please create a .env file based on .env.template"
    exit 1
fi

# Load environment variables
echo "Loading environment variables..."
set -a
source .env
set +a

# Install test requirements if not already installed
echo "Installing test requirements..."
pip install -r tests/requirements-test.txt

# Run pytest with coverage
echo "Running tests..."
pytest

# Print completion message
echo "Tests completed!" 