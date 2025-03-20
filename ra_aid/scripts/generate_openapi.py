#!/usr/bin/env python3
import json
import sys
import yaml
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).parent.parent  # Navigate up to the project root
sys.path.insert(0, str(project_root))

# Import the FastAPI app
from ra_aid.server.server import app  # Adjust if necessary

# Generate OpenAPI schema
openapi_spec = app.openapi()

# Print OpenAPI spec as yaml
print(yaml.dump(openapi_spec, indent=2))
