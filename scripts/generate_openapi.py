#!/usr/bin/env python3
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).parent.parent  # Navigate up to the project root
sys.path.insert(0, str(project_root))

# Import the FastAPI app
from ra_aid.server.server import app  # Adjust if necessary

# Generate OpenAPI schema
openapi_spec = app.openapi()

# Save it to a file (optional)
output_path = Path(__file__).parent / "openapi.json"
# with output_path.open("w") as f:
#     json.dump(openapi_spec, f, indent=2)

# Print OpenAPI spec
print(json.dumps(openapi_spec, indent=2))
