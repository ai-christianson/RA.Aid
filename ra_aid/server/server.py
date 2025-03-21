#!/usr/bin/env python3
import logging
import os
import sys
from pathlib import Path

from fastapi.openapi.utils import get_openapi

# Configure module-specific logging without affecting root logger
logger = logging.getLogger(__name__)
# Only configure this specific logger, not the root logger
if not logger.handlers:  # Avoid adding handlers multiple times
    logger.setLevel(logging.WARNING)
    handler = logging.StreamHandler(sys.__stderr__)  # Use the real stderr
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # Prevent propagation to avoid affecting the root logger configuration
    logger.propagate = False

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from ra_aid.server.api_v1_sessions import router as sessions_router
from ra_aid.server.api_v1_spawn_agent import router as spawn_agent_router

app = FastAPI(
    title="RA.Aid API",
    description="API for RA.Aid - AI Programming Assistant",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include API routers
app.include_router(sessions_router)
app.include_router(spawn_agent_router)

# Directory for the current file
CURRENT_DIR = Path(__file__).parent

# Placeholder WebSocket API endpoint
@app.websocket("/v1/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        await websocket.send_text("Connected to RA.Aid WebSocket placeholder API")
        # Keep the connection open until client disconnects
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message received: {data}")
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")


@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    """Return basic info about RA.Aid API."""
    return HTMLResponse(
        """
        <html>
            <head>
                <title>RA.Aid API</title>
                <style>
                    body { font-family: system-ui, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                    pre { background: #f4f4f4; padding: 10px; border-radius: 5px; }
                </style>
            </head>
            <body>
                <h1>RA.Aid API</h1>
                <p>A placeholder WebSocket API is available at /v1/ws. We recommend using the REST API endpoints for production use.</p>
                <p>See the <a href="/docs">API documentation</a> for more information.</p>
            </body>
        </html>
        """
    )


@app.get("/config")
async def get_config(request: Request):
    """Return server configuration including host and port."""
    return {"host": request.client.host, "port": request.scope.get("server")[1]}

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="RA.Aid API",
        summary="RA.Aid API OpenAPI Spec",
        version="1.0.0",
        description="RA.Aid's API provides REST endpoints for managing sessions and agents",
        routes=app.routes,
        license_info={
            "name": "Apache 2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
        },
        contact={
            "name": "RA.Aid Team",
            "url": "https://github.com/ai-christianson/RA.Aid",
        }
    )

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi



def run_server(host: str = "0.0.0.0", port: int = 1818):
    """Run the FastAPI server."""
    uvicorn.run(app, host=host, port=port)