#!/usr/bin/env python3
import asyncio
import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import AsyncGenerator, Callable

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse
import uvicorn

# Configure module-specific logging without affecting root logger
logger = logging.getLogger(__name__)
# Only configure this specific logger, not the root logger
if not logger.handlers:  # Avoid adding handlers multiple times
    logger.setLevel(logging.INFO) # Changed level to INFO to see connection logs
    handler = logging.StreamHandler(sys.__stderr__)  # Use the real stderr
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s") # Added name to formatter
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # Prevent propagation to avoid affecting the root logger configuration
    logger.propagate = False

# Add project root to Python path - Ensure this runs before other project imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ra_aid.database.pydantic_models import TrajectoryModel
# Import get_trajectory_repository as well
from ra_aid.database.repositories.trajectory_repository import TrajectoryRepository, get_trajectory_repository
from ra_aid.server.api_v1_sessions import router as sessions_router
from ra_aid.server.api_v1_spawn_agent import router as spawn_agent_router
from ra_aid.server.connection_manager import ConnectionManager

# Global variable to hold the app instance, needed by the hook
# This is a simple way, alternatives include passing app context differently.
_app_instance: FastAPI = None

async def broadcast_consumer(queue: asyncio.Queue[TrajectoryModel], manager: ConnectionManager):
    """Consumes trajectories from the queue and broadcasts them."""
    while True:
        try:
            trajectory = await queue.get()
            logger.debug(f"Broadcasting trajectory: {trajectory.trajectory_id}")
            message = trajectory.model_dump_json()
            await manager.broadcast(message)
            queue.task_done()
        except asyncio.CancelledError:
            logger.info("Broadcast consumer task cancelled.")
            break
        except Exception:
            logger.exception("Error in broadcast consumer task.")
            # Avoid breaking the loop on non-cancellation errors

def trajectory_create_hook(trajectory: TrajectoryModel):
    """Hook function called after a trajectory is created."""
    global _app_instance
    if _app_instance and hasattr(_app_instance.state, 'loop') and hasattr(_app_instance.state, 'broadcast_queue'):
        loop = _app_instance.state.loop
        queue = _app_instance.state.broadcast_queue
        if loop and queue:
            try:
                # Use call_soon_threadsafe as this hook might be called from a different thread
                loop.call_soon_threadsafe(queue.put_nowait, trajectory)
                logger.debug(f"Queued trajectory for broadcast: {trajectory.trajectory_id}")
            except Exception:
                logger.exception(f"Failed to queue trajectory {trajectory.trajectory_id} for broadcast.")
        else:
             logger.error("Event loop or broadcast queue not found in app state for hook.")
    else:
        logger.error("App instance or state not available for trajectory create hook.")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages application startup and shutdown events."""
    global _app_instance
    _app_instance = app # Store app instance globally for the hook

    logger.info("Application startup: Initializing resources.")
    # Store the running event loop
    app.state.loop = asyncio.get_running_loop()
    # Create the broadcast queue
    app.state.broadcast_queue = asyncio.Queue()
    # Instantiate the ConnectionManager
    app.state.connection_manager = ConnectionManager()
    # Start the consumer task
    app.state.broadcast_task = asyncio.create_task(
        broadcast_consumer(app.state.broadcast_queue, app.state.connection_manager)
    )

    # Register the trajectory creation hook on the specific instance
    try:
        # Assuming lifespan runs within the context where the repo is set up by launch_server
        trajectory_repo = get_trajectory_repository()
        if trajectory_repo: # Check if repo was successfully retrieved
            trajectory_repo.register_create_hook(trajectory_create_hook)
            logger.info("Trajectory broadcast hook registered on repository instance.")
        else:
             logger.error("Failed to get TrajectoryRepository instance for hook registration (returned None).")
    except RuntimeError as e:
        # This might happen if the contextvar is not set when lifespan runs
        logger.error(f"Failed to get TrajectoryRepository instance for hook registration (RuntimeError): {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during hook registration: {e}")

    yield  # Application is running

    logger.info("Application shutdown: Cleaning up resources.")
    # Unregister hook (optional, good practice if hooks can be removed)
    # Requires an unregister_create_hook instance method on TrajectoryRepository
    try:
        trajectory_repo = get_trajectory_repository()
        if trajectory_repo and hasattr(trajectory_repo, 'unregister_create_hook'):
            trajectory_repo.unregister_create_hook(trajectory_create_hook)
            logger.info("Trajectory broadcast hook unregistered from repository instance.")
    except RuntimeError as e:
        logger.warning(f"Could not get TrajectoryRepository instance during shutdown to unregister hook: {e}")
    except Exception as e:
        logger.exception("Error during trajectory hook unregistration.")


    # Cancel the consumer task
    if hasattr(app.state, 'broadcast_task') and app.state.broadcast_task:
        app.state.broadcast_task.cancel()
        try:
            await asyncio.wait_for(app.state.broadcast_task, timeout=5.0)
            logger.info("Broadcast consumer task successfully cancelled.")
        except asyncio.TimeoutError:
            logger.warning("Broadcast consumer task did not cancel within timeout.")
        except asyncio.CancelledError:
             logger.info("Broadcast consumer task already cancelled.") # Expected case
        except Exception:
            logger.exception("Error during broadcast consumer task cancellation.")

    # Clear global reference
    _app_instance = None
    logger.info("Application shutdown complete.")


# Initialize FastAPI app with lifespan management
app = FastAPI(
    title="RA.Aid API",
    description="API for RA.Aid - AI Programming Assistant",
    version="1.0.0", # Consider fetching version dynamically
    lifespan=lifespan, # Add the lifespan context manager
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(sessions_router)
app.include_router(spawn_agent_router)

# Directory for the current file
CURRENT_DIR = Path(__file__).parent

# WebSocket API endpoint using the ConnectionManager from app state
@app.websocket("/v1/ws")
async def websocket_endpoint(websocket: WebSocket):
    manager: ConnectionManager = websocket.app.state.connection_manager
    await manager.connect(websocket)
    logger.info(f"WebSocket client connected: {websocket.client}")
    try:
        # Keep the connection alive and detect disconnects by waiting for messages
        # This loop does nothing with the received data for now, just keeps the connection open
        while True:
            data = await websocket.receive_text()
            # logger.debug(f"Received message from {websocket.client}: {data}") # Optional: Log received messages
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {websocket.client}")
    except Exception as e:
         logger.error(f"WebSocket error for client {websocket.client}: {e}")
    finally:
        # Ensure disconnection happens even if errors occur within the loop
        manager.disconnect(websocket)
        logger.info(f"WebSocket connection closed for client: {websocket.client}")


@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    """Return basic info about RA.Aid API."""
    # Same HTML content as before...
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
                <p>A WebSocket API is available at /v1/ws for real-time updates (e.g., trajectory events).</p>
                <p>See the <a href="/docs">API documentation</a> for more information on REST endpoints.</p>
            </body>
        </html>
        """
    )


@app.get("/config")
async def get_config_endpoint(request: Request): # Renamed to avoid conflict with imported get_config
    """Return server configuration including host and port."""
    # Use the imported get_config function if needed, or just return client info
    # app_config = get_config() # Example if you needed config values
    return {"host": request.client.host, "port": request.scope.get("server")[1]}

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="RA.Aid API",
        summary="RA.Aid API OpenAPI Spec",
        version="1.0.0", # Consider dynamic version
        description="RA.Aid's API provides REST endpoints for managing sessions and agents, and a WebSocket endpoint (/v1/ws) for real-time communication of events like new trajectories.",
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
    # Modify schema if needed, e.g., add WebSocket info manually if desired
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi


def run_server(host: str = "0.0.0.0", port: int = 1818):
    """Run the FastAPI server."""
    # Note: Reload=True might cause issues with lifespan state in some complex scenarios.
    # It's generally fine for development but be mindful.
    uvicorn.run("ra_aid.server.server:app", host=host, port=port, reload=True, log_level="info")

if __name__ == "__main__":
    # This allows running the server directly for testing
    run_server()
