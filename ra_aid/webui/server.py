"""Web interface server implementation for RA.Aid."""

import asyncio
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for more info
logger = logging.getLogger(__name__)

# Verify ra-aid is available
if not shutil.which("ra-aid"):
    logger.error(
        "ra-aid command not found. Please ensure it's installed and in your PATH"
    )
    sys.exit(1)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the directory containing static files
STATIC_DIR = Path(__file__).parent / "static"
if not STATIC_DIR.exists():
    logger.error(f"Static directory not found at {STATIC_DIR}")
    sys.exit(1)

logger.info(f"Using static directory: {STATIC_DIR}")

# Setup templates
templates = Jinja2Templates(directory=str(STATIC_DIR))


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> bool:
        try:
            logger.debug("Accepting WebSocket connection...")
            await websocket.accept()
            logger.debug("WebSocket connection accepted")
            self.active_connections.append(websocket)
            return True
        except Exception as e:
            logger.error(f"Error accepting WebSocket connection: {e}")
            return False

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_message(self, websocket: WebSocket, message: Dict[str, Any]):
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            await self.handle_error(websocket, str(e))

    async def handle_error(self, websocket: WebSocket, error_message: str):
        try:
            await websocket.send_json(
                {
                    "type": "chunk",
                    "chunk": {
                        "tools": {
                            "messages": [
                                {
                                    "content": f"Error: {error_message}",
                                    "status": "error",
                                }
                            ]
                        }
                    },
                }
            )
        except Exception as e:
            logger.error(f"Error sending error message: {e}")


manager = ConnectionManager()


@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    """Serve the index.html file with port parameter."""
    return templates.TemplateResponse(
        "index.html", {"request": request, "server_port": request.url.port or 8080}
    )


# Mount static files for js and other assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = id(websocket)
    logger.info(f"New WebSocket connection attempt from client {client_id}")

    if not await manager.connect(websocket):
        logger.error(f"Failed to accept WebSocket connection for client {client_id}")
        return

    logger.info(f"WebSocket connection accepted for client {client_id}")

    try:
        # Send initial connection success message
        await manager.send_message(
            websocket,
            {
                "type": "chunk",
                "chunk": {
                    "agent": {
                        "messages": [
                            {"content": "Connected to RA.Aid server", "status": "info"}
                        ]
                    }
                },
            },
        )

        while True:
            try:
                message = await websocket.receive_json()
                logger.debug(f"Received message from client {client_id}: {message}")

                if message["type"] == "request":
                    await manager.send_message(websocket, {"type": "stream_start"})

                    try:
                        # Run ra-aid with the request
                        cmd = ["ra-aid", "-m", message["content"], "--cowboy-mode"]
                        logger.info(f"Executing command: {' '.join(cmd)}")

                        process = await asyncio.create_subprocess_exec(
                            *cmd,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        logger.info(f"Process started with PID: {process.pid}")

                        async def read_stream(stream, is_error=False):
                            while True:
                                line = await stream.readline()
                                if not line:
                                    break

                                try:
                                    decoded_line = line.decode().strip()
                                    if decoded_line:
                                        await manager.send_message(
                                            websocket,
                                            {
                                                "type": "chunk",
                                                "chunk": {
                                                    "tools" if is_error else "agent": {
                                                        "messages": [
                                                            {
                                                                "content": decoded_line,
                                                                "status": "error"
                                                                if is_error
                                                                else "info",
                                                            }
                                                        ]
                                                    }
                                                },
                                            },
                                        )
                                except Exception as e:
                                    logger.error(f"Error processing output: {e}")

                        # Create tasks for reading stdout and stderr
                        stdout_task = asyncio.create_task(read_stream(process.stdout))
                        stderr_task = asyncio.create_task(
                            read_stream(process.stderr, True)
                        )

                        # Wait for both streams to complete
                        await asyncio.gather(stdout_task, stderr_task)

                        # Wait for process to complete
                        return_code = await process.wait()

                        if return_code != 0:
                            await manager.handle_error(
                                websocket, f"Process exited with code {return_code}"
                            )

                        await manager.send_message(
                            websocket,
                            {"type": "stream_end", "request": message["content"]},
                        )

                    except Exception as e:
                        logger.error(f"Error executing ra-aid: {e}")
                        await manager.handle_error(websocket, str(e))

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await manager.handle_error(websocket, str(e))

    except WebSocketDisconnect:
        logger.info(f"WebSocket client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
    finally:
        manager.disconnect(websocket)
        logger.info(f"WebSocket connection cleaned up for client {client_id}")


def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Run the FastAPI server."""
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="debug",
        ws_max_size=16777216,  # 16MB
        timeout_keep_alive=0,  # Disable keep-alive timeout
    )
