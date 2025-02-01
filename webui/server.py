#!/usr/bin/env python3
import argparse
import asyncio
import shutil
import sys
from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Verify ra-aid is available
if not shutil.which("ra-aid"):
    print(
        "Error: ra-aid command not found. Please ensure it's installed and in your PATH"
    )
    sys.exit(1)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates
templates = Jinja2Templates(directory=Path(__file__).parent)


# Create a route for the root to serve index.html with port parameter
@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    """Serve the index.html file with port parameter."""
    return templates.TemplateResponse(
        "index.html", {"request": request, "server_port": request.url.port or 8080}
    )


# Mount static files for js and other assets
app.mount("/static", StaticFiles(directory=Path(__file__).parent), name="static")

# Store WebSocket connections

# Store active WebSocket connections
active_connections: List[WebSocket] = []


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print(f"New WebSocket connection from {websocket.client}")
    await websocket.accept()
    print("WebSocket connection accepted")
    active_connections.append(websocket)

    try:
        while True:
            print("Waiting for message...")
            message = await websocket.receive_json()
            print(f"Received message: {message}")

            if message["type"] == "request":
                print(f"Processing request: {message['content']}")
                # Notify client that streaming is starting
                await websocket.send_json({"type": "stream_start"})

                try:
                    # Run ra-aid with the request using -m flag and cowboy mode
                    cmd = ["ra-aid", "-m", message["content"], "--cowboy-mode"]
                    print(f"Executing command: {' '.join(cmd)}")

                    # Create subprocess
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    print(f"Process started with PID: {process.pid}")

                    # Read output and errors concurrently
                    async def read_stream(stream, is_error=False):
                        stream_type = "stderr" if is_error else "stdout"
                        print(f"Starting to read from {stream_type}")
                        while True:
                            line = await stream.readline()
                            if not line:
                                print(f"End of {stream_type} stream")
                                break

                            try:
                                decoded_line = line.decode().strip()
                                print(f"{stream_type} line: {decoded_line}")
                                if decoded_line:
                                    await websocket.send_json(
                                        {
                                            "type": "chunk",
                                            "chunk": {
                                                "tools" if is_error else "agent": {
                                                    "messages": [
                                                        {
                                                            "content": decoded_line,
                                                            "status": (
                                                                "error"
                                                                if is_error
                                                                else "info"
                                                            ),
                                                        }
                                                    ]
                                                }
                                            },
                                        }
                                    )
                            except Exception as e:
                                print(f"Error sending output: {e}")

                    # Create tasks for reading stdout and stderr
                    stdout_task = asyncio.create_task(read_stream(process.stdout))
                    stderr_task = asyncio.create_task(read_stream(process.stderr, True))

                    # Wait for both streams to complete
                    await asyncio.gather(stdout_task, stderr_task)

                    # Wait for process to complete
                    return_code = await process.wait()

                    if return_code != 0:
                        await websocket.send_json(
                            {
                                "type": "chunk",
                                "chunk": {
                                    "tools": {
                                        "messages": [
                                            {
                                                "content": f"Process exited with code {return_code}",
                                                "status": "error",
                                            }
                                        ]
                                    }
                                },
                            }
                        )

                    # Notify client that streaming is complete
                    await websocket.send_json(
                        {"type": "stream_end", "request": message["content"]}
                    )

                except Exception as e:
                    error_msg = f"Error executing ra-aid: {str(e)}"
                    print(error_msg)
                    await websocket.send_json(
                        {
                            "type": "chunk",
                            "chunk": {
                                "tools": {
                                    "messages": [
                                        {"content": error_msg, "status": "error"}
                                    ]
                                }
                            },
                        }
                    )

    except WebSocketDisconnect:
        print("WebSocket client disconnected")
        active_connections.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
        print("WebSocket connection cleaned up")


@app.get("/config")
async def get_config(request: Request):
    """Return server configuration including host and port."""
    return {"host": request.client.host, "port": request.scope.get("server")[1]}


def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Run the FastAPI server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RA.Aid Web Interface Server")
    parser.add_argument(
        "--port", type=int, default=8080, help="Port to listen on (default: 8080)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to listen on (default: 0.0.0.0)",
    )

    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
