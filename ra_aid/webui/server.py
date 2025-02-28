#!/usr/bin/env python3
import asyncio
import logging
import os
import queue
import sys
import threading
import traceback
from pathlib import Path
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.__stderr__)  # Use the real stderr
    ],
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates and static files directories
CURRENT_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=CURRENT_DIR)

# Mount static files for js and other assets
static_dir = CURRENT_DIR / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Store active WebSocket connections
active_connections: List[WebSocket] = []


def run_ra_aid(message_content, output_queue):
    """Run ra-aid in a separate thread"""
    try:
        import ra_aid.__main__

        logger.info("Successfully imported ra_aid.__main__")

        # Override sys.argv
        sys.argv = [sys.argv[0], "-m", message_content, "--cowboy-mode"]
        logger.info(f"Set sys.argv to: {sys.argv}")

        # Create custom output handler
        class QueueHandler:
            def __init__(self, queue):
                self.queue = queue
                self.buffer = []
                self.box_start = False
                self._real_stderr = sys.__stderr__

            def write(self, text):
                # Always log raw output for debugging
                logger.debug(f"Raw output: {repr(text)}")

                # Check if this is a box drawing character
                if any(c in text for c in "╭╮╰╯│─"):
                    self.box_start = True
                    self.buffer.append(text)
                elif self.box_start and text.strip():
                    self.buffer.append(text)
                    if "╯" in text:  # End of box
                        full_text = "".join(self.buffer)
                        # Extract content from inside the box
                        lines = full_text.split("\n")
                        content_lines = []
                        for line in lines:
                            # Remove box characters and leading/trailing spaces
                            clean_line = line.strip("╭╮╰╯│─ ")
                            if clean_line:
                                content_lines.append(clean_line)
                        if content_lines:
                            self.queue.put("\n".join(content_lines))
                        self.buffer = []
                        self.box_start = False
                elif not self.box_start and text.strip():
                    self.queue.put(text.strip())

            def flush(self):
                if self.buffer:
                    full_text = "".join(self.buffer)
                    # Extract content from partial box
                    lines = full_text.split("\n")
                    content_lines = []
                    for line in lines:
                        # Remove box characters and leading/trailing spaces
                        clean_line = line.strip("╭╮╰╯│─ ")
                        if clean_line:
                            content_lines.append(clean_line)
                    if content_lines:
                        self.queue.put("\n".join(content_lines))
                    self.buffer = []
                    self.box_start = False

        # Replace stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        queue_handler = QueueHandler(output_queue)
        sys.stdout = queue_handler
        sys.stderr = queue_handler

        try:
            logger.info("Starting ra_aid.main()")
            ra_aid.__main__.main()
            logger.info("Finished ra_aid.main()")
        except SystemExit:
            logger.info("Caught SystemExit - this is normal")
        except Exception as e:
            logger.error(f"Error in main: {str(e)}")
            traceback.print_exc(file=sys.__stderr__)
        finally:
            # Flush any remaining output
            queue_handler.flush()
            # Restore stdout and stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    except Exception as e:
        logger.error(f"Error in thread: {str(e)}")
        traceback.print_exc(file=sys.__stderr__)
        output_queue.put(f"Error: {str(e)}")


@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    """Serve the index.html file with port parameter."""
    return templates.TemplateResponse(
        "index.html", {"request": request, "server_port": request.url.port or 8080}
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")
    active_connections.append(websocket)

    try:
        while True:
            message = await websocket.receive_json()
            logger.info(f"Received message: {message}")

            if message["type"] == "request":
                content = message["content"]
                logger.info(f"Processing request: {content}")

                # Create queue for output
                output_queue = queue.Queue()

                # Create and start thread
                thread = threading.Thread(
                    target=run_ra_aid, args=(content, output_queue)
                )
                thread.start()

                try:
                    # Send stream start
                    await websocket.send_json({"type": "stream_start"})

                    while thread.is_alive() or not output_queue.empty():
                        try:
                            # Get output with timeout to allow checking thread status
                            line = output_queue.get(timeout=0.1)
                            if line and line.strip():  # Only send non-empty messages
                                logger.debug(f"WebSocket sending: {repr(line)}")
                                await websocket.send_json(
                                    {
                                        "type": "chunk",
                                        "chunk": {
                                            "agent": {
                                                "messages": [
                                                    {
                                                        "content": line.strip(),
                                                        "status": "info",
                                                    }
                                                ]
                                            }
                                        },
                                    }
                                )
                        except queue.Empty:
                            await asyncio.sleep(0.1)
                        except Exception as e:
                            logger.error(f"WebSocket error: {e}")
                            traceback.print_exc(file=sys.__stderr__)

                    # Wait for thread to finish
                    thread.join()
                    logger.info("Thread finished")

                    # Send stream end
                    await websocket.send_json({"type": "stream_end"})
                    logger.info("Sent stream_end message")

                except Exception as e:
                    error_msg = f"Error running ra-aid: {str(e)}"
                    logger.error(error_msg)
                    await websocket.send_json({"type": "error", "message": error_msg})

            logger.info("Waiting for message...")

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
        active_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        traceback.print_exc()
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
        logger.info("WebSocket connection closed")


@app.get("/config")
async def get_config(request: Request):
    """Return server configuration including host and port."""
    return {"host": request.client.host, "port": request.scope.get("server")[1]}


def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Run the FastAPI server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

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
