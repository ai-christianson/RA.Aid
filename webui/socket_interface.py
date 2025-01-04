"""WebSocket interface for real-time communication."""

import asyncio
import json
import websockets
from typing import Any, Dict, Optional, Callable
from websockets.exceptions import WebSocketException
from webui import logger, log_function

# Get logger for this module
socket_logger = logger.getChild("socket")

class SocketInterface:
    """Interface for WebSocket communication."""
    
    def __init__(self, url: str = "ws://localhost:8765"):
        """Initialize socket interface."""
        self.url = url
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.handlers: Dict[str, Callable] = {}
        self.connected = False
        socket_logger.info(f"Initialized SocketInterface with URL: {url}")
    
    @log_function(socket_logger)
    async def connect(self, max_retries: int = 3, retry_delay: int = 1) -> bool:
        """Connect to WebSocket server with retries."""
        for attempt in range(max_retries):
            try:
                self.websocket = await websockets.connect(self.url)
                self.connected = True
                socket_logger.info("Connected to WebSocket server")
                return True
            except WebSocketException as e:
                socket_logger.error(f"Connection attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    
        socket_logger.error(f"Failed to connect after {max_retries} attempts")
        return False
    
    @log_function(socket_logger)
    async def disconnect(self) -> None:
        """Disconnect from WebSocket server."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            self.connected = False
            socket_logger.info("Disconnected from WebSocket server")
    
    @log_function(socket_logger)
    async def send(self, message: Dict[str, Any]) -> bool:
        """Send message to WebSocket server."""
        if not self.connected or not self.websocket:
            socket_logger.error("Not connected to WebSocket server")
            return False
            
        try:
            await self.websocket.send(json.dumps(message))
            socket_logger.debug(f"Sent message: {message}")
            return True
        except WebSocketException as e:
            socket_logger.error(f"Failed to send message: {str(e)}")
            return False
    
    @log_function(socket_logger)
    def register_handler(self, message_type: str, handler: Callable) -> None:
        """Register handler for message type."""
        self.handlers[message_type] = handler
        socket_logger.info(f"Registered handler for {message_type}")
    
    @log_function(socket_logger)
    async def setup_handlers(self) -> None:
        """Set up message handlers."""
        if not self.connected or not self.websocket:
            socket_logger.error("Not connected to WebSocket server")
            return
            
        try:
            async for message in self.websocket:
                data = json.loads(message)
                message_type = data.get("type")
                socket_logger.debug(f"Received message: {data}")
                
                if message_type in self.handlers:
                    await self.handlers[message_type](data)
                else:
                    socket_logger.warning(f"No handler for message type: {message_type}")
        except WebSocketException as e:
            socket_logger.error(f"Error in message handler: {str(e)}")
            self.connected = False 