import asyncio
import json
from typing import Any, Dict, Optional, Callable
import websockets
from webui.logger import setup_logger

# Set up logger
logger = setup_logger("socket_interface")

class SocketInterface:
    """Interface for WebSocket communication with the server."""
    
    def __init__(self, url: str = "ws://localhost:8765"):
        """Initialize the socket interface."""
        self.url = url
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False
        self.handlers: Dict[str, Callable] = {}
        self._connection_lock = asyncio.Lock()
        self._connection_count = 0
    
    async def connect_server(self, max_retries: int = 3, retry_delay: float = 1.0) -> bool:
        """Connect to the WebSocket server with retries."""
        if self.connected:
            logger.warning("Already connected to server")
            return False  # Return False for concurrent connection attempts
        
        async with self._connection_lock:
            if self.connected:  # Double-check after acquiring lock
                return False
            
            for attempt in range(max_retries):
                try:
                    self.websocket = await websockets.connect(self.url)
                    self.connected = True
                    self._connection_count += 1
                    logger.info("Connected to server")
                    return True
                except Exception as e:
                    logger.error(f"Connection attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
            
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from the WebSocket server."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            self.connected = False
            self._connection_count = 0
            logger.info("Disconnected from server")
    
    async def send_task(self, task: str, config: Dict[str, Any]) -> bool:
        """Send a task to the server."""
        if not self.connected or not self.websocket:
            logger.error("Not connected to server")
            return False
        
        try:
            # Ensure cowboy_mode is properly set in the config
            if 'config' not in config:
                config['config'] = {}
            if 'cowboy_mode' not in config['config']:
                config['config']['cowboy_mode'] = config.get('cowboy_mode', False)
            
            message = {
                "type": "task",
                "content": task,
                "config": config
            }
            await self.websocket.send(json.dumps(message))
            logger.info("Task sent successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to send task: {str(e)}")
            return False
    
    def register_handler(self, event_type: str, handler: Callable) -> None:
        """Register an event handler."""
        self.handlers[event_type] = handler
        logger.debug(f"Registered handler for {event_type}")
    
    async def setup_handlers(self) -> None:
        """Set up event handlers for the WebSocket connection."""
        if not self.websocket:
            logger.error("WebSocket not connected")
            return
        
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    event_type = data.get("type")
                    if event_type in self.handlers:
                        await self.handlers[event_type](data)
                    else:
                        logger.warning(f"No handler for event type: {event_type}")
                except json.JSONDecodeError:
                    logger.error("Failed to decode message")
                except Exception as e:
                    logger.error(f"Error in message handler: {str(e)}")
        except Exception as e:
            logger.error(f"Error in event loop: {str(e)}")
    
    @property
    def connection_count(self) -> int:
        """Get the number of successful connections made."""
        return self._connection_count 