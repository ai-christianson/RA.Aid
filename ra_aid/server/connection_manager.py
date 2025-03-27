\
import logging
from typing import List

from fastapi import WebSocket

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        try:
            self.active_connections.remove(websocket)
        except ValueError:
            # WebSocket might already be removed, ignore
            pass

    async def broadcast(self, message: str):
        # Iterate over a copy in case disconnect happens during broadcast
        for connection in self.active_connections[:]:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Failed to send message to client {connection.client}: {e}")
                # Disconnect logic should handle cleanup, don't remove here
