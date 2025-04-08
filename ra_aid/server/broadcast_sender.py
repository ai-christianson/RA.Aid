import queue
import logging
from typing import Any

logger = logging.getLogger(__name__)

_broadcast_queue: queue.Queue | None = None

def set_broadcast_queue(queue_instance: queue.Queue):
    """Sets the global broadcast queue instance for this module."""
    global _broadcast_queue
    _broadcast_queue = queue_instance
    logger.info("Broadcast queue set in broadcast_sender.")

def send_broadcast(message: Any):
    """Puts a message onto the WebSocket broadcast queue."""
    if _broadcast_queue is None:
        raise RuntimeError("Broadcast queue not initialized")

    _broadcast_queue.put(message)
    logger.debug(f"Message of type {type(message)} put onto broadcast queue.")