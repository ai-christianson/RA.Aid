import queue
import logging
from typing import Any
from ra_aid.database.pydantic_models import TrajectoryModel  # Import TrajectoryModel

logger = logging.getLogger(__name__)

_broadcast_queue: queue.Queue | None = None

def set_broadcast_queue(queue_instance: queue.Queue):
    """Sets the global broadcast queue instance for this module."""
    global _broadcast_queue
    _broadcast_queue = queue_instance
    logger.info("Broadcast queue set in broadcast_sender.")

def send_broadcast(message: Any):
    """Wraps a message and puts it onto the WebSocket broadcast queue."""
    if _broadcast_queue is None:
        raise RuntimeError("Broadcast queue not initialized")

    message_type = 'unknown'
    if isinstance(message, TrajectoryModel):
        message_type = 'trajectory'
    # Future: Add more type checks here if needed
    else:
        try:
            message_type = type(message).__name__
        except Exception:
            pass # Keep 'unknown' if type name retrieval fails

    wrapper = {'type': message_type, 'payload': message}

    _broadcast_queue.put(wrapper)
    logger.debug(f"Wrapped message with type '{message_type}' put onto broadcast queue.")
