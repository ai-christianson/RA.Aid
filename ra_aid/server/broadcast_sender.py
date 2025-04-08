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
    """Puts a message onto the WebSocket broadcast queue.

    If the message is a dictionary containing 'type' and 'payload' keys,
    it is put directly onto the queue. Otherwise, it determines the type,
    wraps the message in a standard structure {'type': ..., 'payload': ...},
    and then puts it onto the queue.
    """
    if _broadcast_queue is None:
        raise RuntimeError("Broadcast queue not initialized")

    # Check if the message is already structured
    if isinstance(message, dict) and 'type' in message and 'payload' in message:
        _broadcast_queue.put(message)
        # Use f-string for cleaner logging and access the type safely
        msg_type = message.get('type', 'unknown')
        logger.debug(f"Pre-structured message with type '{msg_type}' put directly onto broadcast queue.")
        return  # Exit early as the message is already processed

    # Original logic for non-pre-structured messages
    message_type = 'unknown'
    if isinstance(message, TrajectoryModel):
        message_type = 'trajectory'
    else:
        try:
            # Use type(message).__name__ for better type identification
            message_type = type(message).__name__
        except Exception:
            # Keep 'unknown' if type name retrieval fails
            pass

    wrapper = {'type': message_type, 'payload': message}
    _broadcast_queue.put(wrapper)
    logger.debug(f"Wrapped message with type '{message_type}' put onto broadcast queue.")
