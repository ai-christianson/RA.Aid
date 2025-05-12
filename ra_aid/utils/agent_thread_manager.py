import threading
from typing import Dict, Any
from ra_aid.logging_config import get_logger

logger = get_logger(__name__)

# Global registry mapping session_id -> { thread, stop_event }
agent_thread_registry: Dict[int, Dict[str, Any]] = {}
# Lock to protect access to the registry
_registry_lock = threading.RLock()

def register_agent(session_id: int, thread: threading.Thread, stop_event: threading.Event) -> None:
    """
    Register a new agent thread and its associated stop_event for a session.
    """
    logger.info(f"Registering agent for session_id {session_id}; thread_name {thread.name}")
    with _registry_lock:
        agent_thread_registry[session_id] = {
            "thread": thread,
            "stop_event": stop_event
        }

def get_session_id_by_thread_name(thread_name: str) -> int | None:
    """
    Retrieve the session_id associated with a given thread_id.
    """
    logger.info(f"Retrieving session_id for thread_id {thread_name}")
    with _registry_lock:
        for session_id, entry in agent_thread_registry.items():
            # Check if the thread name matches
            logger.info(f"Comparing thread_name {entry['thread'].name}")
            if entry["thread"].name == thread_name:
                logger.info(f"Found session_id {session_id} for thread_id {thread_name}")
                return session_id
    logger.warning(f"No session_id found for thread_id {thread_name}")

    # none for not found
    return None


def unregister_agent(session_id: int) -> None:
    """
    Remove an agent entry from the registry once it has finished or is no longer needed.
    """
    logger.info(f"Unregistering agent for session_id {session_id}")
    with _registry_lock:
        if session_id in agent_thread_registry:
            removed_entry = agent_thread_registry.pop(session_id, None)
            if removed_entry:
                logger.info(f"Successfully unregistered agent for session_id {session_id}, thread: {removed_entry['thread'].name}")
        else:
            logger.warning(f"Attempted to unregister non-existent agent for session_id {session_id}")


def stop_agent(session_id: int) -> bool:
    logger.info(f"Requested to stop agent for session_id {session_id}")
    with _registry_lock:
        entry = agent_thread_registry.get(session_id)
    if entry and "stop_event" in entry:
        entry["stop_event"].set()
        return True
    logger.warning(f"No agent thread found for session_id {session_id}")
    return False


def is_agent_running(session_id: int) -> bool:
    logger.info(f"Checking if agent is running for session_id {session_id}")
    with _registry_lock:
        entry = agent_thread_registry.get(session_id)
    return bool(entry and entry["thread"].is_alive())


def has_received_stop_signal(session_id: int) -> bool:
    logger.info(f"Checking if agent has received stop signal for session_id {session_id}")
    with _registry_lock:
        entry = agent_thread_registry.get(session_id)
    return bool(entry and entry["stop_event"].is_set())
