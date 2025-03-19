from .__version__ import __version__
from .agent_utils import run_agent_with_retry
from .console.formatting import (
    print_error,
    print_interrupt,
    print_stage_header,
    print_task_header,
)
from .console.output import print_agent_output
from .text.processing import truncate_output

import sys
from pathlib import Path

# Add the parent directory of ra_aid to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Now you can import from scripts
try:
    from scripts.get_session_usage import get_latest_session_usage
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"Could not import get_latest_session_usage: {e}")
    # Provide a fallback implementation that returns an error
    def get_latest_session_usage():
        return {"error": "Script not available in this installation"}, 1

__all__ = [
    "print_stage_header",
    "print_task_header",
    "print_agent_output",
    "truncate_output",
    "print_error",
    "print_interrupt",
    "run_agent_with_retry",
    "__version__",
    "get_latest_session_usage",
]
