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

import os
import sys
from pathlib import Path

# Add the scripts directory to the Python path
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.append(str(scripts_dir))

# Import the scripts you want to make available
from scripts import get_latest_session_usage

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
