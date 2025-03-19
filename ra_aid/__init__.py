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

import importlib.util
import sys
from pathlib import Path

def import_script(script_name):
    """
    Dynamically import a script from the scripts directory.
    
    Args:
        script_name: Name of the script file without .py extension
        
    Returns:
        The imported module
    """
    script_path = Path(__file__).parent.parent / "scripts" / f"{script_name}.py"
    if not script_path.exists():
        raise ImportError(f"Cannot find script {script_name} at {script_path}")
        
    spec = importlib.util.spec_from_file_location(script_name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[script_name] = module
    spec.loader.exec_module(module)
    return module

# Dynamically import the get_latest_session_usage function
try:
    get_latest_session_usage = import_script("get_session_usage").get_latest_session_usage
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
