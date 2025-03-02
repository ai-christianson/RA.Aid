import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel


class PrettyHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        self.console = Console()

    def emit(self, record):
        try:
            msg = self.format(record)
            # Determine title and style based on log level
            if record.levelno >= logging.CRITICAL:
                title = "🔥 CRITICAL"
                style = "bold red"
            elif record.levelno >= logging.ERROR:
                title = "❌ ERROR"
                style = "red"
            elif record.levelno >= logging.WARNING:
                title = "⚠️ WARNING"
                style = "yellow"
            elif record.levelno >= logging.INFO:
                title = "ℹ️ INFO"
                style = "green"
            else:
                title = "🐞 DEBUG"
                style = "blue"
            self.console.print(Panel(Markdown(msg.strip()), title=title, style=style))
        except Exception:
            self.handleError(record)


def setup_logging(verbose: bool = False, pretty: bool = False, log_level: Optional[str] = None) -> None:
    """
    Configure logging for ra_aid.
    
    Args:
        verbose: Set to True to enable verbose logging (implies DEBUG level if log_level not specified)
        pretty: Set to True to enable pretty console logging
        log_level: Optional explicit log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                  Takes precedence over verbose flag if specified
    """
    # Create .ra-aid/logs directory if it doesn't exist
    cwd = os.getcwd()
    ra_aid_dir_str = os.path.join(cwd, ".ra-aid")
    logs_dir_str = os.path.join(ra_aid_dir_str, "logs")
    
    # Create directory structure
    for directory in [ra_aid_dir_str, logs_dir_str]:
        path = Path(directory)
        if not path.exists():
            try:
                path.mkdir(mode=0o755, parents=True, exist_ok=True)
            except Exception as e:
                print(f"Warning: Failed to create log directory {directory}: {str(e)}")
    
    # Determine log level
    if log_level is not None:
        # Use provided log level if specified (case-insensitive)
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            # If invalid log level is provided, fall back to default
            print(f"Invalid log level: {log_level}")
            numeric_level = logging.DEBUG if verbose else logging.WARNING
    else:
        # Use verbose flag to determine log level
        numeric_level = logging.DEBUG if verbose else logging.WARNING
    
    # Configure root logger
    logger = logging.getLogger("ra_aid")
    logger.setLevel(numeric_level)
    
    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    # Create console handler
    if pretty:
        console_handler = PrettyHandler()
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    # Create file handler with rotation
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(logs_dir_str, f"ra_aid_{timestamp}.log")
        
        # RotatingFileHandler with 5MB max size and 100 backup files
        file_handler = RotatingFileHandler(
            log_filename,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=100,
            encoding="utf-8"
        )
        
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(numeric_level)
        
        # Add file handler to logger
        logger.addHandler(file_handler)
        
        logger.info(f"Log file created: {log_filename}")
    except Exception as e:
        logger.error(f"Failed to set up file logging: {str(e)}")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(f"ra_aid.{name}" if name else "ra_aid")