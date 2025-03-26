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
            from ra_aid.console.formatting import cpm
            cpm(msg.strip(), title=title, border_style=style)
        except Exception:
            self.handleError(record)


def setup_logging(log_mode: str = "file", pretty: bool = False, log_level: Optional[str] = None, base_dir: Optional[str] = None) -> None:
    """
    Configure logging for ra_aid.

    Args:
        log_mode: Determines where logs are output. Options:
            - "file": Log only to file at the specified log_level. Console logging is disabled.
            - "console": Log to console at the specified log_level (no file logging).
        pretty: Set to True to enable pretty console logging.
        log_level: Optional explicit log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                  When log_mode="file": Affects file logging level. Console logging is disabled.
                  When log_mode="console": Controls the console logging level.

    Console logging behavior:
    - With log_mode="file": No messages are shown in console.
    - With log_mode="console": Console shows messages at the requested log_level.

    File logging behavior:
    - Only active when log_mode="file".
    - Uses the requested log_level.
    """
    # Create logs directory if it doesn't exist
    if base_dir:
        # Use the provided directory directly
        ra_aid_dir_str = base_dir
    else:
        # Use .ra-aid in the current working directory
        cwd = os.getcwd()
        ra_aid_dir_str = os.path.join(cwd, ".ra-aid")
    logs_dir_str = os.path.join(ra_aid_dir_str, "logs")

    # Create directory structure if log_mode is "file"
    if log_mode == "file":
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
        specified_log_level = getattr(logging, log_level.upper(), None)
        if not isinstance(specified_log_level, int):
            # If invalid log level is provided, fall back to default
            print(f"Invalid log level: {log_level}")
            specified_log_level = logging.WARNING
    else:
        # No log_level specified, use WARNING as default
        specified_log_level = logging.WARNING

    # Configure the root logger
    root_logger = logging.getLogger()

    # Always set the root logger to DEBUG level
    # This ensures all messages flow through to their respective handlers
    # Best practice is to set root logger to lowest level and let handlers control message filtering
    root_logger.setLevel(logging.DEBUG)

    # Clear existing handlers from root logger to avoid duplicates
    if root_logger.handlers:
        root_logger.handlers.clear()

    # Create and add console handler only if log_mode is "console"
    if log_mode == "console":
        if pretty:
            console_handler = PrettyHandler()
        else:
            console_handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(formatter)

        # Set console handler log level
        console_handler.setLevel(specified_log_level)

        # Add console handler to root logger
        root_logger.addHandler(console_handler)

    # Create and add file handler only when log_mode is "file"
    elif log_mode == "file":
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
            # File handler always uses the specified log level
            file_handler.setLevel(specified_log_level)

            # Add file handler to root logger
            root_logger.addHandler(file_handler)

            # Create an ra_aid logger for compatibility
            logger = logging.getLogger("ra_aid")
            logger.setLevel(logging.DEBUG)
            logger.propagate = True  # Let messages propagate to root handlers

            # Log configuration details for debugging (to the file)
            logger.debug(f"Logging configuration: log_mode={log_mode}, log_level={log_level}, "                         f"root_level={root_logger.level}, logger_level={logger.level}, "                        f"file_level={file_handler.level}, "                        f"propagate={logger.propagate}")

            logger.info(f"Log file created: {log_filename}")
        except Exception as e:
            # If file logging fails, try to log to stderr as a fallback
            print(f"CRITICAL: Failed to set up file logging: {str(e)}", file=sys.stderr)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(f"ra_aid.{name}" if name else "ra_aid")
