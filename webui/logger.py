import logging
import sys
from pathlib import Path
from rich.logging import RichHandler
from datetime import datetime
from typing import Optional

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Configure logging
def setup_logger(
    name: str = "webui",
    log_format: Optional[str] = None,
    terminal_format: Optional[str] = None,
    file_format: Optional[str] = None
) -> logging.Logger:
    """Set up logger with both file and terminal handlers
    
    Args:
        name: Logger name
        log_format: Optional format to use for both handlers
        terminal_format: Optional format specific to terminal handler
        file_format: Optional format specific to file handler
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    default_terminal_format = "%(asctime)s [%(levelname)s] %(message)s"
    default_file_format = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
    
    terminal_formatter = logging.Formatter(
        log_format or terminal_format or default_terminal_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_formatter = logging.Formatter(
        log_format or file_format or default_file_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Terminal handler with rich formatting
    terminal_handler = RichHandler(
        rich_tracebacks=True,
        show_time=False,
        show_path=False
    )
    terminal_handler.setFormatter(terminal_formatter)
    terminal_handler.setLevel(logging.INFO)
    
    # File handler
    log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Add handlers
    logger.addHandler(terminal_handler)
    logger.addHandler(file_handler)
    
    return logger

# Create default logger instance
logger = setup_logger() 