"""WebUI package for RA.Aid."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from functools import wraps
import asyncio

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and emojis for different log levels"""
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[41m'   # Red background
    }
    EMOJIS = {
        'DEBUG': '🔍',
        'INFO': '📝',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🚨'
    }
    RESET = '\033[0m'

    def format(self, record):
        # Add emoji and color based on log level
        color = self.COLORS.get(record.levelname, '')
        emoji = self.EMOJIS.get(record.levelname, '')
        reset = self.RESET
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        
        # Add function name and line number
        func_info = f"{record.funcName}:{record.lineno}"
        
        # Format the message
        message = super().format(record)
        return f"{color}{emoji} [{timestamp}] {record.levelname:<8} | {func_info:<30} | {message}{reset}"

def setup_logger(name: str = "webui", level: str = "INFO") -> logging.Logger:
    """Set up logger with both file and terminal handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    terminal_formatter = ColoredFormatter(
        "%(message)s"  # The format is handled in the formatter class
    )
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Terminal handler
    terminal_handler = logging.StreamHandler(sys.stdout)
    terminal_handler.setFormatter(terminal_formatter)
    terminal_handler.setLevel(getattr(logging, level))
    
    # File handler
    log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(getattr(logging, level))
    
    # Add handlers
    logger.addHandler(terminal_handler)
    logger.addHandler(file_handler)
    
    return logger

def log_function(logger: logging.Logger = None):
    """Decorator to log function entry and exit"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger()
            
            func_name = func.__name__
            logger.debug(f"Entering {func_name}")
            try:
                result = await func(*args, **kwargs)
                logger.debug(f"Exiting {func_name}")
                return result
            except Exception as e:
                logger.error(f"Error in {func_name}: {str(e)}")
                raise
                
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger()
            
            func_name = func.__name__
            logger.debug(f"Entering {func_name}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Exiting {func_name}")
                return result
            except Exception as e:
                logger.error(f"Error in {func_name}: {str(e)}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Create default logger instance
logger = setup_logger()

# Import after logger setup to avoid circular imports
from webui.config import WebUIConfig, load_environment_status
from webui.socket_interface import SocketInterface

__all__ = [
    'WebUIConfig', 
    'load_environment_status', 
    'SocketInterface',
    'setup_logger',
    'log_function',
    'logger'
] 