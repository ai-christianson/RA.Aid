import pytest
from unittest.mock import patch, MagicMock
import logging
import sys
from webui.logger import setup_logger, ColoredFormatter

def test_setup_logger():
    """Test logger setup and configuration"""
    # Capture log output
    with patch('logging.StreamHandler') as mock_handler:
        mock_handler_instance = MagicMock()
        mock_handler.return_value = mock_handler_instance
        
        logger = setup_logger('test_logger')
        
        # Verify logger was created with correct level
        assert logger.level == logging.INFO
        
        # Verify handler was added
        assert len(logger.handlers) == 1
        mock_handler.assert_called_once_with(sys.stdout)
        
        # Verify formatter was set
        mock_handler_instance.setFormatter.assert_called_once()
        formatter = mock_handler_instance.setFormatter.call_args[0][0]
        assert isinstance(formatter, ColoredFormatter)

def test_log_formatting():
    """Test custom log formatting"""
    # Create a formatter directly
    formatter = ColoredFormatter("%(asctime)s [%(levelname)s] %(message)s")
    
    # Create a test record
    record = logging.LogRecord(
        name='test_logger',
        level=logging.INFO,
        pathname='test.py',
        lineno=42,
        msg='Test message',
        args=None,
        exc_info=None
    )
    
    # Test formatting
    formatted = formatter.format(record)
    assert '📝' in formatted  # INFO emoji
    assert 'Test message' in formatted
    assert 'INFO' in formatted

def test_log_level_colors():
    """Test color coding for different log levels"""
    formatter = ColoredFormatter("%(asctime)s [%(levelname)s] %(message)s")
    
    # Test different log levels
    levels = {
        'DEBUG': '🔍',
        'INFO': '📝',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🚨'
    }
    
    for level_name, emoji in levels.items():
        record = logging.LogRecord(
            name='test_logger',
            level=getattr(logging, level_name),
            pathname='test.py',
            lineno=42,
            msg=f'{level_name} message',
            args=None,
            exc_info=None
        )
        formatted = formatter.format(record)
        assert emoji in formatted
        assert level_name in formatted
        assert f'{level_name} message' in formatted

def test_logger_initialization():
    """Test logger initialization with existing handlers"""
    # Create a logger with existing handlers
    existing_logger = logging.getLogger('test_logger')
    existing_logger.addHandler(logging.NullHandler())
    
    # Setup logger should remove existing handlers
    logger = setup_logger('test_logger')
    assert len(logger.handlers) == 1
    assert not isinstance(logger.handlers[0], logging.NullHandler)
