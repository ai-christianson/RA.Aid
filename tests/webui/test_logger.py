import pytest
import logging
import os
from pathlib import Path
from datetime import datetime
from webui.logger import setup_logger, log_dir

@pytest.fixture
def clean_logs():
    """Clean up log files before and after tests"""
    # Get current day's log file
    log_file = log_dir / f"test_logger_{datetime.now().strftime('%Y%m%d')}.log"
    if log_file.exists():
        log_file.unlink()
    yield
    if log_file.exists():
        log_file.unlink()

def test_setup_logger(clean_logs):
    """Test logger setup and configuration"""
    logger = setup_logger('test_logger')
    
    # Verify logger configuration
    assert logger.name == 'test_logger'
    assert logger.level == logging.DEBUG
    
    # Verify handlers
    assert len(logger.handlers) == 2  # File and Stream handlers
    
    file_handler = next(h for h in logger.handlers if isinstance(h, logging.FileHandler))
    stream_handler = next(h for h in logger.handlers if isinstance(h, logging.StreamHandler))
    
    assert file_handler.level == logging.DEBUG
    assert stream_handler.level == logging.DEBUG  # Both handlers are set to DEBUG level
    
    # Test logging
    test_message = "Test log message"
    logger.info(test_message)
    
    # Verify log file contents
    log_file = log_dir / f"test_logger_{datetime.now().strftime('%Y%m%d')}.log"
    assert log_file.exists()
    
    with open(log_file, 'r') as f:
        log_content = f.read()
        assert test_message in log_content

def test_log_directory_creation():
    """Test log directory creation"""
    assert log_dir.exists()
    assert log_dir.is_dir()

def test_logger_formatting(clean_logs):
    """Test logger message formatting"""
    logger = setup_logger('test_format')
    test_message = "Test format message"
    logger.info(test_message)
    
    log_file = log_dir / f"test_format_{datetime.now().strftime('%Y%m%d')}.log"
    with open(log_file, 'r') as f:
        log_content = f.read()
        # Verify timestamp format
        assert datetime.now().strftime('%Y-%m-%d') in log_content  # Date
        assert ':' in log_content    # Time separator
        # Verify log level
        assert 'INFO' in log_content
        # Verify message
        assert test_message in log_content

def test_multiple_logger_instances():
    """Test multiple logger instances"""
    logger1 = setup_logger('test_logger1')
    logger2 = setup_logger('test_logger2')
    
    assert logger1.name != logger2.name
    assert len(logger1.handlers) == len(logger2.handlers)
    
    # Verify both loggers write to their respective files
    logger1.info("Message from logger1")
    logger2.info("Message from logger2")
    
    log_file1 = log_dir / f"test_logger1_{datetime.now().strftime('%Y%m%d')}.log"
    log_file2 = log_dir / f"test_logger2_{datetime.now().strftime('%Y%m%d')}.log"
    
    with open(log_file1, 'r') as f:
        log_content = f.read()
        assert "Message from logger1" in log_content
    
    with open(log_file2, 'r') as f:
        log_content = f.read()
        assert "Message from logger2" in log_content

def test_logger_levels(clean_logs):
    """Test different logging levels"""
    logger = setup_logger('test_levels')
    
    # Test all logging levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
    
    log_file = log_dir / f"test_levels_{datetime.now().strftime('%Y%m%d')}.log"
    with open(log_file, 'r') as f:
        log_content = f.read()
        
        # Debug should be logged in file (file handler level is DEBUG)
        assert "Debug message" in log_content
        
        # All other levels should be logged
        assert "Info message" in log_content
        assert "Warning message" in log_content
        assert "Error message" in log_content
        assert "Critical message" in log_content

def test_custom_formats():
    """Test logger with custom formats"""
    custom_format = "%(levelname)s - %(message)s"
    logger = setup_logger('test_custom', log_format=custom_format)
    
    test_message = "Custom format test"
    logger.info(test_message)
    
    log_file = log_dir / f"test_custom_{datetime.now().strftime('%Y%m%d')}.log"
    with open(log_file, 'r') as f:
        log_content = f.read()
        assert f"INFO - {test_message}" in log_content

def test_handler_reuse():
    """Test that loggers with same name reuse handlers"""
    logger1 = setup_logger('test_reuse')
    initial_handlers = len(logger1.handlers)
    
    # Get another logger with same name
    logger2 = setup_logger('test_reuse')
    
    # Should be the same logger instance
    assert logger1 is logger2
    # Should not add duplicate handlers
    assert len(logger2.handlers) == initial_handlers 