import logging
import sys

# Create a logger
logger = logging.getLogger('ra_aid')
logger.setLevel(logging.INFO)

# Create console handler with a higher log level
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create file handler which logs even debug messages
file_handler = logging.FileHandler('ra_aid.log')
file_handler.setLevel(logging.DEBUG)

# Create formatters and add them to the handlers
console_formatter = logging.Formatter('%(levelname)s: %(message)s')
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
file_handler.setFormatter(file_formatter)

# Add the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Export the logger
__all__ = ['logger'] 