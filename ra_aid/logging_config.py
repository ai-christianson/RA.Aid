import logging
import sys
from typing import Optional

def setup_logging(verbose: bool = False) -> None:
    logger = logging.getLogger("ra_aid")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(f"ra_aid.{name}" if name else "ra_aid")
