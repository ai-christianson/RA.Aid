import logging
import sys
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


def setup_logging(verbose: bool = False, pretty: bool = False) -> None:
    logger = logging.getLogger("ra_aid")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    if not logger.handlers:
        if pretty:
            handler = PrettyHandler()
        else:
            print("USING STREAM HANDLER LOGGER")
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
        logger.addHandler(handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(f"ra_aid.{name}" if name else "ra_aid")
