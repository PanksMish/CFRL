"""
utils/logger.py
----------------
Centralised logging configuration for CFRL-FND.

Provides:
    - get_logger(name)  : Return a named logger with pre-configured formatting.
    - setup_logging()   : Configure root logger with file + console handlers.
"""

import logging
import os
import sys
from typing import Optional

from config import cfg


def setup_logging(
    log_dir:  str           = cfg.paths.LOGS_DIR,
    log_file: str           = "cfrl_fnd.log",
    level:    int           = logging.INFO,
) -> None:
    """
    Configure the root logger with:
        - StreamHandler (console) at INFO level.
        - FileHandler   (file)    at DEBUG level.

    Args:
        log_dir  : Directory for log files.
        log_file : Log filename.
        level    : Minimum logging level for the console handler.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove any existing handlers (avoid duplicate logs in notebooks)
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(fmt)
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)

    root_logger.info("Logging initialised. Log file: %s", log_path)


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Return a named logger.

    Args:
        name  : Logger name (typically __name__ of the calling module).
        level : Optional override for this logger's level.

    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger
