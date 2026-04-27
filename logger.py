"""
logger.py — Centralised rotating logger.
Import `get_logger(__name__)` in every module.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

import config


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger with:
      • Console handler  — coloured level prefix, human-readable
      • File handler     — rotating, plain text, always DEBUG level for full trace
    Call once per module: `log = get_logger(__name__)`
    """
    os.makedirs(config.LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when the module is reloaded
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)  # Capture everything; handlers filter level

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ──────────────────────────────────────────────────────
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # ── Rotating file handler ────────────────────────────────────────────────
    fh = RotatingFileHandler(
        filename=config.LOG_FILE,
        maxBytes=config.LOG_MAX_MB * 1024 * 1024,
        backupCount=config.LOG_BACKUP,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
