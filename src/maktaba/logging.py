"""Lightweight logging utilities for Maktaba."""

import logging
import os
from typing import Optional


def get_logger(name: str = "maktaba", level: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    lvl = level or os.getenv("MAKTABA_LOG_LEVEL", "INFO")
    logger.setLevel(getattr(logging, lvl.upper() if lvl else "INFO", logging.INFO))
    logger.propagate = False
    return logger


def set_level(level: str) -> None:
    logging.getLogger("maktaba").setLevel(getattr(logging, level.upper(), logging.INFO))

