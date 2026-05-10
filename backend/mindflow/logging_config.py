"""Centralized logging configuration for MindFlow."""

import logging
import sys

from mindflow.config import settings

_logger: logging.Logger | None = None


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(level)

    if not name.startswith("mindflow"):
        logger = logger.getChild("mindflow")

    return logger
