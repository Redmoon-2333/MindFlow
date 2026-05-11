"""Centralized logging configuration for MindFlow."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from mindflow.config import settings


def _ensure_log_dir():
    log_dir = Path(__file__).resolve().parents[2] / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        try:
            log_dir = _ensure_log_dir()
            file_handler = RotatingFileHandler(
                str(log_dir / "mindflow.log"),
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception:
            pass

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(level)

    if not name.startswith("mindflow"):
        logger = logger.getChild("mindflow")

    return logger
