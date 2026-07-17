"""Loguru logging configuration.

Provides structured logging with:
  - Console output (development-friendly)
  - File output with rotation (10 MB), retention (30 days), and gzip compression
  - JSON format option for production/ELK consumption
  - Request ID binding via context

All log paths live under platformdirs user data directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import loguru
import platformdirs

from mindflow.config import Settings


def setup_logging(settings: Settings) -> None:
    """Configure loguru with console and rotating file handlers.

    Removes the default loguru handler and replaces it with:
      1. stderr console handler with colorized format
      2. Rotating file handler with configurable size, retention, and compression

    Args:
        settings: Application settings containing log configuration.
    """
    loguru.logger.remove()

    log_dir = Path(platformdirs.user_data_dir("mindflow", ensure_exists=True)) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Console handler — colorized, development-friendly
    loguru.logger.add(
        sys.stderr,
        level=settings.log.level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,
        diagnose=False,  # Don't expose local variables in tracebacks
    )

    # File handler — text format, rotated
    loguru.logger.add(
        log_dir / "mindflow_{time:YYYY-MM-DD}.log",
        level=settings.log.level,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} | {message}"
        ),
        rotation=settings.log.rotation,
        retention=settings.log.retention,
        compression=settings.log.compression,
        backtrace=True,
        diagnose=False,
        serialize=settings.log.json_format,
    )

    # JSON-structured handler (separate file, shorter retention)
    if settings.log.json_format:
        loguru.logger.add(
            log_dir / "mindflow_json_{time:YYYY-MM-DD}.log",
            level=settings.log.level,
            rotation=settings.log.rotation,
            retention="7 days",
            compression=settings.log.compression,
            serialize=True,
        )

    loguru.logger.info(
        "Logging configured — level={}, rotation={}, retention={}",
        settings.log.level,
        settings.log.rotation,
        settings.log.retention,
    )
