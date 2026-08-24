"""Loguru logging configuration.

Provides structured logging with:
  - Console output (development-friendly)
  - File output with rotation (10 MB), retention (30 days), and gzip compression
  - JSON format option for production/ELK consumption
  - Request ID binding via context
  - OpenTelemetry trace ID injection via patcher (privacy-safe: trace IDs only)

All log paths live under platformdirs user data directory, with a fallback
to the project-local logs/ directory when the user data directory is
inaccessible (e.g., sandboxed environments or permission-denied).
"""

from __future__ import annotations

import sys
from pathlib import Path

import loguru
import platformdirs

from mindflow.config import Settings


def _patch_trace_id(record: loguru.Record) -> None:
    """Injects the current OpenTelemetry trace ID into the log record's extra dict.

    Called by loguru before each record is formatted. Fails silently when
    no span is active or when the telemetry module is not yet loaded.
    """
    try:
        from mindflow.telemetry.tracing import current_trace_id

        trace_id = current_trace_id()
        if trace_id is not None:
            record["extra"]["trace_id"] = trace_id
        else:
            record["extra"].setdefault("trace_id", "")
    except Exception:
        record["extra"].setdefault("trace_id", "")


def _resolve_log_dir() -> Path:
    """Return the best-effort log directory.

    Priority:
      1. platformdirs user_data_dir (e.g. %LOCALAPPDATA%\\mindflow\\mindflow\\logs)
      2. Project-local "logs/" next to the project root (fallback)

    A quick write-test is performed; if the preferred directory rejects
    writes (PermissionError / OSError), the fallback is used instead.
    """
    candidates: list[tuple[str, Path]] = []

    try:
        preferred = Path(platformdirs.user_data_dir("mindflow", ensure_exists=True)) / "logs"
        preferred.mkdir(parents=True, exist_ok=True)
        candidates.append(("user-data", preferred))
    except OSError as exc:
        import logging
        logging.getLogger(__name__).warning("user-data log dir unavailable: %s", exc)

    # Fallback: project-local logs/ directory (3 levels up from this file)
    project_log = Path(__file__).resolve().parent.parent.parent / "logs"
    candidates.append(("project-local", project_log))

    for _label, candidate in candidates:
        try:
            probe = candidate / ".write_test"
            probe.touch(exist_ok=True)
            probe.unlink(missing_ok=True)
            return candidate
        except OSError:
            continue

    # Last resort: use the first candidate even if probe failed
    first = candidates[0][1] if candidates else Path("logs")
    first.mkdir(parents=True, exist_ok=True)
    return first


def setup_logging(settings: Settings) -> None:
    """Configure loguru with console and rotating file handlers.

    Removes the default loguru handler and replaces it with:
      1. stderr console handler with colorized format
      2. Rotating file handler with configurable size, retention, and
         compression (best-effort; failure falls back to console-only)

    A trace_id patcher is configured so every log record automatically
    carries the active OpenTelemetry trace ID when available.

    Args:
        settings: Application settings containing log configuration.
    """
    loguru.logger.remove()

    # Inject trace_id into every log record's extra dict (privacy-safe).
    loguru.logger.configure(patcher=_patch_trace_id)

    log_dir = _resolve_log_dir()

    # Console handler — colorized, development-friendly
    loguru.logger.add(
        sys.stderr,
        level=settings.log.level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
            " | <level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,
        diagnose=False,  # Don't expose local variables in tracebacks
    )

    # File handler — text format, rotated (best-effort)
    try:
        loguru.logger.add(
            log_dir / "mindflow_{time:YYYY-MM-DD}.log",
            level=settings.log.level,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line}"
                " | trace_id={extra[trace_id]} | {message}"
            ),
            rotation=settings.log.rotation,
            retention=settings.log.retention,
            compression=settings.log.compression,
            backtrace=True,
            diagnose=False,
            serialize=settings.log.json_format,
        )
    except OSError as exc:
        loguru.logger.warning(
            f"File log handler skipped (fallback to console-only): {exc}"
        )

    # JSON-structured handler (separate file, shorter retention)
    if settings.log.json_format:
        try:
            loguru.logger.add(
                log_dir / "mindflow_json_{time:YYYY-MM-DD}.log",
                level=settings.log.level,
                rotation=settings.log.rotation,
                retention="7 days",
                compression=settings.log.compression,
                serialize=True,
            )
        except OSError as exc:
            loguru.logger.warning(f"JSON log handler skipped: {exc}")

    loguru.logger.info(
        "Logging configured — level={}, rotation={}, retention={}, dir={}",
        settings.log.level,
        settings.log.rotation,
        settings.log.retention,
        log_dir,
    )
