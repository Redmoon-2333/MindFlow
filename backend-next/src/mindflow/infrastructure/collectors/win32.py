"""Windows active-window collector using win32gui and psutil.

Implements the EventCollector protocol for the Windows platform:
  - ``snapshot()``: Active window via win32gui foreground window + psutil pid
  - ``idle_seconds()``: GetLastInputInfo via ctypes

All blocking Win32 calls are wrapped in asyncio.to_thread so they do
not block the async event loop (ADR-007).

Degradation policy:
  Transient failures (win32gui error, psutil access denied, etc.)
  produce a degraded WindowSnapshot (app_name=\"unknown\") with a
  logger.warning — never raise.
"""

from __future__ import annotations

import asyncio
import ctypes
import sys
from datetime import UTC, datetime

from loguru import logger

from mindflow.domain.events import WindowSnapshot
from mindflow.infrastructure.collectors.base import CollectorUnavailableError


class _LastInputInfoStruct(ctypes.Structure):
    """GetLastInputInfo requires a cbSize-initialised structure."""
    _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]


class Win32Collector:
    """Windows active-window collector using native Win32 APIs.

    Requires pywin32 and psutil — raises CollectorUnavailableError
    in the constructor if either is missing.
    """

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise CollectorUnavailableError("Win32Collector requires Windows")

        try:
            import psutil  # noqa: F401
            import win32gui  # noqa: F401
            import win32process  # noqa: F401
        except ImportError as exc:
            raise CollectorUnavailableError(
                "Win32Collector requires pywin32 and psutil"
            ) from exc

    async def snapshot(self) -> WindowSnapshot:
        """Capture active window via Win32 APIs (runs in thread)."""
        try:
            return await asyncio.to_thread(self._snapshot_sync)
        except Exception:
            logger.warning("Win32 snapshot failed", exc_info=True)
            return _degraded()

    async def idle_seconds(self) -> float:
        """Return idle seconds via GetLastInputInfo (runs in thread)."""
        try:
            return await asyncio.to_thread(self._idle_seconds_sync)
        except Exception:
            logger.warning("Win32 idle detection failed", exc_info=True)
            return 0.0

    # ── Synchronous helpers (called via asyncio.to_thread) ──────────────

    def _snapshot_sync(self) -> WindowSnapshot:
        """Synchronous Win32 window capture — runs in a thread."""
        import psutil as _psutil
        import win32gui
        import win32process

        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            logger.warning("Win32 GetForegroundWindow returned NULL")
            return _degraded()

        window_title = win32gui.GetWindowText(hwnd) or ""
        _, pid = win32process.GetWindowThreadProcessId(hwnd)

        try:
            proc = _psutil.Process(pid)
            process_name = proc.name() or "unknown"
        except (_psutil.NoSuchProcess, _psutil.AccessDenied):
            process_name = "unknown"

        app_name = process_name

        return WindowSnapshot(
            app_name=app_name,
            window_title=window_title,
            process_name=process_name,
            is_idle=False,
            timestamp_utc=datetime.now(UTC),
        )

    def _idle_seconds_sync(self) -> float:
        """Synchronous idle detection — runs in a thread."""
        lii = _LastInputInfoStruct()
        lii.cbSize = ctypes.sizeof(_LastInputInfoStruct)

        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
            logger.warning("Win32 GetLastInputInfo returned FALSE")
            return 0.0

        millis_since_boot = ctypes.windll.kernel32.GetTickCount()

        # Guard against uint wraparound (GetTickCount wraps every ~49.7 days)
        if lii.dwTime > millis_since_boot:
            return 0.0

        return (millis_since_boot - lii.dwTime) / 1000.0  # type: ignore[no-any-return]


def _degraded() -> WindowSnapshot:
    """Return a degraded snapshot indicating collector failure."""
    return WindowSnapshot(
        app_name="unknown",
        window_title="",
        process_name="unknown",
        is_idle=False,
        timestamp_utc=datetime.now(UTC),
    )
