"""macOS active-window collector using NSWorkspace via PyObjC.

Implements the EventCollector protocol for macOS:
  - ``snapshot()``: Active application via NSWorkspace.sharedWorkspace
  - ``idle_seconds()``: CGEventSourceSecondsSinceLastEvent via Quartz

Requires PyObjC (AppKit + Quartz frameworks). Constructor raises
CollectorUnavailableError when dependencies are not installed.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime

from loguru import logger

from mindflow.domain.events import WindowSnapshot
from mindflow.infrastructure.collectors.base import CollectorUnavailableError


class DarwinCollector:
    """macOS collector using PyObjC (AppKit/Quartz) bridge."""

    def __init__(self) -> None:
        if sys.platform != "darwin":
            raise CollectorUnavailableError("DarwinCollector requires macOS")

        try:
            import AppKit  # noqa: F401
            import Quartz  # noqa: F401
        except ImportError as exc:
            raise CollectorUnavailableError(
                "DarwinCollector requires PyObjC (AppKit + Quartz)"
            ) from exc

    async def snapshot(self) -> WindowSnapshot:
        """Capture active window via NSWorkspace (runs in thread)."""
        try:
            return await asyncio.to_thread(self._snapshot_sync)
        except Exception:
            logger.warning("macOS snapshot failed", exc_info=True)
            return _degraded()

    async def idle_seconds(self) -> float:
        """Return idle seconds via Quartz event system."""
        try:
            import Quartz

            idle = Quartz.CGEventSourceSecondsSinceLastEvent(
                Quartz.kCGEventSourceStateCombinedSessionState
            )
            return float(idle) if idle is not None else 0.0
        except Exception:
            logger.warning("macOS idle detection failed", exc_info=True)
            return 0.0

    # ── Synchronous helper (called via asyncio.to_thread) ──────────────

    def _snapshot_sync(self) -> WindowSnapshot:
        """Synchronous macOS window capture — runs in a thread."""
        import AppKit

        ws = AppKit.NSWorkspace.sharedWorkspace()
        app = ws.activeApplication()

        if app is None:
            logger.warning("NSWorkspace.activeApplication returned None")
            return _degraded()

        app_name = str(app.get("NSApplicationName", "unknown") or "unknown")
        process_name = str(app.get("NSApplicationBundleIdentifier", app_name) or app_name)

        # Try to get the localized name of the active app as window title
        window_title = ""
        for running_app in ws.runningApplications():
            if running_app.isActive():
                window_title = str(running_app.localizedName() or "")
                break

        return WindowSnapshot(
            app_name=app_name,
            window_title=window_title,
            process_name=process_name,
            is_idle=False,
            timestamp_utc=datetime.now(UTC),
        )


def _degraded() -> WindowSnapshot:
    """Return a degraded snapshot indicating collector failure."""
    return WindowSnapshot(
        app_name="unknown",
        window_title="",
        process_name="unknown",
        is_idle=False,
        timestamp_utc=datetime.now(UTC),
    )
