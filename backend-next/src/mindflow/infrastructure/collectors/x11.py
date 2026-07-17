"""Linux X11 active-window collector using python-xlib with EWMH.

Implements the EventCollector protocol for Linux X11:
  - ``snapshot()``: Active window via EWMH _NET_ACTIVE_WINDOW + psutil pid
  - ``idle_seconds()``: XScreenSaver extension idle info

Requires python-xlib and psutil. Constructor raises
CollectorUnavailableError when python-xlib is not installed.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime

from loguru import logger

from mindflow.domain.events import WindowSnapshot
from mindflow.infrastructure.collectors.base import CollectorUnavailableError, degraded_snapshot


class X11Collector:
    """Linux X11 collector using python-xlib with EWMH hints."""

    def __init__(self) -> None:
        if sys.platform != "linux":
            raise CollectorUnavailableError("X11Collector requires Linux")

        try:
            import Xlib  # noqa: F401
        except ImportError as exc:
            raise CollectorUnavailableError(
                "X11Collector requires python-xlib"
            ) from exc

    async def snapshot(self) -> WindowSnapshot:
        """Capture active window via X11 EWMH (runs in thread)."""
        try:
            return await asyncio.to_thread(self._snapshot_sync)
        except Exception:
            logger.warning("X11 snapshot failed", exc_info=True)
            return degraded_snapshot()

    async def idle_seconds(self) -> float:
        """Return idle seconds via XScreenSaver extension."""
        try:
            return await asyncio.to_thread(self._idle_seconds_sync)
        except Exception:
            logger.warning("X11 idle detection failed", exc_info=True)
            return 0.0

    # ── Synchronous helpers (called via asyncio.to_thread) ──────────────

    def _snapshot_sync(self) -> WindowSnapshot:
        """Synchronous X11 EWMH window capture — runs in a thread."""
        from Xlib import display as xdisplay
        from Xlib.ext import ewmh

        d = xdisplay.Display()
        e = ewmh.EWMH(d)
        try:
            active = e.getActiveWindow()
            if not active:
                logger.warning("X11 EWMH getActiveWindow returned None")
                return degraded_snapshot()

            name = active.get_wm_name() or ""

            # Try to resolve process name from _NET_WM_PID
            process_name = "unknown"
            pid_prop = active.get_property(e._NET_WM_PID)
            if pid_prop and pid_prop.value:
                try:
                    import psutil

                    proc = psutil.Process(pid_prop.value[0])
                    process_name = proc.name() or "unknown"
                except ImportError:
                    pass  # psutil not installed — degraded name is acceptable
                except Exception:
                    logger.warning("X11 process name resolution failed", exc_info=True)

            return WindowSnapshot(
                app_name=process_name,
                window_title=name,
                process_name=process_name,
                is_idle=False,
                timestamp_utc=datetime.now(UTC),
            )
        finally:
            d.close()

    def _idle_seconds_sync(self) -> float:
        """Synchronous X11 idle detection — runs in a thread."""
        from Xlib import display as xdisplay

        d = xdisplay.Display()
        try:
            info = d.screen_saver_info()
            idle_ms = info.idle if hasattr(info, "idle") else 0
            return float(idle_ms) / 1000.0
        except Exception:
            return 0.0
        finally:
            d.close()
