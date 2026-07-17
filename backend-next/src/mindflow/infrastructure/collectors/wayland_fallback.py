"""Linux Wayland fallback collector using psutil (pid-level only).

Wayland does not expose a standard API to query the active window
without compositor-specific protocols (zwlr-foreign-toplevel-management,
etc.). This collector provides best-effort pid-level data as a fallback.

Limitations:
  - No reliable foreground window detection (Wayland security model).
  - No idle detection (no standard Wayland idle protocol).
  - Best-effort terminal process detection via /proc.

Implements the EventCollector protocol for graceful degradation when
the user's session type is "wayland".
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import UTC, datetime

from loguru import logger

from mindflow.domain.events import WindowSnapshot
from mindflow.infrastructure.collectors.base import CollectorUnavailableError


class WaylandFallbackCollector:
    """Linux Wayland fallback: pid-level process detection only.

    Provides degraded window detection (process name from session TTY)
    and no idle detection. Used when XDG_SESSION_TYPE=wayland and no
    compositor-specific protocol is available.
    """

    def __init__(self) -> None:
        if sys.platform != "linux":
            raise CollectorUnavailableError(
                "WaylandFallbackCollector requires Linux"
            )

    async def snapshot(self) -> WindowSnapshot:
        """Return best-effort window snapshot.

        Tries to identify the foreground process via session TTY.
        Falls back to \"unknown\" on any failure.
        """
        try:
            return await asyncio.to_thread(self._snapshot_sync)
        except Exception:
            logger.warning("Wayland snapshot failed", exc_info=True)
            return _degraded()

    async def idle_seconds(self) -> float:
        """No idle detection available on Wayland fallback.

        Returns 0.0 unconditionally.
        """
        return 0.0

    # ── Synchronous helper (called via asyncio.to_thread) ──────────────

    def _snapshot_sync(self) -> WindowSnapshot:
        """Synchronous best-effort process detection — runs in a thread."""
        import psutil

        process_name = "unknown"
        window_title = ""

        # Best effort: try to find a terminal-based foreground process
        try:
            if os.isatty(0):
                for proc in psutil.process_iter(["name", "status", "pid"]):
                    try:
                        pname = (
                            proc.info.get("name") or ""
                        ).lower()
                        if pname and (
                            "terminal" in pname
                            or "konsole" in pname
                            or "gnome-terminal" in pname
                            or "alacritty" in pname
                            or "kitty" in pname
                            or "wezterm" in pname
                        ):
                            process_name = proc.info["name"]
                            window_title = process_name
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
        except (OSError, Exception):
            pass

        # Fallback: use any non-root interactive process
        if process_name == "unknown":
            try:
                for proc in psutil.process_iter(["name", "username", "pid"]):
                    try:
                        uname = proc.info.get("username") or ""
                        if uname and uname != "root":
                            process_name = proc.info.get("name") or "unknown"
                            window_title = process_name
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except Exception:
                pass

        return WindowSnapshot(
            app_name=process_name,
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
