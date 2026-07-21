"""EventCollector protocol and platform-specific collector factory.

Defines the collector abstraction used throughout the application.
Platform-specific implementations live in sibling modules (win32.py,
darwin.py, x11.py, wayland_fallback.py).

Design decisions:
  - Protocol (not ABC) for structural typing — mypy --strict catches
    missing methods at compile time without requiring explicit subclassing.
  - Factory function handles platform detection and dependency checks.
  - CollectorUnavailableError is raised when platform or dependencies
    are missing (not at import time — only when construction is attempted).
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from mindflow.domain.events import WindowSnapshot
from mindflow.errors import CollectorError


class CollectorUnavailableError(CollectorError):
    """Raised when a platform-specific collector cannot be instantiated.

    Reasons include:
      - Unsupported platform (e.g. unknown OS)
      - Missing native dependencies (e.g. pywin32, PyObjC, python-xlib)
      - Runtime environment incompatibility (e.g. Wayland without fallback)
    """


@runtime_checkable
class EventCollector(Protocol):
    """Protocol for platform-specific active-window collectors.

    Implementations must:
      - Be safe to construct (constructor never raises except
        for CollectorUnavailableError).
      - Never raise from snapshot() or idle_seconds() — return
        degraded values and log warnings instead.
      - Use asyncio.to_thread for any blocking native calls.
    """

    async def snapshot(self) -> WindowSnapshot:
        """Capture the current active-window state.

        Returns:
            A WindowSnapshot with the current active window details.
            On transient failure returns a degraded snapshot
            (app_name=\"unknown\") with a logged warning.
        """
        ...

    async def idle_seconds(self) -> float:
        """Return the number of seconds since last user input.

        Returns:
            Seconds since last keyboard/mouse input.
            Returns 0.0 when idle detection is unavailable or fails.
        """
        ...


def degraded_snapshot() -> WindowSnapshot:
    """Return a degraded snapshot indicating collector failure."""
    return WindowSnapshot(
        app_name="unknown",
        window_title="",
        process_name="unknown",
        is_idle=False,
        timestamp_utc=datetime.now(UTC),
    )


_MAX_TITLE_LEN = 512
"""Length cap for collector-sourced text fields (F4 security hardening).

A malicious/misbehaving app can set an arbitrarily long window title; with
no bound it is stored verbatim, growing unboundedly and expanding the PII
surface with every snapshot. Truncated here — the ingestion boundary where
raw platform data first becomes a WindowSnapshot — rather than in
WindowSnapshot itself: the domain type stays a plain ``str`` with no
constraint (it is a dataclass, not Pydantic, so there is no
``Field(max_length=...)`` convention to lean on), and truncation is a
data-hygiene concern about untrusted platform input, not a type invariant.
"""


def truncate_text_field(value: str, max_len: int = _MAX_TITLE_LEN) -> str:
    """Truncate collector-sourced text (window title, app name) to *max_len*.

    Every platform-specific ``snapshot()`` implementation should pass raw
    ``window_title``/``app_name`` strings through this before constructing
    a ``WindowSnapshot`` (F4) — centralised here instead of duplicated in
    each of win32/darwin/x11/wayland_fallback so a future 5th collector
    cannot forget it.
    """
    return value[:max_len] if len(value) > max_len else value


def create_collector(platform: str | None = None) -> EventCollector:
    """Factory: return the appropriate EventCollector for *platform*.

    The platform argument follows ``sys.platform`` convention:
      - ``win32``: Windows (win32gui + psutil + GetLastInputInfo)
      - ``darwin``: macOS (AppKit/NSWorkspace via PyObjC)
      - ``linux``: Linux X11 (python-xlib EWMH) or Wayland fallback
        based on ``XDG_SESSION_TYPE`` environment variable.

    Args:
        platform: Target platform name (defaults to ``sys.platform``).

    Returns:
        An EventCollector instance for the current platform.

    Raises:
        CollectorUnavailableError: When no collector is available.
    """
    if platform is None:
        platform = sys.platform

    if platform == "win32":
        from mindflow.infrastructure.collectors.win32 import Win32Collector

        return Win32Collector()

    if platform == "darwin":
        from mindflow.infrastructure.collectors.darwin import DarwinCollector

        return DarwinCollector()

    if platform == "linux":
        import os

        xdg_session = os.environ.get("XDG_SESSION_TYPE", "").lower()
        if xdg_session == "wayland":
            from mindflow.infrastructure.collectors.wayland_fallback import (
                WaylandFallbackCollector,
            )

            return WaylandFallbackCollector()

        from mindflow.infrastructure.collectors.x11 import X11Collector

        return X11Collector()

    raise CollectorUnavailableError(f"No collector available for platform: {platform!r}")
