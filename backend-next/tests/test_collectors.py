"""Tests for mindflow.infrastructure.collectors — factory, protocol, degraded.

Tests cover:
  - Factory selection logic for all platforms (mock sys.platform + imports)
  - Degraded snapshot on collector failure
  - EventCollector protocol isinstance checks
  - Win32 collector on actual Windows (1 integration test)
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from mindflow.domain.events import WindowSnapshot
from mindflow.infrastructure.collectors.base import (
    CollectorUnavailableError,
    EventCollector,
    create_collector,
)


class TestCreateCollector:
    """Factory function — platform selection and error handling."""

    def test_win32_selects_win32_collector(self):
        """Factory returns Win32Collector for win32 platform."""
        collector = create_collector("win32")
        from mindflow.infrastructure.collectors.win32 import Win32Collector

        assert isinstance(collector, Win32Collector)
        assert isinstance(collector, EventCollector)

    def test_darwin_selects_darwin_collector(self):
        """Factory returns DarwinCollector for darwin platform.

        Patches sys.platform and the DarwinCollector constructor
        because PyObjC is not installed on CI/Windows.
        """
        with (
            patch("sys.platform", "darwin"),
            patch(
                "mindflow.infrastructure.collectors.darwin.DarwinCollector",
                autospec=True,
            ) as mock_cls,
        ):
            mock_instance = mock_cls.return_value
            collector = create_collector("darwin")
            assert collector is mock_instance
            mock_cls.assert_called_once()

    def test_linux_x11_selects_x11_collector(self):
        """Factory returns X11Collector for linux with XDG_SESSION_TYPE != wayland."""
        with (
            patch("sys.platform", "linux"),
            patch(
                "mindflow.infrastructure.collectors.x11.X11Collector",
                autospec=True,
            ) as mock_cls,
        ):
            mock_instance = mock_cls.return_value
            collector = create_collector("linux")
            assert collector is mock_instance
            mock_cls.assert_called_once()

    def test_linux_wayland_selects_wayland_fallback(self):
        """Factory returns WaylandFallbackCollector for linux + wayland."""
        with (
            patch("sys.platform", "linux"),
            patch.dict("os.environ", {"XDG_SESSION_TYPE": "wayland"}),
            patch(
                "mindflow.infrastructure.collectors.wayland_fallback.WaylandFallbackCollector",
                autospec=True,
            ) as mock_cls,
        ):
            mock_instance = mock_cls.return_value
            collector = create_collector("linux")
            assert collector is mock_instance
            mock_cls.assert_called_once()

    def test_unsupported_platform_raises(self):
        """Factory raises CollectorUnavailableError for unknown platforms."""
        with pytest.raises(CollectorUnavailableError):
            create_collector("unknown_os")

    def test_defaults_to_sys_platform(self):
        """Factory defaults to sys.platform when argument is None."""
        collector = create_collector()
        if sys.platform == "win32":
            from mindflow.infrastructure.collectors.win32 import Win32Collector

            assert isinstance(collector, Win32Collector)
        else:
            # On non-Windows, it should raise or return something valid
            assert collector is not None


class TestWin32DegradedSnapshot:
    """Win32 collector — degraded snapshots on failure."""

    async def test_snapshot_returns_degraded_on_failure(self):
        """snapshot() returns degraded WindowSnapshot when Win32 API fails."""
        from mindflow.infrastructure.collectors.win32 import Win32Collector

        collector = Win32Collector()

        with patch.object(collector, "_snapshot_sync", side_effect=Exception("API fail")):
            snap = await collector.snapshot()
            assert snap.app_name == "unknown"
            assert snap.process_name == "unknown"
            assert snap.window_title == ""
            assert snap.is_idle is False

    async def test_idle_seconds_returns_zero_on_failure(self):
        """idle_seconds() returns 0.0 when GetLastInputInfo fails."""
        from mindflow.infrastructure.collectors.win32 import Win32Collector

        collector = Win32Collector()

        with patch.object(collector, "_idle_seconds_sync", side_effect=Exception("fail")):
            idle = await collector.idle_seconds()
            assert idle == 0.0


class TestWin32Collector:
    """Win32 collector — real integration test (Windows only)."""

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
    async def test_snapshot_returns_valid_window_snapshot(self):
        """On Windows, snapshot() returns a real WindowSnapshot."""
        from mindflow.infrastructure.collectors.win32 import Win32Collector

        collector = Win32Collector()
        snap = await collector.snapshot()

        assert isinstance(snap, WindowSnapshot)
        assert isinstance(snap.app_name, str)
        assert isinstance(snap.window_title, str)
        assert isinstance(snap.timestamp_utc, datetime)
        assert snap.timestamp_utc.tzinfo is not None  # timezone-aware

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
    async def test_idle_seconds_returns_float(self):
        """On Windows, idle_seconds() returns a non-negative float."""
        from mindflow.infrastructure.collectors.win32 import Win32Collector

        collector = Win32Collector()
        idle = await collector.idle_seconds()

        assert isinstance(idle, float)
        assert idle >= 0.0


class TestEventCollectorProtocol:
    """Structural typing — isinstance checks with runtime_checkable."""

    def test_collector_must_have_snapshot_and_idle(self):
        """A valid collector must have both snapshot() and idle_seconds()."""
        # An object with both async methods should pass isinstance check
        class ValidCollector:
            async def snapshot(self) -> WindowSnapshot:
                return WindowSnapshot(
                    app_name="test",
                    window_title="",
                    process_name="test",
                    is_idle=False,
                    timestamp_utc=datetime.now(UTC),
                )

            async def idle_seconds(self) -> float:
                return 0.0

        assert isinstance(ValidCollector(), EventCollector)

    def test_collector_missing_method_fails(self):
        """An object missing a protocol method fails isinstance check."""

        class MissingIdle:
            async def snapshot(self) -> WindowSnapshot:
                return WindowSnapshot(
                    app_name="test",
                    window_title="",
                    process_name="test",
                    is_idle=False,
                    timestamp_utc=datetime.now(UTC),
                )

        assert not isinstance(MissingIdle(), EventCollector)


class TestCreateCollectorErrors:
    """Factory — error cases for platform constructors."""

    def test_win32_on_non_win32_raises(self):
        """Win32Collector constructor raises on non-Windows."""
        from mindflow.infrastructure.collectors.win32 import Win32Collector

        with patch("sys.platform", "linux"), pytest.raises(CollectorUnavailableError):
            Win32Collector()
