"""System tray launcher for MindFlow backend.

Starts ``python -m mindflow.main`` as a subprocess and provides a tray icon
with start / stop / dashboard / quit actions.  Works cross-platform via
``pystray``.

Usage::

    python -m mindflow.tray
"""

from __future__ import annotations

import subprocess
import sys
import webbrowser

import pystray
from loguru import logger
from PIL import Image

_DASHBOARD_URL = "http://localhost:8765/docs"

# Simple 1x1 green pixel used as a fallback tray icon when no .ico is found.
_ICON_IMAGE = Image.new("RGB", (64, 64), (0, 180, 0))


class TrayApp:
    """System tray icon that manages the MindFlow backend subprocess.

    Parameters
    ----------
    host:
        Server bind address (forwarded to ``mindflow.main``).
    port:
        Server port (forwarded to ``mindflow.main``).
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self._host = host
        self._port = port
        self._process: subprocess.Popen[str] | None = None  # type: ignore[type-arg]
        self._icon: pystray.Icon | None = None

    # ------------------------------------------------------------------
    # Subprocess lifecycle
    # ------------------------------------------------------------------

    def _start_backend(self) -> None:
        """Start the backend subprocess if not already running."""
        if self._process is not None and self._process.poll() is None:
            logger.info("Backend already running (pid {})", self._process.pid)
            return

        cmd = [sys.executable, "-m", "mindflow.main"]
        logger.info("Starting backend: {}", " ".join(cmd))
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("Backend started (pid {})", self._process.pid)

    def _stop_backend(self) -> None:
        """Gracefully stop the backend subprocess (SIGTERM then wait)."""
        if self._process is None or self._process.poll() is not None:
            logger.info("Backend not running — nothing to stop")
            self._process = None
            return

        pid = self._process.pid
        logger.info("Stopping backend (pid {}) …", pid)
        try:
            self._process.terminate()
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Backend did not exit in time — killing (pid {})", pid)
            self._process.kill()
            self._process.wait(timeout=3)
        except Exception:
            logger.opt(exception=True).error("Error stopping backend")
        else:
            logger.info("Backend stopped (pid {})", pid)
        finally:
            self._process = None

    # ------------------------------------------------------------------
    # Menu actions
    # ------------------------------------------------------------------

    def _on_start(self, _icon: pystray.Icon, _item: pystray.MenuItem) -> None:
        self._start_backend()

    def _on_stop(self, _icon: pystray.Icon, _item: pystray.MenuItem) -> None:
        self._stop_backend()

    def _on_open_dashboard(self, _icon: pystray.Icon, _item: pystray.MenuItem) -> None:
        logger.info("Opening dashboard: {}", _DASHBOARD_URL)
        webbrowser.open(_DASHBOARD_URL)

    def _on_quit(self, _icon: pystray.Icon, _item: pystray.MenuItem) -> None:
        self._stop_backend()
        _icon.stop()

    # ------------------------------------------------------------------
    # Icon & menu
    # ------------------------------------------------------------------

    def _build_menu(self) -> pystray.Menu:
        return pystray.Menu(
            pystray.MenuItem("启动采集", self._on_start),
            pystray.MenuItem("停止采集", self._on_stop),
            pystray.MenuItem("打开 Dashboard", self._on_open_dashboard),
            pystray.MenuItem("退出", self._on_quit),
        )

    def run(self) -> None:
        """Block and run the tray icon (starts backend immediately)."""
        self._start_backend()

        self._icon = pystray.Icon(
            name="MindFlow",
            icon=_ICON_IMAGE,
            title="MindFlow",
            menu=self._build_menu(),
        )
        logger.info("Tray icon running — right-click for menu")
        self._icon.run()


def main() -> None:
    """Entry point for ``python -m mindflow.tray``."""
    from mindflow.config import get_settings

    settings = get_settings()
    app = TrayApp(host=settings.host, port=settings.port)
    app.run()


if __name__ == "__main__":
    main()
