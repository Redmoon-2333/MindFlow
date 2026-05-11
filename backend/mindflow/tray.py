"""MindFlow system tray application.

Usage:
    python -m mindflow.tray
"""

import threading
import webbrowser
import sys
import time

from PIL import Image, ImageDraw
import pystray
import httpx

from mindflow.main import app as fastapi_app
from mindflow.logging_config import get_logger

logger = get_logger(__name__)

API_BASE = "http://127.0.0.1:8765/api/v1"
DASHBOARD_URL = "http://127.0.0.1:8765/docs"


def _create_icon_image(color: str = "#4A90D9") -> Image.Image:
    """Draw a simple 64x64 circle icon."""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((8, 8, 56, 56), fill=color)
    return img


def _api_call(method: str, path: str) -> bool:
    """Make an API call, return True on success."""
    try:
        resp = httpx.request(method, f"{API_BASE}{path}", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


class MindFlowTray:
    def __init__(self):
        self.icon = None
        self._server_thread = None

    def _run_server(self):
        import uvicorn
        uvicorn.run(fastapi_app, host="127.0.0.1", port=8765, log_level="info")

    def _start_server(self):
        if self._server_thread and self._server_thread.is_alive():
            return
        self._server_thread = threading.Thread(target=self._run_server, daemon=True)
        self._server_thread.start()
        time.sleep(1.5)

    def on_start_collector(self, icon, item):
        self._start_server()
        ok = _api_call("POST", "/collector/start")
        if ok:
            icon.notify("采集已启动", "MindFlow")
        else:
            icon.notify("启动失败，请检查后端是否正常运行", "MindFlow")

    def on_stop_collector(self, icon, item):
        ok = _api_call("POST", "/collector/stop")
        if ok:
            icon.notify("采集已停止", "MindFlow")

    def on_open_dashboard(self, icon, item):
        webbrowser.open(DASHBOARD_URL)

    def on_status(self, icon, item):
        try:
            resp = httpx.get(f"{API_BASE}/status", timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()["data"]
                running = "运行中" if data["collector_running"] else "已停止"
                total = data["total_activities"]
                icon.notify(f"采集状态: {running}\n已采集 {total} 条记录", "MindFlow 状态")
            else:
                icon.notify("无法获取状态", "MindFlow")
        except Exception:
            icon.notify("后端未响应", "MindFlow")

    def on_quit(self, icon, item):
        _api_call("POST", "/collector/stop")
        icon.stop()

    def run(self):
        self._start_server()
        image = _create_icon_image("#4A90D9")

        menu = pystray.Menu(
            pystray.MenuItem("📊 状态", self.on_status, default=True),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("▶ 启动采集", self.on_start_collector),
            pystray.MenuItem("⏹ 停止采集", self.on_stop_collector),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("🌐 打开 Dashboard", self.on_open_dashboard),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("❌ 退出", self.on_quit),
        )

        self.icon = pystray.Icon(
            "mindflow",
            image,
            "MindFlow - 智能专注助手",
            menu,
        )
        logger.info("MindFlow tray started")
        self.icon.run()


def main():
    tray = MindFlowTray()
    tray.run()


if __name__ == "__main__":
    main()
