import sys
import ctypes
from datetime import datetime
from typing import Optional


def get_active_window_info() -> Optional[dict]:
    if sys.platform != "win32":
        return {
            "process_name": "unknown",
            "window_title": "Non-Windows platform",
            "timestamp": datetime.utcnow().isoformat(),
        }
    try:
        import win32gui
        import win32process
        import psutil

        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            return None

        window_title = win32gui.GetWindowText(hwnd)
        window_class = win32gui.GetClassName(hwnd)

        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        try:
            proc = psutil.Process(pid)
            process_name = proc.name()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            process_name = "unknown"

        return {
            "process_name": process_name,
            "window_title": window_title or "",
            "window_class": window_class or "",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception:
        return None


class _LastInputInfo(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]


def is_user_idle(threshold_seconds: int = 60) -> bool:
    if sys.platform != "win32":
        return False
    try:
        lii = _LastInputInfo()
        lii.cbSize = ctypes.sizeof(_LastInputInfo)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
            return False
        millis_since_boot = ctypes.windll.kernel32.GetTickCount()
        idle_millis = millis_since_boot - lii.dwTime
        return idle_millis > (threshold_seconds * 1000)
    except Exception:
        return False
