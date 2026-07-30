"""Notification service for local desktop notifications.

Defines a ``NotificationService`` protocol and platform-specific implementations.

Per 3.7 of the architecture:
  - Windows: Tkinter desktop popup (preferred for interventions)
    -> win10toast (plain notifications, no buttons)
    -> winrt (requires packaged app)
    -> plyer
  - macOS: pyobjc NSUserNotification (not yet implemented, Wave 7+)
  - Linux: notify-send (not yet implemented, Wave 7+)
  - Fallback: LogOnlyNotifier (writes to log, no desktop popup)

Interactive intervention notifications (Windows):
  When ``intervention_id`` is provided, the notifier creates a desktop popup with
  Accept/Ignore/Dismiss buttons.  Button clicks call back to the MindFlow
  API to record the response — no need to return to the web UI.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from loguru import logger

Urgency = Literal["low", "normal", "critical"]


class NotificationService(Protocol):
    """Protocol for platform-specific desktop notifications.

    Args:
        title: Notification title.
        body: Notification body text.
        urgency: Notification priority level (low, normal, critical).
        intervention_id: Optional intervention UUID.  When provided, the
            notification includes interactive action buttons.
        auth_token: Optional API auth token for button-click callbacks.

    Returns:
        True if the notification was sent successfully, False on failure.
    """

    async def send(
        self,
        title: str,
        body: str,
        urgency: Urgency = "normal",
        intervention_id: str | None = None,
        auth_token: str | None = None,
    ) -> bool:
        ...


class LogOnlyNotifier:
    """Fallback notifier that writes notifications to the log.

    Used when the platform has no desktop notification support
    or when dependencies are missing. Never fails — just logs.
    """

    async def send(
        self,
        title: str,
        body: str,
        urgency: Urgency = "normal",
        intervention_id: str | None = None,
        auth_token: str | None = None,
    ) -> bool:
        """Log the notification and always return True."""
        logger.info("NOTIFICATION [{}] {}: {}", urgency, title, body)
        return True


class Win10ToastNotifier:
    """Windows notification via win10toast (works for unpackaged apps).

    win10toast internally registers a temporary shortcut so Windows
    recognizes the app and allows toast notifications, which makes it
    the most reliable choice for Python apps that are not MSIX-packaged.

    Note: win10toast does NOT support interactive buttons.
    """

    def __init__(self) -> None:
        try:
            toast_notifier_cls = import_module("win10toast").ToastNotifier
            self._toaster = toast_notifier_cls()
            self._available = True
            logger.debug("Win10ToastNotifier: win10toast available")
        except ImportError:
            self._available = False
            logger.debug("Win10ToastNotifier: win10toast not available")

    async def send(
        self,
        title: str,
        body: str,
        urgency: Urgency = "normal",
        intervention_id: str | None = None,
        auth_token: str | None = None,
    ) -> bool:
        if not self._available:
            return False
        try:
            def _show() -> None:
                self._toaster.show_toast(
                    title,
                    body,
                    threaded=True,
                    duration=8,
                )

            await asyncio.to_thread(_show)
            return True
        except Exception as exc:
            logger.warning("Win10ToastNotifier failed: {}", exc)
            return False


class _TkinterInteractivePopup:
    """Launch a dedicated Tkinter process and wait until its window is ready."""

    _READY_TIMEOUT_S = 5.0
    _POLL_INTERVAL_S = 0.05

    def __init__(self, api_base_url: str = "http://127.0.0.1:8765") -> None:
        self._api_base_url = api_base_url.rstrip("/")

    async def send(
        self,
        title: str,
        body: str,
        urgency: Urgency = "normal",
        intervention_id: str | None = None,
        auth_token: str | None = None,
    ) -> bool:
        if intervention_id is None or not auth_token:
            return False

        temp_dir = Path(tempfile.mkdtemp(prefix="mindflow_intervention_"))
        payload_path = temp_dir / "payload.json"
        ready_path = temp_dir / "ready"
        process: subprocess.Popen[bytes] | None = None
        ready = False

        try:
            payload = {
                "title": title,
                "body": body,
                "intervention_id": intervention_id,
                "api_url": (
                    f"{self._api_base_url}/api/v1/intervention/"
                    f"{intervention_id}/response"
                ),
                "timeout_s": 120,
            }
            payload_path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )

            python_executable = Path(sys.executable)
            pythonw_executable = python_executable.with_name("pythonw.exe")
            if pythonw_executable.exists():
                python_executable = pythonw_executable

            popup_script = Path(__file__).with_name("intervention_popup.py")
            child_env = os.environ.copy()
            child_env["MINDFLOW_POPUP_TOKEN"] = auth_token
            process = await asyncio.to_thread(
                subprocess.Popen,
                [
                    str(python_executable),
                    str(popup_script),
                    str(payload_path),
                    str(ready_path),
                ],
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=child_env,
            )

            deadline = time.monotonic() + self._READY_TIMEOUT_S
            while True:
                if ready_path.is_file():
                    ready = True
                    logger.debug("Intervention popup ready (pid={})", process.pid)
                    return True
                if process.poll() is not None:
                    logger.warning(
                        "Intervention popup exited before ready (code={})",
                        process.returncode,
                    )
                    return False
                if time.monotonic() >= deadline:
                    process.terminate()
                    logger.warning("Intervention popup did not become ready in time")
                    return False
                await asyncio.sleep(self._POLL_INTERVAL_S)
        except Exception as exc:
            logger.warning("Intervention popup failed: {}", exc)
            return False
        finally:
            if not ready and process is not None and process.poll() is None:
                process.terminate()
            shutil.rmtree(temp_dir, ignore_errors=True)


class WindowsNotifier:
    """Windows Desktop notification with multiple backends.

    Backends (tried in order):
      1. Tkinter desktop popup (buttons for interventions)
      2. win10toast (plain notifications)
      3. winrt (requires packaged app)
      4. plyer (cross-platform fallback)
      5. LogOnlyNotifier (last resort)
    """

    def __init__(self, api_base_url: str = "http://127.0.0.1:8765") -> None:
        self._interactive = _TkinterInteractivePopup(api_base_url=api_base_url)
        self._backends: list[NotificationService] = []

        # 1. win10toast (most reliable for unpackaged apps, no buttons)
        try:
            w = Win10ToastNotifier()
            if w._available:
                self._backends.append(w)
                logger.debug("WindowsNotifier: win10toast backend added")
        except Exception as exc:
            logger.debug("WindowsNotifier: win10toast init failed: {}", exc)

        # 2. winrt (richer API, requires packaged app or valid AUMID)
        try:
            from winrt.windows.data.xml.dom import XmlDocument  # noqa: F401
            from winrt.windows.ui.notifications import (  # noqa: F401
                ToastNotification,
                ToastNotificationManager,
            )

            self._backends.append(_WinRTNotifier())
            logger.debug("WindowsNotifier: winrt backend added")
        except ImportError:
            logger.debug("WindowsNotifier: winrt not available")
        except Exception as exc:
            logger.debug("WindowsNotifier: winrt init failed: {}", exc)

        # 3. plyer (cross-platform, last resort)
        try:
            from plyer import notification as plyer_notification  # noqa: F401

            self._backends.append(_PlyerNotifier())
            logger.debug("WindowsNotifier: plyer backend added")
        except ImportError:
            logger.debug("WindowsNotifier: plyer not available")

        if not self._backends:
            logger.warning(
                "WindowsNotifier: no backends available, notifications disabled"
            )

    async def send(
        self,
        title: str,
        body: str,
        urgency: Urgency = "normal",
        intervention_id: str | None = None,
        auth_token: str | None = None,
    ) -> bool:
        """Send notification. For interventions, try the desktop popup first."""
        # Interactive popup (with buttons) — only for interventions
        interactive_requested = bool(intervention_id and auth_token)
        if interactive_requested:
            try:
                if await self._interactive.send(
                    title, body, urgency, intervention_id, auth_token
                ):
                    return True
            except Exception as exc:
                logger.debug("Interactive toast failed, falling back: {}", exc)

        # Plain notification backends
        for backend in self._backends:
            try:
                if await backend.send(title, body, urgency):
                    if interactive_requested:
                        logger.warning(
                            "Interactive popup unavailable; sent plain notification "
                            "without response actions"
                        )
                        return False
                    return True
            except Exception as exc:
                logger.debug(
                    "WindowsNotifier backend {} failed: {}",
                    type(backend).__name__,
                    exc,
                )
        # All backends failed — log as last resort
        logger.info("NOTIFICATION [{}] {}: {}", urgency, title, body)
        return False


class _WinRTNotifier:
    """Backend using winrt Toast notifications (no buttons)."""

    async def send(
        self,
        title: str,
        body: str,
        urgency: Urgency = "normal",
        intervention_id: str | None = None,
        auth_token: str | None = None,
    ) -> bool:
        try:
            from xml.sax.saxutils import escape as _escape

            from winrt.windows.data.xml.dom import XmlDocument
            from winrt.windows.ui.notifications import (
                ToastNotification,
                ToastNotificationManager,
            )

            def _create_and_show() -> None:
                app_ids_to_try = [
                    "Microsoft.Windows.Explorer",
                    "{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}"
                    "\\WindowsPowerShell\\v1.0\\powershell.exe",
                ]
                toast_xml = (
                    '<?xml version="1.0" encoding="utf-8"?>'
                    "<toast>"
                    "  <visual>"
                    '    <binding template="ToastGeneric">'
                    f"      <text>{_escape(title)}</text>"
                    f"      <text>{_escape(body)}</text>"
                    "    </binding>"
                    "  </visual>"
                    "</toast>"
                )

                last_err = None
                for app_id in app_ids_to_try:
                    try:
                        toast_manager = cast(
                            Any, ToastNotificationManager
                        ).create_toast_notifier(
                            app_id
                        )
                        xml_doc = XmlDocument()
                        xml_doc.load_xml(toast_xml)
                        toast = ToastNotification(xml_doc)
                        toast_manager.show(toast)
                        return
                    except Exception as exc:
                        last_err = exc
                        continue

                if last_err:
                    raise last_err

            await asyncio.to_thread(_create_and_show)
            return True
        except Exception as exc:
            logger.debug("WinRT notifier failed: {}", exc)
            return False


class _PlyerNotifier:
    """Backend using plyer notification (no buttons)."""

    async def send(
        self,
        title: str,
        body: str,
        urgency: Urgency = "normal",
        intervention_id: str | None = None,
        auth_token: str | None = None,
    ) -> bool:
        try:
            from plyer import notification as plyer_notification

            plyer_notification.notify(title=title, message=body, timeout=8)
            return True
        except Exception as exc:
            logger.debug("Plyer notifier failed: {}", exc)
            return False


def create_notifier(
    api_base_url: str = "http://127.0.0.1:8765",
) -> NotificationService:
    """Factory function that returns the best available notifier for the platform.

    Returns:
        A ``NotificationService`` implementation:
          - Windows: ``WindowsNotifier`` (popup -> win10toast -> winrt -> plyer -> LogOnly)
          - macOS: ``LogOnlyNotifier`` (desktop backend not implemented)
          - Linux: ``LogOnlyNotifier`` (desktop backend not implemented)
    """
    platform = sys.platform

    if platform == "win32":
        try:
            return WindowsNotifier(api_base_url=api_base_url)
        except Exception as exc:
            logger.warning("WindowsNotifier init failed: {}, falling back to LogOnly", exc)

    if platform == "darwin":
        logger.info("macOS notifications not yet implemented (Wave 7+), using LogOnly")

    if platform.startswith("linux"):
        logger.info("Linux notifications not yet implemented (Wave 7+), using LogOnly")

    return LogOnlyNotifier()
