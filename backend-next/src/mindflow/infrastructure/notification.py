"""Notification service for local desktop notifications.

Defines a ``NotificationService`` protocol and platform-specific implementations.

Per §3.7 of the architecture:
  - Windows: winrt Toast notifications (fallback: plyer)
  - macOS: pyobjc NSUserNotification (not yet implemented, Wave 7+)
  - Linux: notify-send (not yet implemented, Wave 7+)
  - Fallback: LogOnlyNotifier (writes to log, no desktop popup)

The ``create_notifier()`` factory selects the best available implementation.
If platform dependencies are missing, imports fail silently and fall back
to ``LogOnlyNotifier`` — notification failures never propagate to callers.
"""

from __future__ import annotations

import sys
from typing import Literal, Protocol

from loguru import logger

Urgency = Literal["low", "normal", "critical"]


class NotificationService(Protocol):
    """Protocol for platform-specific desktop notifications.

    Args:
        title: Notification title.
        body: Notification body text.
        urgency: Notification priority level (low, normal, critical).

    Returns:
        True if the notification was sent successfully, False on failure.
    """

    async def send(self, title: str, body: str, urgency: Urgency = "normal") -> bool:
        ...


class LogOnlyNotifier:
    """Fallback notifier that writes notifications to the log.

    Used when the platform has no desktop notification support
    or when dependencies are missing. Never fails — just logs.
    """

    async def send(self, title: str, body: str, urgency: Urgency = "normal") -> bool:
        """Log the notification and always return True."""
        logger.info("NOTIFICATION [{}] {}: {}", urgency, title, body)
        return True


class WindowsNotifier:
    """Windows Toast notification via winrt (preferred) or plyer (fallback).

    Tries winrt first (richer API), falls back to plyer if not available.
    If neither is available, logs and returns False (caller decides escalation).
    """

    def __init__(self) -> None:
        self._notifier: NotificationService | None = None

        # Try winrt first
        try:
            from winrt.windows.data.xml.dom import XmlDocument
            from winrt.windows.ui.notifications import (
                ToastNotification,
                ToastNotificationManager,
                ToastNotifier,
            )

            self._toast_notifier: ToastNotifier | None = None
            self._ToastNotificationManager = ToastNotificationManager
            self._ToastNotification = ToastNotification
            self._XmlDocument = XmlDocument
            self._use_winrt = True
            logger.debug("WindowsNotifier: using winrt toast notifications")
        except ImportError:
            self._use_winrt = False
            logger.debug("WindowsNotifier: winrt not available, trying plyer")
            # Try plyer fallback
            try:
                from plyer import notification as plyer_notification

                self._plyer_notification = plyer_notification
                self._use_plyer = True
                logger.debug("WindowsNotifier: using plyer notification")
            except ImportError:
                self._use_plyer = False
                logger.warning(
                    "WindowsNotifier: neither winrt nor plyer available, "
                    "notifications disabled"
                )

    async def send(self, title: str, body: str, urgency: Urgency = "normal") -> bool:
        """Send a Windows Toast notification.

        Falls back to logging if neither winrt nor plyer is available.
        """
        if self._use_winrt:
            try:
                return await self._send_winrt(title, body)
            except Exception as exc:
                logger.warning("WindowsNotifier winrt failed ({}), trying plyer", exc)
                if self._use_plyer:
                    return await self._send_plyer(title, body)
                return False

        if self._use_plyer:
            try:
                return await self._send_plyer(title, body)
            except Exception as exc:
                logger.warning("WindowsNotifier plyer failed: {}", exc)
                return False

        logger.info("NOTIFICATION [{}] {}: {}", urgency, title, body)
        return True

    async def _send_winrt(self, title: str, body: str) -> bool:
        """Send a Toast notification via winrt."""
        try:
            import asyncio

            def _create_and_show() -> None:
                """Create and show the toast notification (runs in executor)."""
                app_id = "MindFlow.MindFlow"
                toast_manager = self._ToastNotificationManager.create_toast_notifier(app_id)

                # Escape user-influenced text — LLM-generated intervention
                # copy must not inject XML elements (security audit M1).
                from xml.sax.saxutils import escape

                toast_xml = (
                    f'<?xml version="1.0" encoding="utf-8"?>'
                    f"<toast>"
                    f'  <visual>'
                    f'    <binding template="ToastGeneric">'
                    f"      <text>{escape(title)}</text>"
                    f"      <text>{escape(body)}</text>"
                    f"    </binding>"
                    f"  </visual>"
                    f"</toast>"
                )

                xml_doc = self._XmlDocument()
                xml_doc.load_xml(toast_xml)
                toast = self._ToastNotification(xml_doc)
                toast_manager.show(toast)

            await asyncio.to_thread(_create_and_show)
            return True
        except Exception as exc:
            logger.warning("WindowsNotifier winrt toast failed: {}", exc)
            return False

    async def _send_plyer(self, title: str, body: str) -> bool:
        """Send a notification via plyer."""
        try:
            self._plyer_notification.notify(title=title, message=body, timeout=5)
            return True
        except Exception as exc:
            logger.warning("WindowsNotifier plyer failed: {}", exc)
            return False


def create_notifier() -> NotificationService:
    """Factory function that returns the best available notifier for the platform.

    Returns:
        A ``NotificationService`` implementation:
          - Windows: ``WindowsNotifier`` (winrt -> plyer -> LogOnly)
          - macOS: ``LogOnlyNotifier`` (TODO: implement via pyobjc in Wave 7+)
          - Linux: ``LogOnlyNotifier`` (TODO: implement via notify-send in Wave 7+)
    """
    platform = sys.platform

    if platform == "win32":
        try:
            return WindowsNotifier()
        except Exception as exc:
            logger.warning("WindowsNotifier init failed: {}, falling back to LogOnly", exc)

    if platform == "darwin":
        logger.info("macOS notifications not yet implemented (Wave 7+), using LogOnly")

    if platform.startswith("linux"):
        logger.info("Linux notifications not yet implemented (Wave 7+), using LogOnly")

    return LogOnlyNotifier()
