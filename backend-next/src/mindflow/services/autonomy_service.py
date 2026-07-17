"""Autonomy service — G005 user-level switch for the autonomous agent.

Provides ``is_enabled()``, ``pause()``, and ``resume()`` backed by the
``PreferencesRepository`` ``autonomy`` namespace:

  - ``autonomy.enabled``: bool (default True).  Master switch.
  - ``autonomy.paused_until``: ISO 8601 timestamp or None.  When present
    and not expired, ``is_enabled()`` returns False regardless of ``enabled``.

Design (§7 of 07-agent-upgrade-design.md):
  The autonomy switch is checked by EVERY entry point into the autonomous
  intervention agent — the scheduler's ``auto_intervention_check``, the
  daily-panel cron job, and any future autonomous actions.  This ensures the
  user has a single "pause all" kill switch that takes effect globally.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast

from loguru import logger

from mindflow.infrastructure.repositories.preferences import (
    PreferencesRepository,
)

# Namespace key used inside the preferences JSON blob
_AUTONOMY_KEY: str = "autonomy"


class AutonomyService:
    """Manages the user-level autonomy switch.

    All state is persisted via ``PreferencesRepository``.  A single instance
    is safe to share — the class is stateless beyond the injected repo.

    Args:
        preferences_repo: Repository for user preferences.
    """

    def __init__(self, preferences_repo: PreferencesRepository) -> None:
        self._prefs = preferences_repo

    async def is_enabled(self, user_id: int = 1) -> bool:
        """Return True if the autonomous agent is currently active.

        Checks two conditions:
          1. ``autonomy.enabled`` must not be False (default True).
          2. ``autonomy.paused_until``, if set, must be in the past.

        Both conditions must be met for the agent to be enabled.
        """
        prefs = await self._prefs.get(user_id)
        auto: dict[str, Any] = prefs.get(_AUTONOMY_KEY, {})

        # Master switch
        if not auto.get("enabled", True):
            logger.debug("Autonomy disabled by user preference")
            return False

        # Pause window
        paused_until_raw: str | None = auto.get("paused_until")
        if paused_until_raw is not None:
            try:
                paused_until = datetime.fromisoformat(paused_until_raw)
                if paused_until.tzinfo is None:
                    paused_until = paused_until.replace(tzinfo=UTC)
            except (ValueError, TypeError) as exc:
                logger.warning("Invalid paused_until timestamp: {}", exc)
                return False  # Treat unparseable timestamps as paused

            if paused_until > datetime.now(UTC):
                logger.debug(
                    "Autonomy paused until {} ({})",
                    paused_until_raw,
                    paused_until - datetime.now(UTC),
                )
                return False

        return True

    async def pause(self, hours: float = 1.0, user_id: int = 1) -> None:
        """Pause the autonomous agent for *hours*.

        Sets ``autonomy.paused_until`` to the current time plus *hours*.
        Does NOT flip ``autonomy.enabled`` — resuming early is a separate call.

        Args:
            hours: Number of hours to pause for (default 1, min 0.5).
            user_id: User identifier.
        """
        if hours < 0.5:
            hours = 0.5  # Enforce minimum pause duration
        paused_until = datetime.now(UTC) + timedelta(hours=hours)

        prefs = await self._prefs.get(user_id)
        auto: dict[str, Any] = prefs.setdefault(_AUTONOMY_KEY, {})
        auto["paused_until"] = paused_until.isoformat()

        await self._prefs.set(user_id, prefs)
        logger.info("Autonomy paused for {}h (until {})", hours, paused_until.isoformat())

    async def resume(self, user_id: int = 1) -> None:
        """Resume the autonomous agent immediately.

        Clears ``autonomy.paused_until``.  Does NOT flip ``autonomy.enabled``
        (the user may have explicitly disabled it, and resume should only
        undo a previous pause).
        """
        prefs = await self._prefs.get(user_id)
        auto: dict[str, Any] = prefs.setdefault(_AUTONOMY_KEY, {})
        auto.pop("paused_until", None)

        await self._prefs.set(user_id, prefs)
        logger.info("Autonomy resumed (paused_until cleared)")

    async def get_status(self, user_id: int = 1) -> dict[str, object]:
        """Return a dict with ``enabled`` and ``paused_until`` for API responses.

        Args:
            user_id: User identifier.

        Returns:
            A dict like ``{"enabled": True, "paused_until": "2026-07-18T12:00:00+00:00"}``.
            ``paused_until`` is None when not paused.
        """
        prefs = await self._prefs.get(user_id)
        auto: dict[str, Any] = prefs.get(_AUTONOMY_KEY, {})
        paused_until_raw = auto.get("paused_until")
        paused_until: str | None = cast(str | None, paused_until_raw)
        enabled = await self.is_enabled(user_id)
        return {"enabled": enabled, "paused_until": paused_until}
