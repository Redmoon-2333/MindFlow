"""Tests for AutonomyService (G005 user-level autonomy switch).

Covers:
  - Default state (enabled, no pause)
  - Pause (active, expired, minimum duration enforced)
  - Resume (clear pause)
  - Master switch (autonomy.enabled = False)
  - Invalid paused_until handling
  - get_status() return shape
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from mindflow.services.autonomy_service import AutonomyService


@pytest.fixture
def mock_prefs() -> MagicMock:
    """Create a mock PreferencesRepository."""
    repo = MagicMock()
    repo.get = AsyncMock()
    repo.set = AsyncMock()
    return repo


@pytest.fixture
def service(mock_prefs: MagicMock) -> AutonomyService:
    """Create an AutonomyService with mocked preferences repo."""
    return AutonomyService(preferences_repo=mock_prefs)


class TestAutonomyService:
    """AutonomyService unit tests."""

    async def test_default_enabled(self, service: AutonomyService, mock_prefs: MagicMock) -> None:
        """Default state: no preferences → enabled (default True)."""
        mock_prefs.get.return_value = {}
        assert await service.is_enabled() is True

    async def test_enabled_when_no_autonomy_key(self, service: AutonomyService, mock_prefs: MagicMock) -> None:
        """Empty preferences with no 'autonomy' key → enabled."""
        mock_prefs.get.return_value = {"some_other_key": "value"}
        assert await service.is_enabled() is True

    async def test_disabled_when_enabled_false(self, service: AutonomyService, mock_prefs: MagicMock) -> None:
        """autonomy.enabled = False → disabled."""
        mock_prefs.get.return_value = {"autonomy": {"enabled": False}}
        assert await service.is_enabled() is False

    async def test_paused_when_paused_until_future(self, service: AutonomyService, mock_prefs: MagicMock) -> None:
        """paused_until in the future → disabled."""
        future = datetime.now(UTC) + timedelta(hours=2)
        mock_prefs.get.return_value = {"autonomy": {"paused_until": future.isoformat()}}
        assert await service.is_enabled() is False

    async def test_paused_until_expired(self, service: AutonomyService, mock_prefs: MagicMock) -> None:
        """paused_until in the past → enabled (pause expired)."""
        past = datetime.now(UTC) - timedelta(minutes=5)
        mock_prefs.get.return_value = {"autonomy": {"paused_until": past.isoformat()}}
        assert await service.is_enabled() is True

    async def test_paused_until_naive_tz(self, service: AutonomyService, mock_prefs: MagicMock) -> None:
        """paused_until without timezone → treated as UTC."""
        future = datetime.now(UTC) + timedelta(hours=1)
        naive = future.replace(tzinfo=None)
        mock_prefs.get.return_value = {"autonomy": {"paused_until": naive.isoformat()}}
        assert await service.is_enabled() is False

    async def test_invalid_paused_until(self, service: AutonomyService, mock_prefs: MagicMock) -> None:
        """Unparseable paused_until → treated as disabled (safe fallback)."""
        mock_prefs.get.return_value = {"autonomy": {"paused_until": "not-a-timestamp"}}
        assert await service.is_enabled() is False

    async def test_pause_sets_paused_until(self, service: AutonomyService, mock_prefs: MagicMock) -> None:
        """pause() writes paused_until to preferences."""
        mock_prefs.get.return_value = {}

        await service.pause(hours=2.0)

        # Verify set was called with updated preferences
        mock_prefs.set.assert_awaited_once()
        call_args = mock_prefs.set.await_args
        assert call_args is not None
        prefs = call_args[0][1]  # args[1] is the prefs dict
        assert "autonomy" in prefs
        assert "paused_until" in prefs["autonomy"]

    async def test_pause_enforces_minimum(self, service: AutonomyService, mock_prefs: MagicMock) -> None:
        """pause(hours=0.1) enforces minimum 0.5 hours."""
        mock_prefs.get.return_value = {}

        await service.pause(hours=0.1)

        mock_prefs.set.assert_awaited_once()
        call_args = mock_prefs.set.await_args
        assert call_args is not None
        prefs = call_args[0][1]
        paused_until = datetime.fromisoformat(prefs["autonomy"]["paused_until"])
        now = datetime.now(UTC)
        # Should be at least 0.5 hours in the future
        assert paused_until > now + timedelta(minutes=25)

    async def test_resume_clears_paused_until(self, service: AutonomyService, mock_prefs: MagicMock) -> None:
        """resume() removes paused_until from preferences."""
        mock_prefs.get.return_value = {
            "autonomy": {
                "enabled": True,
                "paused_until": (datetime.now(UTC) + timedelta(hours=2)).isoformat(),
            }
        }

        await service.resume()

        mock_prefs.set.assert_awaited_once()
        call_args = mock_prefs.set.await_args
        assert call_args is not None
        prefs = call_args[0][1]
        assert "paused_until" not in prefs["autonomy"]

    async def test_resume_preserves_enabled(self, service: AutonomyService, mock_prefs: MagicMock) -> None:
        """resume() does not change autonomy.enabled."""
        mock_prefs.get.return_value = {
            "autonomy": {
                "enabled": False,
                "paused_until": (datetime.now(UTC) + timedelta(hours=2)).isoformat(),
            }
        }

        await service.resume()

        mock_prefs.set.assert_awaited_once()
        call_args = mock_prefs.set.await_args
        assert call_args is not None
        prefs = call_args[0][1]
        assert prefs["autonomy"]["enabled"] is False  # unchanged

    async def test_get_status_shape(self, service: AutonomyService, mock_prefs: MagicMock) -> None:
        """get_status() returns dict with enabled and paused_until."""
        mock_prefs.get.return_value = {
            "autonomy": {
                "paused_until": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            }
        }

        status = await service.get_status()
        assert "enabled" in status
        assert "paused_until" in status
        assert status["enabled"] is False  # paused
        assert isinstance(status["paused_until"], str)

    async def test_get_status_no_pause(self, service: AutonomyService, mock_prefs: MagicMock) -> None:
        """get_status() returns paused_until=None when not paused."""
        mock_prefs.get.return_value = {"autonomy": {}}
        status = await service.get_status()
        assert status["enabled"] is True
        assert status["paused_until"] is None
