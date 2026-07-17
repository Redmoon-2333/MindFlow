"""Tests for MaintenanceService.

Covers:
  - cleanup_old_events: old events deleted, recent events preserved,
    batch deletion
  - run_daily_backup: backup file created, notification on failure
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import text

from mindflow.infrastructure.repositories.activity import (
    activity_events,
)
from mindflow.services.maintenance_service import MaintenanceService

_BASE = datetime(2026, 7, 17, tzinfo=UTC)

# Fixed clock for deterministic tests — never expires
def _clock() -> datetime:
    return datetime(2026, 7, 17, 12, 0, 0, tzinfo=UTC)


async def _insert_events(engine, count: int, days_ago: int, user_id: int = 1) -> None:
    """Insert test events via raw SQL (bypasses heartbeat merge)."""
    ts = (_BASE - timedelta(days=days_ago)).isoformat()
    data = (
        '{"app_name":"Test","window_title":"Test","process_name":"test.exe",'
        '"is_idle":false,"timestamp_utc":"' + ts + '"}'
    )
    async with engine.begin() as conn:
        for i in range(count):
            await conn.execute(
                activity_events.insert().values(
                    id=f"test-{days_ago}-{i}",
                    user_id=user_id,
                    timestamp=ts,
                    duration_s=5.0,
                    data_json=data,
                    event_type="window_snapshot",
                )
            )


@pytest.fixture
async def setup_events(engine, session_factory):
    """Create table and insert test events at various ages."""
    async with engine.begin() as conn:
        await conn.run_sync(activity_events.metadata.create_all)
    await _insert_events(engine, 5, days_ago=40)  # Old events (> 30 days)
    await _insert_events(engine, 3, days_ago=10)  # Recent events


class TestCleanupOldEvents:
    """Event cleanup tests."""

    async def test_deletes_old_events(self, engine, session_factory, setup_events):
        """Events older than retention_days should be deleted."""
        notifier = AsyncMock()
        svc = MaintenanceService(
            engine=engine, session_factory=session_factory, notifier=notifier, clock=_clock
        )
        deleted = await svc.cleanup_old_events(retention_days=30)
        assert deleted == 5

        async with engine.connect() as conn:
            remaining = await conn.execute(
                text("SELECT count(*) FROM activity_events")
            )
            assert remaining.scalar() == 3

    async def test_preserves_recent_events(self, engine, session_factory, setup_events):
        """Events within retention window should be preserved."""
        notifier = AsyncMock()
        svc = MaintenanceService(
            engine=engine, session_factory=session_factory, notifier=notifier, clock=_clock
        )
        await svc.cleanup_old_events(retention_days=30)

        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT count(*) FROM activity_events WHERE timestamp >= :ts"),
                {"ts": (_BASE - timedelta(days=15)).isoformat()},
            )
            assert result.scalar() == 3

    async def test_no_events_to_delete(self, engine, session_factory, setup_events):
        """Calling cleanup when no old events exist should return 0."""
        notifier = AsyncMock()
        svc = MaintenanceService(
            engine=engine, session_factory=session_factory, notifier=notifier, clock=_clock
        )
        # First cleanup removes old events
        await svc.cleanup_old_events(retention_days=30)
        # Second call should find nothing to delete
        deleted = await svc.cleanup_old_events(retention_days=30)
        assert deleted == 0


class TestRunDailyBackup:
    """Daily backup tests."""

    async def test_backup_creates_file(self, engine, session_factory, tmp_path):
        """Backup should create a .db file."""
        notifier = AsyncMock()
        svc = MaintenanceService(
            engine=engine,
            session_factory=session_factory,
            notifier=notifier,
            data_dir=tmp_path,
        )
        success = await svc.run_daily_backup()
        assert success

        backup_dir = tmp_path / "backups"
        assert backup_dir.exists()
        files = list(backup_dir.glob("*.db"))
        assert len(files) >= 1

    async def test_backup_failure_notifies(self, engine, session_factory, tmp_path):
        """On backup failure, notification should be sent."""
        notifier = AsyncMock()
        # Use a path with invalid characters to force backup failure
        # VACUUM INTO with a quote in the path should fail
        bad_dir = tmp_path / "test'"  # Single quote is rejected by VACUUM INTO
        svc = MaintenanceService(
            engine=engine,
            session_factory=session_factory,
            notifier=notifier,
            data_dir=bad_dir,
        )
        success = await svc.run_daily_backup()
        assert not success
        notifier.send.assert_called_once()
