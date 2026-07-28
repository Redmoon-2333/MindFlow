"""Tests for MaintenanceService.

Covers:
  - cleanup_old_events: old events deleted, recent events preserved,
    batch deletion
  - run_daily_backup: backup file created, notification on failure
  - P1-5: WAL checkpoint after daily event cleanup
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from mindflow.infrastructure.repositories.activity import (
    activity_events,
)
from mindflow.services import maintenance_service as maintenance_module
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


# ═══════════════════════════════════════════════════════════════════════
# P1-5: WAL checkpoint after daily event cleanup
# ═══════════════════════════════════════════════════════════════════════


_WAL_TABLE_DEF = activity_events


class TestWalCheckpointAfterCleanup:
    """WAL truncation after cleanup_old_events."""

    async def test_checkpoint_truncates_wal_after_cleanup(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """After cleanup_old_events, the WAL file is truncated to zero bytes."""
        db_path = tmp_path / "test.db"
        wal_path = tmp_path / "test.db-wal"

        engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )

        async with engine.begin() as conn:
            await conn.run_sync(lambda c: c.execute(sa.text("PRAGMA journal_mode=WAL")))
            await conn.run_sync(lambda c: c.execute(sa.text("PRAGMA synchronous=NORMAL")))
            await conn.run_sync(_WAL_TABLE_DEF.metadata.create_all)
            for batch in range(3):
                rows = []
                for idx in range(200):
                    rows.append({
                        "id": f"old-{batch}-{idx}",
                        "user_id": 1,
                        "timestamp": "2026-01-01T00:00:00+00:00",
                        "duration_s": 1.0,
                        "data_json": json.dumps({
                            "app_name": "Test",
                            "window_title": "",
                            "process_name": "test.exe",
                            "is_idle": False,
                            "timestamp_utc": "2026-01-01T00:00:00+00:00",
                        }),
                        "event_type": "window_snapshot",
                    })
                await conn.execute(_WAL_TABLE_DEF.insert(), rows)

        assert wal_path.exists()
        wal_size_before = wal_path.stat().st_size
        assert wal_size_before > 0, "WAL must have content before cleanup"

        monkeypatch.setattr(maintenance_module, "_BATCH_SIZE", 50)

        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        svc = MaintenanceService(
            engine=engine,
            session_factory=session_factory,
            notifier=AsyncMock(),
            clock=lambda: datetime(2026, 7, 26, tzinfo=UTC),
        )

        deleted = await svc.cleanup_old_events(retention_days=30)
        assert deleted == 600

        wal_size_after = wal_path.stat().st_size
        assert wal_size_after == 0, (
            f"WAL should be truncated after cleanup, got {wal_size_after} bytes"
        )

        await engine.dispose()

    async def test_checkpoint_runs_even_with_zero_deletions(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When no events match retention, checkpoint still runs harmlessly."""
        db_path = tmp_path / "test.db"
        wal_path = tmp_path / "test.db-wal"

        engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )

        async with engine.begin() as conn:
            await conn.run_sync(lambda c: c.execute(sa.text("PRAGMA journal_mode=WAL")))
            await conn.run_sync(lambda c: c.execute(sa.text("PRAGMA synchronous=NORMAL")))
            await conn.run_sync(_WAL_TABLE_DEF.metadata.create_all)
            # Insert recent events — none match retention
            ts = datetime(2026, 7, 20, tzinfo=UTC).isoformat()
            for idx in range(50):
                await conn.execute(
                    _WAL_TABLE_DEF.insert().values(
                        id=f"recent-{idx}",
                        user_id=1,
                        timestamp=ts,
                        duration_s=1.0,
                        data_json=json.dumps({
                            "app_name": "Test",
                            "window_title": "",
                            "process_name": "test.exe",
                            "is_idle": False,
                            "timestamp_utc": ts,
                        }),
                        event_type="window_snapshot",
                    )
                )

        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        svc = MaintenanceService(
            engine=engine,
            session_factory=session_factory,
            notifier=AsyncMock(),
            clock=lambda: datetime(2026, 7, 26, tzinfo=UTC),
        )

        deleted = await svc.cleanup_old_events(retention_days=30)
        assert deleted == 0

        # Checkpoint still runs — WAL truncated
        wal_size_after = wal_path.stat().st_size
        assert wal_size_after == 0, (
            f"WAL should be truncated even with zero deletions, "
            f"got {wal_size_after} bytes"
        )

        await engine.dispose()

    async def test_checkpoint_failure_propagates(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the checkpoint operation fails, the exception propagates.

        The scheduler catches and logs exceptions from cron coroutines,
        so a checkpoint failure here is observable to the daily-maintenance
        caller.
        """
        db_path = tmp_path / "test.db"

        engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )

        async with engine.begin() as conn:
            await conn.run_sync(lambda c: c.execute(sa.text("PRAGMA journal_mode=WAL")))
            await conn.run_sync(lambda c: c.execute(sa.text("PRAGMA synchronous=NORMAL")))
            await conn.run_sync(_WAL_TABLE_DEF.metadata.create_all)
            for idx in range(50):
                await conn.execute(
                    _WAL_TABLE_DEF.insert().values(
                        id=f"old-{idx}",
                        user_id=1,
                        timestamp="2026-01-01T00:00:00+00:00",
                        duration_s=1.0,
                        data_json=json.dumps({
                            "app_name": "Test",
                            "window_title": "",
                            "process_name": "test.exe",
                            "is_idle": False,
                            "timestamp_utc": "2026-01-01T00:00:00+00:00",
                        }),
                        event_type="window_snapshot",
                    )
                )

        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        svc = MaintenanceService(
            engine=engine,
            session_factory=session_factory,
            notifier=AsyncMock(),
            clock=lambda: datetime(2026, 7, 26, tzinfo=UTC),
        )

        # Inject a failure into the checkpoint method itself
        async def _failing_checkpoint() -> None:
            raise OSError("disk full during checkpoint")

        with patch.object(svc, "_wal_checkpoint_truncate", _failing_checkpoint), \
             pytest.raises(OSError, match="disk full"):
            await svc.cleanup_old_events(retention_days=30)

        await engine.dispose()
