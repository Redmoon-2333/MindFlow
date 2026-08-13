"""Collector interval persistence tests.

Covers the runtime storage layer contract:
  - ``open()`` creates exactly one row with UTC ISO ``started_at``.
  - ``close()`` targets the exact interval by ``id`` and is idempotent —
    a second close must NOT rewrite terminal facts.
  - ``list_by_user`` / ``list_by_user_range`` are user-scoped, ordered,
    and survive repository reconstruction (fresh instance, same DB).
  - Migration 0021 chains from 0020 and cycles cleanly.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mindflow.infrastructure.repositories.collector_intervals import (
    CollectorIntervalsRepository,
)
from mindflow.infrastructure.schema import collector_intervals


async def _create_table(engine) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(collector_intervals.metadata.create_all)


async def test_open_creates_one_row_with_utc_started_at(
    engine, session_factory,
) -> None:
    await _create_table(engine)
    repository = CollectorIntervalsRepository(session_factory)

    record = await repository.open(1)

    assert record.user_id == 1
    assert record.ended_at is None
    assert record.reason is None
    assert record.manual_stop is False
    assert record.failure is False
    assert record.sleep is False
    assert record.last_error is None
    # UTC ISO8601 (ends with Z or carries an explicit +00:00 offset).
    started = datetime.fromisoformat(record.started_at.replace("Z", "+00:00"))
    assert started.tzinfo is not None

    async with session_factory() as session:
        rows = (
            await session.execute(
                collector_intervals.select().where(
                    collector_intervals.c.user_id == 1
                )
            )
        ).fetchall()
    assert len(rows) == 1
    assert rows[0].id == record.id


async def test_close_targets_exact_interval_once_and_is_idempotent(
    engine, session_factory,
) -> None:
    await _create_table(engine)
    repository = CollectorIntervalsRepository(session_factory)
    opened = await repository.open(1)

    closed = await repository.close(
        opened.id,
        reason="manual stop",
        manual_stop=True,
        failure=False,
        sleep=False,
        last_error=None,
    )
    assert closed is not None
    assert closed.ended_at is not None
    assert closed.reason == "manual stop"
    assert closed.manual_stop is True
    assert closed.failure is False

    first_ended_at = closed.ended_at
    # A second close must NOT rewrite terminal facts.
    again = await repository.close(
        opened.id,
        reason="overwritten",
        manual_stop=False,
        failure=True,
        sleep=True,
        last_error="boom",
    )
    assert again is not None
    assert again.ended_at == first_ended_at
    assert again.reason == "manual stop"
    assert again.manual_stop is True
    assert again.failure is False
    assert again.sleep is False
    assert again.last_error is None


async def test_close_unknown_interval_returns_none(
    engine, session_factory,
) -> None:
    await _create_table(engine)
    repository = CollectorIntervalsRepository(session_factory)
    assert await repository.close("no-such-interval") is None


async def test_list_by_user_is_scoped_and_ordered(
    engine, session_factory,
) -> None:
    await _create_table(engine)
    repository = CollectorIntervalsRepository(session_factory)
    base = datetime(2026, 8, 1, tzinfo=UTC)
    first = await repository.open(1, now=base)
    second = await repository.open(1, now=base + timedelta(minutes=5))
    third = await repository.open(1, now=base + timedelta(minutes=10))
    await repository.open(2, now=base + timedelta(minutes=1))  # other user

    listed = await repository.list_by_user(1)
    assert [r.id for r in listed] == [third.id, second.id, first.id]
    assert all(r.user_id == 1 for r in listed)

    limited = await repository.list_by_user(1, limit=2)
    assert [r.id for r in limited] == [third.id, second.id]


async def test_list_by_user_range_is_scoped_and_survives_reconstruction(
    engine, session_factory,
) -> None:
    await _create_table(engine)
    repository = CollectorIntervalsRepository(session_factory)
    base = datetime(2026, 8, 1, tzinfo=UTC)
    inside_1 = await repository.open(1, now=base)
    outside_before = await repository.open(1, now=base - timedelta(days=1))
    inside_2 = await repository.open(1, now=base + timedelta(minutes=5))
    await repository.open(1, now=base + timedelta(hours=2))
    await repository.open(2, now=base + timedelta(minutes=2))  # other user

    # Repository reconstruction: fresh instance over the same DB.
    rebuilt = CollectorIntervalsRepository(session_factory)
    ranged = await rebuilt.list_by_user_range(
        1, start=base, end=base + timedelta(minutes=10),
    )
    assert [r.id for r in ranged] == [inside_1.id, inside_2.id]
    assert outside_before.id not in [r.id for r in ranged]

    rebuilt_open = await rebuilt.list_by_user(1)
    assert len(rebuilt_open) == 4
    assert inside_1.id in {r.id for r in rebuilt_open}


def test_migration_0021_chains_from_0020_and_cycles_cleanly(tmp_path: Path) -> None:
    """Migration 0021 upgrades, downgrades one step, and upgrades again."""
    import alembic.command
    import alembic.config

    db_path = tmp_path / "migration_cycle.db"
    url = f"sqlite:///{db_path}"
    config = alembic.config.Config(str(Path(__file__).parents[1] / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", url)

    alembic.command.upgrade(config, "head")
    with sqlite3.connect(db_path) as conn:
        version = conn.execute("SELECT version_num FROM alembic_version").fetchone()[0]
        cols = {
            row[1] for row in conn.execute("PRAGMA table_info(collector_intervals)")
        }
        indexes = {
            row[1]
            for row in conn.execute("PRAGMA index_list(collector_intervals)")
        }
    # The chain head is now 0022 (feature-window columns added after 0021);
    # 0021 remains an intermediate link on the chain.
    assert version == "0022_feature_window_columns"
    assert cols == {
        "id", "user_id", "started_at", "ended_at", "reason",
        "manual_stop", "failure", "sleep", "last_error",
    }
    assert "idx_collector_intervals_user_started" in indexes

    # Downgrade one step → 0020, then upgrade again → clean cycle.
    alembic.command.downgrade(config, "0020_create_intervention_slot_reservations")
    with sqlite3.connect(db_path) as conn:
        version = conn.execute("SELECT version_num FROM alembic_version").fetchone()[0]
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
    assert version == "0020_create_intervention_slot_reservations"
    assert "collector_intervals" not in tables

    alembic.command.upgrade(config, "head")
    with sqlite3.connect(db_path) as conn:
        version = conn.execute("SELECT version_num FROM alembic_version").fetchone()[0]
        cols = {
            row[1] for row in conn.execute("PRAGMA table_info(collector_intervals)")
        }
    assert version == "0022_feature_window_columns"
    assert "last_error" in cols
