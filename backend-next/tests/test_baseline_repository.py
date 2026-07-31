"""BaselineRepository upsert persistence tests.

Covers the true user-keyed upsert (one row per ``user_id`` via the existing
``UNIQUE(user_id)`` constraint):

  - create then update yields exactly one row, preserving the first-insert
    creation time while the model JSON and ``updated_at`` move forward
  - a persisted V2 baseline round-trips through ``get_latest`` with schema
    version, timezone and sample counts intact
  - sequential idempotent upserts never violate the unique constraint
  - the database itself rejects a second row for one user
  - a serialization failure inside the write rolls the transaction back,
    leaving the previously persisted row untouched
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker

from mindflow.domain.baseline import BaselineModel
from mindflow.domain.feature_schema import V2_FEATURE_NAMES
from mindflow.infrastructure.repositories.baseline import (
    BaselineRepository,
    baseline_models,
)

# ── Helpers ──────────────────────────────────────────────────────────────


def _v2_row(
    hour: int = 9,
    dow: int = 0,
    app_switch_count: float = 15.0,
    active_seconds_ratio: float = 0.5,
) -> dict[str, Any]:
    """One flattened V2 feature-window row bucketed at local (hour, dow)."""
    local = datetime(2026, 7, 27, hour, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    shift = (dow - local.weekday()) % 7
    start = (local + timedelta(days=shift)).astimezone(UTC)
    features = {name: 0.0 for name in V2_FEATURE_NAMES}
    features["app_switch_count"] = app_switch_count
    features["active_seconds_ratio"] = active_seconds_ratio
    return {
        "window_start_utc": start.isoformat(),
        "features_json": json.dumps(features),
    }


async def _create_tables(engine: Any) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(baseline_models.metadata.create_all)


async def _count_rows(
    session_factory: async_sessionmaker,
) -> int:
    stmt = sa.select(sa.func.count()).select_from(baseline_models)
    async with session_factory() as session:
        count = (await session.execute(stmt)).scalar()
    return int(count or 0)


async def _fetch_row(
    session_factory: async_sessionmaker,
    user_id: int,
) -> Any:
    stmt = (
        sa.select(
            baseline_models.c.model_json,
            baseline_models.c.created_at,
            baseline_models.c.updated_at,
            baseline_models.c.training_events_count,
        )
        .where(baseline_models.c.user_id == user_id)
    )
    async with session_factory() as session:
        result = await session.execute(stmt)
        return result.mappings().first()


# ── Tests ────────────────────────────────────────────────────────────────


async def test_create_then_update_keeps_one_row_and_creation_time(
    engine, session_factory,
) -> None:
    """Given: no baseline. When: upsert, then upsert again after new windows.
    Then: exactly one row; created_at untouched; model JSON/updated_at move on."""
    await _create_tables(engine)
    repository = BaselineRepository(session_factory=session_factory)

    assert await repository.get_latest(user_id=1) is None

    model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
    model.update([_v2_row(app_switch_count=10.0)])
    await repository.upsert(model)
    birth = model.created_at
    first_updated_at = model.updated_at.isoformat()

    model.update([_v2_row(hour=14, dow=3, app_switch_count=20.0)])
    await repository.upsert(model)

    assert await _count_rows(session_factory) == 1
    row = await _fetch_row(session_factory, 1)
    assert row is not None
    assert row["created_at"] == birth.isoformat()
    assert row["updated_at"] != first_updated_at

    reloaded = await repository.get_latest(user_id=1)
    assert reloaded is not None
    assert reloaded.created_at == birth
    # 2 windows x 24 v2 features
    assert reloaded.total_samples() == 2 * len(V2_FEATURE_NAMES)


async def test_upsert_round_trip_preserves_v2_state(engine, session_factory) -> None:
    """Given: a V2 model in Asia/Shanghai. When: upsert once.
    Then: reload preserves schema version, timezone, days and Welford means."""
    await _create_tables(engine)
    repository = BaselineRepository(session_factory=session_factory)

    model = BaselineModel(user_id=7, timezone="Asia/Shanghai")
    model.update([_v2_row(app_switch_count=10.0), _v2_row(app_switch_count=12.0)])
    await repository.upsert(model)

    reloaded = await repository.get_latest(user_id=7)
    assert reloaded is not None
    assert reloaded.user_id == 7
    assert reloaded.FEATURE_SCHEMA_VERSION == 3
    assert reloaded.timezone == "Asia/Shanghai"
    assert reloaded.total_days == 1
    assert reloaded.total_samples() == 2 * len(V2_FEATURE_NAMES)
    assert reloaded.overall_mean("app_switch_count") == pytest.approx(11.0)
    assert reloaded.overall_mean("active_seconds_ratio") == pytest.approx(0.5)


async def test_sequential_idempotent_upserts_never_add_a_row(
    engine, session_factory,
) -> None:
    """Given: a persisted baseline. When: five fresh models upserted in order.
    Then: still one row, first-insert created_at retained, latest content wins."""
    await _create_tables(engine)
    repository = BaselineRepository(session_factory=session_factory)

    first = BaselineModel(user_id=1, timezone="Asia/Shanghai")
    first.update([_v2_row(app_switch_count=10.0)])
    await repository.upsert(first)
    row_after_first = await _fetch_row(session_factory, 1)
    assert row_after_first is not None

    for _ in range(5):
        fresh = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        fresh.update([_v2_row(hour=14, dow=3, app_switch_count=30.0)])
        await repository.upsert(fresh)

    assert await _count_rows(session_factory) == 1
    row = await _fetch_row(session_factory, 1)
    assert row is not None
    assert row["created_at"] == row_after_first["created_at"]

    reloaded = await repository.get_latest(user_id=1)
    assert reloaded is not None
    # latest model carries exactly one fresh window
    assert reloaded.total_samples() == len(V2_FEATURE_NAMES)
    assert reloaded.overall_mean("app_switch_count") == pytest.approx(30.0)


async def test_database_enforces_unique_user_id(engine, session_factory) -> None:
    """Given: a persisted baseline. When: a second row is inserted directly.
    Then: UNIQUE(user_id) raises IntegrityError."""
    await _create_tables(engine)
    repository = BaselineRepository(session_factory=session_factory)

    model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
    model.update([_v2_row()])
    await repository.upsert(model)

    with pytest.raises(IntegrityError):
        async with session_factory() as session:
            await session.execute(
                sa.insert(baseline_models).values(
                    id="second-row-for-user-1",
                    user_id=1,
                    model_json="{}",
                    training_events_count=0,
                    created_at="2026-07-27T00:00:00+00:00",
                    updated_at="2026-07-27T00:00:00+00:00",
                )
            )
            await session.commit()

    assert await _count_rows(session_factory) == 1


async def test_invalid_model_serialization_rolls_back_leaving_prior_row(
    engine, session_factory, monkeypatch,
) -> None:
    """Given: a persisted baseline. When: an upsert whose model cannot be
    JSON-encoded fails mid-transaction. Then: the prior row is unchanged."""
    await _create_tables(engine)
    repository = BaselineRepository(session_factory=session_factory)

    model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
    model.update([_v2_row(app_switch_count=10.0)])
    await repository.upsert(model)
    before = await _fetch_row(session_factory, 1)
    assert before is not None

    def broken_to_dict(self: BaselineModel) -> dict[str, Any]:
        # A set is not JSON-serializable -> json.dumps raises inside the txn.
        return {"user_id": self.user_id, "boom": {1, 2, 3}}

    monkeypatch.setattr(BaselineModel, "to_dict", broken_to_dict)
    with pytest.raises(TypeError):
        await repository.upsert(BaselineModel(user_id=1, timezone="Asia/Shanghai"))

    after = await _fetch_row(session_factory, 1)
    assert after is not None
    assert after["model_json"] == before["model_json"]
    assert after["created_at"] == before["created_at"]
    assert after["updated_at"] == before["updated_at"]
    assert await _count_rows(session_factory) == 1
