"""Regression tests for phase-2 grouping, weighting and data-quality work.

Covers:

* cross-midnight sessions grouped into one date block (no fold leakage);
* session-balanced weights (a long session cannot outvote a short one);
* auxiliary labels bounded by a budget instead of a fixed per-window weight;
* the window upsert never erasing a user label on a routine re-roll;
* missing ≠ zero: an unenabled collector is not "observed";
* break context neither supervises the state model nor loses its protection.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION
from mindflow.domain.label_contract import (
    is_state_supervision,
    normalise_context_type,
    normalise_goal_alignment,
    normalise_label_source,
    protects_from_reminders,
)
from mindflow.infrastructure.repositories.telemetry import TelemetryRepository
from mindflow.infrastructure.schema import metadata
from mindflow.services.window_quality import (
    SOURCE_ACTIVITY,
    SOURCE_BROWSER,
    build_window_quality,
)
from mindflow.train.grouping import build_grouping_plan
from mindflow.train.v2 import (
    V2_FEATURE_NAMES,
    prepare_v2_training_data,
)

# ── Helpers ─────────────────────────────────────────────────────────────


def _window(start: datetime, **overrides: float) -> dict[str, object]:
    features = {name: 0.0 for name in V2_FEATURE_NAMES}
    features.update({"top_app_ratio": 0.8, "input_active_ratio": 0.5})
    features.update(overrides)
    return {
        "window_start_utc": start.isoformat(),
        "window_end_utc": (start + timedelta(minutes=5)).isoformat(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features": features,
    }


def _feedback(
    session_id: str, start: datetime, end: datetime, label: str, score: int
) -> dict[str, object]:
    return {
        "session_id": session_id,
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "label": label,
        "score": score,
        "task_type": "coding",
    }


# ── Grouping ────────────────────────────────────────────────────────────


def test_cross_midnight_session_merges_its_dates() -> None:
    """A session spanning midnight must produce ONE group, not two."""
    session_start = datetime(2026, 7, 1, 23, 50, tzinfo=UTC)
    session_end = datetime(2026, 7, 2, 0, 30, tzinfo=UTC)
    windows = [
        _window(datetime(2026, 7, 1, 23, 50, tzinfo=UTC)),
        _window(datetime(2026, 7, 2, 0, 5, tzinfo=UTC)),
        _window(datetime(2026, 7, 3, 10, 0, tzinfo=UTC)),
    ]
    feedback = [_feedback("late", session_start, session_end, "focus", 5)]

    data = prepare_v2_training_data(windows, feedback)

    session_dates = data.session_dates["late"]
    assert len(session_dates) == 2, "the session really does span two dates"

    groups_for_session = {
        g for g, s in zip(data.group_ids, data.sample_feedback_ids, strict=True)
        if s == "late"
    }
    assert len(groups_for_session) == 1, "both dates must collapse to one group"

    # The unrelated day stays in its own group.
    assert len(set(data.group_ids)) == 2


def test_plan_merges_dates_transitively() -> None:
    plan = build_grouping_plan(
        dates=["d1", "d2", "d2", "d3"],
        labels=[1, 1, 0, 0],
        sources=["explicit", "explicit", "explicit", "explicit"],
        window_dates_by_session={"s1": ["d1", "d2"], "s2": ["d2", "d3"]},
        session_id_by_sample=["s1", "s1", "s2", "s2"],
    )
    # d2 links d1 and d3, so all three end up in one block.
    assert len(set(plan.group_ids)) == 1
    assert len(plan.merged_sessions) == 2


# ── Weighting ───────────────────────────────────────────────────────────


def test_session_weights_are_balanced_not_window_counted() -> None:
    """A 6-window session must not weigh 3x a 2-window session."""
    base = datetime(2026, 7, 1, 9, tzinfo=UTC)
    long_windows = [_window(base + timedelta(minutes=5 * i)) for i in range(6)]
    short_windows = [
        _window(datetime(2026, 7, 5, 9, tzinfo=UTC) + timedelta(minutes=5 * i))
        for i in range(2)
    ]
    feedback = [
        _feedback("long", base, base + timedelta(minutes=30), "focus", 5),
        _feedback(
            "short",
            datetime(2026, 7, 5, 9, tzinfo=UTC),
            datetime(2026, 7, 5, 9, 10, tzinfo=UTC),
            "distracted",
            1,
        ),
    ]

    data = prepare_v2_training_data(long_windows + short_windows, feedback)

    long_total = sum(
        w for w, s in zip(data.sample_weights, data.sample_feedback_ids, strict=True)
        if s == "long"
    )
    short_total = sum(
        w for w, s in zip(data.sample_weights, data.sample_feedback_ids, strict=True)
        if s == "short"
    )
    assert long_total == pytest.approx(short_total, rel=1e-6), (
        "each feedback session must contribute the same total weight"
    )
    assert long_total == pytest.approx(1.0, rel=1e-6)


def test_auxiliary_labels_share_a_bounded_budget() -> None:
    """Auxiliary labels cannot outweigh explicit feedback overall."""
    base = datetime(2026, 7, 1, 9, tzinfo=UTC)
    windows = [_window(base + timedelta(minutes=5 * i)) for i in range(4)]
    for i, w in enumerate(windows):
        w["id"] = f"w{i}"
    feedback = [_feedback("s1", base, base + timedelta(minutes=10), "focus", 5)]
    labels = {"w2": 1, "w3": 0}

    data = prepare_v2_training_data(windows, feedback, window_labels=labels)

    explicit_total = float(
        np.sum(data.sample_weights[data.explicit_mask])
    )
    aux_total = float(
        np.sum(
            data.sample_weights[
                data.window_label_mask if data.window_label_mask is not None
                else np.zeros(len(data.labels), dtype=bool)
            ]
        )
    )
    assert explicit_total > 0
    assert aux_total <= explicit_total * 1.0 + 1e-9


def test_weak_heuristics_carry_no_supervision_weight() -> None:
    """Heuristic labels are not evidence and must not train the gate."""
    base = datetime(2026, 7, 1, 9, tzinfo=UTC)
    # Deep-focus shape so `_weak_label` yields a real label (not the -1 that a
    # highly-idle window would produce, which is dropped entirely).
    windows = [
        _window(
            base + timedelta(minutes=5 * i),
            top_app_ratio=0.95,
            idle_ratio=0.05,
            app_switch_count=1,
        )
        for i in range(2)
    ]

    data = prepare_v2_training_data(windows, [])

    assert all(s == "weak" for s in data.label_sources)
    assert data.sample_weights.tolist() == [0.0, 0.0]
    assert data.train_mask is not None
    assert not data.train_mask.any(), "weak rows are never part of supervision"


# ── Data quality: missing ≠ zero ────────────────────────────────────────


def test_disabled_collector_is_not_observed() -> None:
    """A source that was switched off is never 'observed'."""
    start = datetime(2026, 7, 1, 9, tzinfo=UTC)
    quality = build_window_quality(
        window_start=start,
        window_end=start + timedelta(minutes=5),
        events=[],
        interaction_buckets=[],
        browser_segments=[],
        enabled={"activity": True, "browser": False, "input": True},
        available={"activity": True, "browser": False, "input": True},
    )

    assert quality.is_observed(SOURCE_BROWSER) is False
    assert quality.feature_observed("browser_ratio") is False
    # Activity was on and delivered, so its features are observable.
    assert quality.is_observed(SOURCE_ACTIVITY) is True


def test_zero_with_collector_on_is_an_observation() -> None:
    """The same zero IS a real observation when the collector was running."""
    start = datetime(2026, 7, 1, 9, tzinfo=UTC)
    quality = build_window_quality(
        window_start=start,
        window_end=start + timedelta(minutes=5),
        events=[],
        interaction_buckets=[],
        browser_segments=[],
        enabled={"activity": True, "browser": True, "input": True},
        available={"activity": True, "browser": True, "input": True},
    )

    assert quality.is_observed(SOURCE_BROWSER) is True
    assert quality.feature_observed("browser_ratio") is True


def test_coverage_and_gaps_are_measured_not_assumed() -> None:
    start = datetime(2026, 7, 1, 9, tzinfo=UTC)

    class _Event:
        def __init__(self, duration: float) -> None:
            self.duration_s = duration
            self.timestamp_utc = start

    quality = build_window_quality(
        window_start=start,
        window_end=start + timedelta(minutes=5),
        events=[_Event(150.0)],           # half the 300s window
        interaction_buckets=[],
        browser_segments=[],
        enabled={"activity": True},
        available={"activity": True},
    )

    assert quality.coverage[SOURCE_ACTIVITY] == pytest.approx(0.5, abs=1e-3)
    assert quality.uncovered_seconds[SOURCE_ACTIVITY] == pytest.approx(150.0, abs=1e-3)
    assert quality.gap_seconds[SOURCE_ACTIVITY] is None
    assert quality.quality_tier in {"activity_only", "partial"}


def test_empty_window_is_tier_empty() -> None:
    start = datetime(2026, 7, 1, 9, tzinfo=UTC)
    quality = build_window_quality(
        window_start=start,
        window_end=start + timedelta(minutes=5),
        events=[],
        interaction_buckets=[],
        browser_segments=[],
    )
    assert quality.quality_tier == "empty"


# ── Label contract ──────────────────────────────────────────────────────


def test_unknown_is_never_invented() -> None:
    """Missing context stays unknown; nothing is defaulted in."""
    assert normalise_context_type(None) == "unknown"
    assert normalise_context_type("") == "unknown"
    assert normalise_context_type("nonsense") == "unknown"
    assert normalise_goal_alignment(None) == "unknown"
    assert normalise_label_source(None) == "unknown"
    assert normalise_label_source("gpt") == "unknown"


def test_break_does_not_supervise_state_but_protects_reminders() -> None:
    assert is_state_supervision("break", "focus") is False
    assert is_state_supervision("break", "distracted") is False
    assert protects_from_reminders("break") is True
    assert protects_from_reminders("task_execution") is False


def test_mixed_label_does_not_supervise_state() -> None:
    assert is_state_supervision("task_execution", "mixed") is False
    assert is_state_supervision("task_execution", "focus") is True


# ── Window upsert must not erase labels ─────────────────────────────────


@pytest.fixture
async def telemetry_repo(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'q.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
    try:
        yield TelemetryRepository(
            session_factory=async_sessionmaker(engine, expire_on_commit=False)
        )
    finally:
        await engine.dispose()


async def test_reroll_preserves_existing_label(telemetry_repo) -> None:
    """The scheduler re-rolls ranges; that must not delete user labels."""
    start = datetime(2026, 7, 1, 9, tzinfo=UTC)
    row = {
        "user_id": 1,
        "window_start_utc": start,
        "window_end_utc": start + timedelta(minutes=5),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features_json": json.dumps({"feature_schema_version": FEATURE_SCHEMA_VERSION}),
        "label": "focus",
        "quality_json": json.dumps({"quality_tier": "full"}),
    }
    await telemetry_repo.upsert_feature_windows([row])

    # A routine re-roll submits no new label.
    reroll = dict(row)
    reroll["label"] = None
    reroll["features_json"] = json.dumps(
        {"feature_schema_version": FEATURE_SCHEMA_VERSION, "changed": 1},
    )
    await telemetry_repo.upsert_feature_windows([reroll])

    rows = await telemetry_repo.list_feature_windows(
        1, feature_schema_version=FEATURE_SCHEMA_VERSION,
    )
    assert len(rows) == 1
    stored = rows[0]
    assert stored["label"] == "focus", "the user's label must survive a re-roll"
    assert "changed" in stored["features_json"], "features still refresh"


async def test_reroll_can_set_a_label_when_none_existed(telemetry_repo) -> None:
    start = datetime(2026, 7, 1, 9, tzinfo=UTC)
    base = {
        "user_id": 1,
        "window_start_utc": start,
        "window_end_utc": start + timedelta(minutes=5),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features_json": json.dumps({"feature_schema_version": FEATURE_SCHEMA_VERSION}),
        "label": None,
    }
    await telemetry_repo.upsert_feature_windows([base])

    labelled = dict(base, label="distracted")
    await telemetry_repo.upsert_feature_windows([labelled])

    rows = await telemetry_repo.list_feature_windows(
        1, feature_schema_version=FEATURE_SCHEMA_VERSION,
    )
    assert rows[0]["label"] == "distracted"


async def test_quality_record_is_persisted(telemetry_repo) -> None:
    start = datetime(2026, 7, 1, 9, tzinfo=UTC)
    await telemetry_repo.upsert_feature_windows([{
        "user_id": 1,
        "window_start_utc": start,
        "window_end_utc": start + timedelta(minutes=5),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features_json": json.dumps({"feature_schema_version": FEATURE_SCHEMA_VERSION}),
        "label": None,
        "quality_json": json.dumps({"quality_tier": "partial", "coverage": {}}),
    }])

    rows = await telemetry_repo.list_feature_windows(
        1, feature_schema_version=FEATURE_SCHEMA_VERSION,
    )
    assert rows[0]["quality_json"] is not None
    assert json.loads(rows[0]["quality_json"])["quality_tier"] == "partial"
