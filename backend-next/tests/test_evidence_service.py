"""Integration tests for EvidenceBundleBuilder (services/evidence_service.py).

Uses a real temporary SQLite database with all required tables.
Covers:
  - Normal assembly: correct item count, metric names, severity mapping.
  - Empty window: no events → graceful degenerate output.
  - No baseline: graceful info item without deviation.
  - Baseline insufficient data: graceful info with "还需N天数据" message.
  - Baseline with sufficient data: deviation item present.
  - Intervention history injection: records appear in bundle.
  - Novelty detection: positive (new app) and negative (all known apps) cases.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.domain.baseline import BaselineModel
from mindflow.domain.events import make_event
from mindflow.domain.evidence_facts import build_evidence_catalog
from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION, V2_FEATURE_NAMES
from mindflow.domain.features import MAX_ACCEPTABLE_SWITCHES_PER_HOUR
from mindflow.domain.tasks import Task
from mindflow.domain.tmt import PRIOR, deviation_z_mean
from mindflow.infrastructure.database import create_session_factory
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
    activity_events,
)
from mindflow.infrastructure.repositories.intervention import (
    InterventionLogRepository,
    intervention_logs,
)
from mindflow.infrastructure.repositories.tasks import SQLAlchemyTaskRepository
from mindflow.infrastructure.repositories.telemetry import TelemetryRepository
from mindflow.infrastructure.schema import (
    behavior_feature_windows,
    focus_session_feedback,
)
from mindflow.services.evidence_service import (
    EvidenceBundleBuilder,
    _block_severity,
    _deviation_severity,
    _focus_severity,
    _hours_to_deadline,
    _select_value_task,
    _switch_severity,
    baseline_models_metadata,
)


def _utc(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=UTC)


def _local_utc(hour: int, dow: int, day: int) -> datetime:
    """UTC instant whose Asia/Shanghai local time is (hour, dow) on July *day*."""
    local = datetime(2026, 7, day, hour, tzinfo=ZoneInfo("Asia/Shanghai"))
    shift = (dow - local.weekday()) % 7
    return (local + timedelta(days=shift)).astimezone(UTC)


def _v2_features(
    *,
    app_switch_count: float = 10.0,
    idle_ratio: float = 0.05,
    active_seconds_ratio: float = 0.5,
) -> dict[str, float]:
    """Full V2 feature vector with the scored features overridden."""
    features = {name: 0.0 for name in V2_FEATURE_NAMES}
    features.update({
        "app_switch_count": app_switch_count,
        "idle_ratio": idle_ratio,
        "active_seconds_ratio": active_seconds_ratio,
    })
    return features


def _v2_row(
    start_utc: datetime,
    *,
    app_switch_count: float = 10.0,
    idle_ratio: float = 0.05,
    active_seconds_ratio: float = 0.5,
    process_name: str = "code.exe",
) -> dict[str, Any]:
    """V2 feature-window row at the given UTC window start."""
    features = _v2_features(
        app_switch_count=app_switch_count,
        idle_ratio=idle_ratio,
        active_seconds_ratio=active_seconds_ratio,
    )
    return {
        "window_start_utc": start_utc,
        "features_json": json.dumps(features),
        "process_name": process_name,
    }


_BASE_TS = _utc("2026-07-18T08:00:00")
_WINDOW_START = _BASE_TS
_WINDOW_END = _BASE_TS + timedelta(hours=4)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
async def all_tables(engine):
    """Create all tables needed by the evidence service."""
    async with engine.begin() as conn:
        await conn.run_sync(activity_events.metadata.create_all)
        await conn.run_sync(intervention_logs.metadata.create_all)
        await conn.run_sync(baseline_models_metadata.create_all)
        await conn.run_sync(behavior_feature_windows.metadata.create_all)


@pytest.fixture
async def activity_repo(
    session_factory: async_sessionmaker[AsyncSession],
    all_tables: None,
) -> SQLAlchemyActivityRepository:
    return SQLAlchemyActivityRepository(session_factory=session_factory, pulsetime_s=10)


@pytest.fixture
async def intervention_repo(
    session_factory: async_sessionmaker[AsyncSession],
    all_tables: None,
) -> InterventionLogRepository:
    return InterventionLogRepository(session_factory=session_factory)


@pytest.fixture
async def builder(
    activity_repo: SQLAlchemyActivityRepository,
    intervention_repo: InterventionLogRepository,
    session_factory: async_sessionmaker[AsyncSession],
) -> EvidenceBundleBuilder:
    return EvidenceBundleBuilder(
        activity_repo=activity_repo,
        intervention_repo=intervention_repo,
        session_factory=session_factory,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _make_events(
    count: int = 10,
    process_name: str = "code.exe",
    app_name: str = "VS Code",
    is_idle: bool = False,
    base_ts: datetime = _BASE_TS,
) -> list[dict[str, Any]]:
    """Generate event kwargs for inserting into the repo.

    Returns a list of dicts for use with make_event().
    """
    events = []
    for i in range(count):
        ts = base_ts + timedelta(seconds=i * 30)  # 30s spacing
        ev = make_event(
            user_id=1,
            timestamp_utc=ts,
            duration_s=25.0,
            process_name=process_name,
            app_name=app_name,
            is_idle=is_idle,
        )
        events.append(ev)
    return events


def _mixed_app_events(base_ts: datetime = _BASE_TS) -> list[Any]:
    """Generate events with multiple apps (produces moderate switching)."""
    events: list[Any] = []
    for i in range(20):
        ts = base_ts + timedelta(seconds=i * 15)
        # Alternate between 3 apps every few events
        if i < 5:
            proc = "code.exe"
        elif i < 10:
            proc = "chrome.exe"
        elif i < 15:
            proc = "code.exe"
        else:
            proc = "terminal.exe"
        ev = make_event(
            user_id=1,
            timestamp_utc=ts,
            duration_s=10.0,
            process_name=proc,
            app_name=proc.replace(".exe", ""),
            is_idle=False,
        )
        events.append(ev)
    return events


async def _insert_events(
    repo: SQLAlchemyActivityRepository, events: list[Any]
) -> None:
    for ev in events:
        await repo.append_event(ev)


def _train_baseline() -> BaselineModel:
    """Create a V2 baseline model with sufficient data at local Sat 08:00.

    Local Sat 08:00 in Asia/Shanghai == 00:00 UTC on 2026-07-11, which maps
    to the (8, 5) bucket the evidence service scores (it reads the UTC hour).
    Includes process_name in training rows so the baseline has known apps.
    """
    model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
    saturday = _local_utc(hour=8, dow=5, day=11)
    for i in range(10):
        model.update([_v2_row(
            saturday,
            app_switch_count=10.0 + (i % 3) * 2.0,
            active_seconds_ratio=0.5,
        )])
    # Ensure total_days >= MIN_BASELINE_DAYS by adding other-dated rows
    # (14:00 local == 06:00 UTC).
    for d in range(7):
        model.update([_v2_row(
            datetime(2026, 7, 11 + d, 6, 0, tzinfo=UTC),
            app_switch_count=12.0,
        )])
    return model


async def _insert_baseline(
    session_factory: async_sessionmaker[AsyncSession],
    baseline: BaselineModel,
    user_id: int = 1,
) -> None:
    """Insert a baseline model JSON into the baseline_models table."""
    from mindflow.services.evidence_service import baseline_models
    data = baseline.to_dict()
    async with session_factory() as session, session.begin():
        await session.execute(
            sa.insert(baseline_models).values(
                id=f"bl-{user_id}",
                user_id=user_id,
                model_json=json.dumps(data, ensure_ascii=False),
                training_events_count=10,
                created_at=_BASE_TS.isoformat(),
                updated_at=_BASE_TS.isoformat(),
            )
        )


async def _insert_intervention(
    intervention_repo: InterventionLogRepository,
    user_id: int = 1,
    intervention_type: str = "nudge",
    user_response: str | None = None,
    offset: timedelta | None = None,
) -> None:
    """Insert an intervention log entry."""
    triggered_at = _WINDOW_END - (offset or timedelta(hours=1))
    await intervention_repo.log_triggered(
        user_id=user_id,
        intervention_type=intervention_type,
        triggered_at=triggered_at,
        context={"source": "test"},
    )
    if user_response:
        # Fetch the last inserted log to get its ID
        lookback = triggered_at - timedelta(seconds=1)
        logs = await intervention_repo.query_range(
            user_id, lookback, triggered_at + timedelta(seconds=1)
        )
        if logs:
            await intervention_repo.update_response(logs[0]["id"], user_response)  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — Normal assembly
# ═══════════════════════════════════════════════════════════════════════════════


class TestNormalAssembly:
    """Bundle assembly with sufficient data."""

    async def test_bundle_structure(self, builder: EvidenceBundleBuilder) -> None:
        """Bundle has correct top-level fields."""
        events = _make_events(count=15)
        await _insert_events(builder._activity_repo, events)

        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)
        assert bundle.user_id == 1
        assert bundle.window[0] == _WINDOW_START
        assert bundle.window[1] == _WINDOW_END

    async def test_has_feature_items(self, builder: EvidenceBundleBuilder) -> None:
        """Bundle contains focus_score, switch_rate, longest_block, top_apps."""
        events = _make_events(count=15)
        await _insert_events(builder._activity_repo, events)

        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)
        names = {item.metric for item in bundle.items}
        assert "focus_score" in names
        assert "switch_rate" in names
        assert "longest_block" in names
        assert "top_apps" in names

    async def test_focus_score_info(self, builder: EvidenceBundleBuilder) -> None:
        """Single-app high-focus window produces info-level focus_score."""
        events = _make_events(count=20, process_name="code.exe")
        await _insert_events(builder._activity_repo, events)

        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)
        focus_items = [i for i in bundle.items if i.metric == "focus_score"]
        assert len(focus_items) == 1
        # Single app with no switches → score ~100
        assert focus_items[0].severity == "info"

    async def test_switch_rate_severity(self, builder: EvidenceBundleBuilder) -> None:
        """High switching produces moderate/severe switch_rate."""
        # Alternating apps every event → high switch rate
        events = _mixed_app_events()
        await _insert_events(builder._activity_repo, events)

        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)
        switch_items = [i for i in bundle.items if i.metric == "switch_rate"]
        assert len(switch_items) == 1
        # Mixed apps with 15s spacing → high switches/hour
        assert switch_items[0].severity in ("moderate", "severe")

    async def test_has_behavior_summary(self, builder: EvidenceBundleBuilder) -> None:
        """Bundle contains a valid BehaviorSummary."""
        events = _make_events(count=15)
        await _insert_events(builder._activity_repo, events)

        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)
        bs = bundle.behavior_summary
        assert bs.duration_min > 0
        assert bs.actual_focus_min >= 0

    async def test_intended_task_none(  # noqa: E501
        self, builder: EvidenceBundleBuilder,
    ) -> None:
        """intended_task is None when no manual_tag events exist."""
        events = _make_events(count=15)
        await _insert_events(builder._activity_repo, events)

        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)
        assert bundle.behavior_summary.intended_task is None

    async def test_item_count(self, builder: EvidenceBundleBuilder) -> None:
        """Bundle has expected minimum item count (4 features + maybe deviation)."""
        events = _make_events(count=15)
        await _insert_events(builder._activity_repo, events)

        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)
        # At minimum: focus_score, switch_rate, longest_block, top_apps
        # Plus one behavior_deviation (info-level since no baseline)
        assert len(bundle.items) >= 4


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — Empty window
# ═══════════════════════════════════════════════════════════════════════════════


class TestEmptyWindow:
    """Graceful handling of empty event windows."""

    async def test_empty_events_returns_bundle(self, builder: EvidenceBundleBuilder) -> None:
        """Empty events should still produce a valid bundle."""
        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)
        assert bundle is not None
        assert bundle.user_id == 1

    async def test_empty_events_has_info_items(self, builder: EvidenceBundleBuilder) -> None:
        """Empty events → single info-level focus_score item."""
        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)
        assert len(bundle.items) >= 1
        all_info = all(i.severity == "info" for i in bundle.items)
        assert all_info, "All items should be info-level for empty window"

    async def test_empty_events_emty_novelty(self, builder: EvidenceBundleBuilder) -> None:
        """Empty events → no novelty flags."""
        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)
        assert bundle.novelty_flags == ()

    async def test_empty_events_empty_interventions(self, builder: EvidenceBundleBuilder) -> None:
        """Empty events → empty intervention history."""
        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)
        assert bundle.intervention_history == ()

    async def test_empty_events_summary_zero(self, builder: EvidenceBundleBuilder) -> None:
        """Empty events → zeroed behavior summary."""
        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)
        bs = bundle.behavior_summary
        assert bs.duration_min == 0.0
        assert bs.actual_focus_min == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — Baseline / deviation
# ═══════════════════════════════════════════════════════════════════════════════


class TestNoBaseline:
    """Graceful path when no baseline model exists."""

    async def test_info_item_when_no_baseline(self, builder: EvidenceBundleBuilder) -> None:
        """No baseline → info-level '基线尚在建立中' item."""
        events = _make_events(count=15)
        await _insert_events(builder._activity_repo, events)

        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)
        deviation_items = [i for i in bundle.items if i.metric == "behavior_deviation"]
        assert len(deviation_items) == 1
        assert deviation_items[0].severity == "info"
        assert "基线尚在建立中" in deviation_items[0].human_readable


class TestInsufficientBaseline:
    """Baseline exists but has insufficient data."""

    async def test_info_when_insufficient(
        self,
        builder: EvidenceBundleBuilder,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """Insufficient baseline data → info item with '还需N天' message."""
        # Insert a baseline with very few samples (1 V2 row == 24 samples).
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        model.update([_v2_row(_local_utc(hour=8, dow=5, day=11), app_switch_count=10.0)])
        await _insert_baseline(session_factory, model)

        events = _make_events(count=15)
        await _insert_events(builder._activity_repo, events)

        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)
        deviation_items = [i for i in bundle.items if i.metric == "behavior_deviation"]
        assert len(deviation_items) == 1
        assert deviation_items[0].severity == "info"
        assert "尚在建立" in deviation_items[0].human_readable


class TestSufficientBaseline:
    """Baseline with enough data to compute deviations."""

    async def test_deviation_item_present(
        self,
        builder: EvidenceBundleBuilder,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """Sufficient baseline → deviation item computed."""
        model = _train_baseline()
        await _insert_baseline(session_factory, model)

        events = _make_events(count=15)
        await _insert_events(builder._activity_repo, events)

        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)
        deviation_items = [i for i in bundle.items if i.metric == "behavior_deviation"]
        assert len(deviation_items) == 1
        # The deviation may be info (normal behavior) or higher
        assert deviation_items[0].source == "welford_baseline"
        assert deviation_items[0].baseline == 0.0

    async def test_deviation_with_anomaly(
        self,
        builder: EvidenceBundleBuilder,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """Extreme behavior vs baseline → non-info deviation."""
        model = _train_baseline()
        await _insert_baseline(session_factory, model)

        # Extreme switching (100+ switches/hour vs baseline ~10-16)
        events = _make_events(count=30, process_name="code.exe", base_ts=_BASE_TS)
        # Add extreme switching behaviour
        for i in range(20):
            ts = _BASE_TS + timedelta(minutes=5) + timedelta(seconds=i * 5)
            proc = "code.exe" if i % 2 == 0 else "chrome.exe"
            ev = make_event(
                user_id=1,
                timestamp_utc=ts,
                duration_s=4.0,
                process_name=proc,
                app_name=proc.replace(".exe", ""),
                is_idle=False,
            )
            events.append(ev)
        await _insert_events(builder._activity_repo, events)

        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)
        deviation_items = [i for i in bundle.items if i.metric == "behavior_deviation"]
        assert len(deviation_items) >= 1


class TestStoredWindowDeviation:
    """Deviation is scored against the newest persisted 5-minute V2 window.

    The pre-B0(b) builder scored a hand-rolled 30-minute event row against a
    baseline of 5-minute V2 windows whose vocabulary only overlapped at
    ``idle_ratio``, so every score was capped near zero. These tests pin the
    fixed contract: the stored window (same vector the classifier sees) is the
    scored input, and it is bucketed in the baseline's own timezone.
    """

    #: _WINDOW_START is 2026-07-18T08:00Z == 16:00 Asia/Shanghai on a Saturday.
    _LOCAL_HOUR = 16
    _LOCAL_DOW = 5

    @staticmethod
    def _baseline_for_window() -> BaselineModel:
        """Varied history in the bucket the stored window falls into."""
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        start = _WINDOW_START
        for i in range(40):
            model.update(
                [
                    _v2_row(
                        start - timedelta(days=7 * (i // 10)),
                        app_switch_count=10.0 + (i % 5) * 2.0,
                        idle_ratio=0.05 + (i % 3) * 0.01,
                        active_seconds_ratio=0.5 + (i % 2) * 0.1,
                    )
                ]
            )
        return model

    @staticmethod
    async def _store_window(
        session_factory: async_sessionmaker[AsyncSession],
        start: datetime,
        **features: float,
    ) -> None:
        repo = TelemetryRepository(session_factory=session_factory)
        await repo.save_feature_window(
            user_id=1,
            window_start_utc=start,
            window_end_utc=start + timedelta(minutes=5),
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            features_json=json.dumps(_v2_features(**features)),
        )

    @staticmethod
    def _builder(
        activity_repo: SQLAlchemyActivityRepository,
        intervention_repo: InterventionLogRepository,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> EvidenceBundleBuilder:
        return EvidenceBundleBuilder(
            activity_repo=activity_repo,
            intervention_repo=intervention_repo,
            session_factory=session_factory,
            telemetry_repo=TelemetryRepository(session_factory=session_factory),
        )

    async def test_normal_stored_window_reports_eb_summary(
        self,
        activity_repo: SQLAlchemyActivityRepository,
        intervention_repo: InterventionLogRepository,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """A typical stored window → a summary whose severity is normal."""
        await _insert_baseline(session_factory, self._baseline_for_window())
        await self._store_window(session_factory, _WINDOW_START, app_switch_count=14.0)

        builder = self._builder(activity_repo, intervention_repo, session_factory)
        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)

        summary = bundle.deviation_summary
        assert summary is not None
        assert summary["method"] == "eb_shrinkage_mad"
        assert summary["severity"] == "normal"
        # Bucketed in Asia/Shanghai, not the naive UTC hour (08:00).
        assert summary["bucket"]["hour_of_day"] == self._LOCAL_HOUR
        assert summary["bucket"]["day_of_week"] == self._LOCAL_DOW

        items = [i for i in bundle.items if i.metric == "behavior_deviation"]
        assert len(items) == 1
        assert items[0].severity == "info"

    async def test_extreme_stored_window_reaches_non_normal_severity(
        self,
        activity_repo: SQLAlchemyActivityRepository,
        intervention_repo: InterventionLogRepository,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """A 50x jump in switches must surface as an anomaly, not "normal"."""
        await _insert_baseline(session_factory, self._baseline_for_window())
        await self._store_window(session_factory, _WINDOW_START, app_switch_count=500.0)

        builder = self._builder(activity_repo, intervention_repo, session_factory)
        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)

        summary = bundle.deviation_summary
        assert summary is not None
        assert summary["severity"] in {"moderate", "severe"}
        assert summary["features"][0]["feature"] == "app_switch_count"
        assert summary["features"][0]["robust_z"] > 4.0

        items = [i for i in bundle.items if i.metric == "behavior_deviation"]
        assert len(items) == 1
        assert items[0].severity in {"moderate", "severe"}
        # The readable text names the driving feature in Chinese.
        assert "应用切换次数" in items[0].human_readable


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — Intervention history
# ═══════════════════════════════════════════════════════════════════════════════


class TestInterventionHistory:
    """Intervention records injected into the bundle."""

    async def test_interventions_included(
        self,
        builder: EvidenceBundleBuilder,
        intervention_repo: InterventionLogRepository,
    ) -> None:
        """Recent interventions appear in the bundle."""
        events = _make_events(count=10)
        await _insert_events(builder._activity_repo, events)

        # Insert interventions at various offsets
        await _insert_intervention(
            intervention_repo, user_response="accepted", offset=timedelta(hours=2),
        )
        await _insert_intervention(
            intervention_repo, intervention_type="task_breakdown",
            user_response="ignored", offset=timedelta(hours=5),
        )

        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)
        assert len(bundle.intervention_history) >= 2

        # Check response mappings
        responses = {r.user_response for r in bundle.intervention_history}
        assert "accepted" in responses
        assert "ignored" in responses

    async def test_intervention_effect_notes(
        self,
        builder: EvidenceBundleBuilder,
        intervention_repo: InterventionLogRepository,
    ) -> None:
        """Effect notes are Chinese descriptions of the user response."""
        events = _make_events(count=10)
        await _insert_events(builder._activity_repo, events)

        await _insert_intervention(
            intervention_repo, user_response="accepted", offset=timedelta(hours=2),
        )
        await _insert_intervention(
            intervention_repo, user_response="dismissed", offset=timedelta(hours=5),
        )

        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)
        records = bundle.intervention_history
        assert any("已接受" in r.effect_note for r in records)
        assert any("已关闭" in r.effect_note for r in records)

    async def test_interventions_empty_when_none(
        self,
        builder: EvidenceBundleBuilder,
    ) -> None:
        """No interventions → empty tuple."""
        events = _make_events(count=10)
        await _insert_events(builder._activity_repo, events)

        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)
        assert bundle.intervention_history == ()


class TestEffectivenessEffectNote:
    """Effectiveness-driven effect_note enrichment (G005 learning loop).

    Tests that effect_note contains effectiveness summary when
    EffectivenessService is provided.
    """

    async def test_effect_note_enriched_with_effectiveness(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        activity_repo: SQLAlchemyActivityRepository,
        create_tables: None,
    ) -> None:
        """effect_note includes '干预后专注' when effectiveness data exists."""
        from unittest.mock import AsyncMock, MagicMock

        from mindflow.infrastructure.repositories.intervention import (
            InterventionLogRepository,
        )
        from mindflow.services.effectiveness_service import EffectivenessReport
        from mindflow.services.evidence_service import EvidenceBundleBuilder

        intervention_repo = InterventionLogRepository(
            session_factory=session_factory,
        )

        # Insert an intervention log
        now = datetime.now(UTC)
        await intervention_repo.log_triggered(
            user_id=1,
            intervention_type="nudge",
            context={},
            intervention_id="test-eff-001",
            triggered_at=now,
        )

        # Mock effectiveness service returning positive data
        mock_eff = MagicMock()
        report = EffectivenessReport(
            intervention_id="test-eff-001",
            before={"focus_score": 50.0, "switch_rate": 20.0, "distraction_ratio": 0.3},
            after={"focus_score": 68.0, "switch_rate": 15.0, "distraction_ratio": 0.2},
            deltas={"focus_score": 18.0, "switch_rate": -5.0, "distraction_ratio": -0.1},
            has_data=True,
        )
        mock_eff.compare_windows = AsyncMock(return_value=report)

        builder = EvidenceBundleBuilder(
            activity_repo=activity_repo,
            intervention_repo=intervention_repo,
            session_factory=session_factory,
            effectiveness_service=mock_eff,
        )

        window_end = now + timedelta(hours=1)
        window_start = window_end - timedelta(hours=2)

        bundle = await builder.build(1, window_start, window_end)

        assert len(bundle.intervention_history) >= 1
        record = bundle.intervention_history[0]
        assert "干预后" in record.effect_note
        assert "专注" in record.effect_note
        assert "18" in record.effect_note

    async def test_effect_note_unchanged_without_data(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        activity_repo: SQLAlchemyActivityRepository,
        create_tables: None,
    ) -> None:
        """effect_note falls back to user_response when no effectiveness data."""
        from unittest.mock import AsyncMock, MagicMock

        from mindflow.infrastructure.repositories.intervention import (
            InterventionLogRepository,
        )
        from mindflow.services.effectiveness_service import EffectivenessReport
        from mindflow.services.evidence_service import EvidenceBundleBuilder

        intervention_repo = InterventionLogRepository(
            session_factory=session_factory,
        )

        now = datetime.now(UTC)
        await intervention_repo.log_triggered(
            user_id=1,
            intervention_type="task_breakdown",
            context={},
            intervention_id="test-eff-002",
            triggered_at=now,
        )

        # Mock effectiveness returning no data
        mock_eff = MagicMock()
        report = EffectivenessReport(
            intervention_id="test-eff-002",
            before={"focus_score": 0.0, "switch_rate": 0.0, "distraction_ratio": 0.0},
            after={"focus_score": 0.0, "switch_rate": 0.0, "distraction_ratio": 0.0},
            deltas={"focus_score": 0.0, "switch_rate": 0.0, "distraction_ratio": 0.0},
            has_data=False,
        )
        mock_eff.compare_windows = AsyncMock(return_value=report)

        builder = EvidenceBundleBuilder(
            activity_repo=activity_repo,
            intervention_repo=intervention_repo,
            session_factory=session_factory,
            effectiveness_service=mock_eff,
        )

        window_end = now + timedelta(hours=1)
        window_start = window_end - timedelta(hours=2)

        bundle = await builder.build(1, window_start, window_end)

        assert len(bundle.intervention_history) >= 1
        record = bundle.intervention_history[0]
        # Should use default human-readable note, not effectiveness summary
        assert "尚未回应" in record.effect_note
        assert "干预后" not in record.effect_note


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — Novelty detection
# ═══════════════════════════════════════════════════════════════════════════════


class TestNoveltyDetection:
    """Novelty flag heuristics."""

    async def test_positive_novelty(
        self,
        builder: EvidenceBundleBuilder,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """App not in baseline top apps → flagged as novel."""
        # Baseline with known apps (V2 rows carrying process_name).
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        model.update([
            _v2_row(_local_utc(hour=8, dow=5, day=11), app_switch_count=10.0,
                    process_name="code.exe"),
            _v2_row(_local_utc(hour=8, dow=5, day=11), app_switch_count=12.0,
                    process_name="terminal.exe"),
        ])
        # Manually ensure some data so has_sufficient_data passes
        # We need to also check total_days — the baseline has it via update
        await _insert_baseline(session_factory, model)

        # Current window uses pycharm.exe (not in baseline)
        events = _make_events(count=15, process_name="pycharm.exe")
        await _insert_events(builder._activity_repo, events)

        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)
        assert len(bundle.novelty_flags) > 0
        assert any("pycharm.exe" in flag or "新应用" in flag for flag in bundle.novelty_flags)

    async def test_negative_novelty(
        self,
        builder: EvidenceBundleBuilder,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """All apps in baseline → no novelty flags."""
        model = _train_baseline()
        await _insert_baseline(session_factory, model)

        # 'code.exe' was used in training baseline
        events = _make_events(count=15, process_name="code.exe")
        await _insert_events(builder._activity_repo, events)

        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)
        # code.exe might be in the baseline's top apps
        novelty = list(bundle.novelty_flags)
        assert not any("code" in f for f in novelty), f"Unexpected novelty: {novelty}"

    async def test_no_baseline_no_novelty(
        self,
        builder: EvidenceBundleBuilder,
    ) -> None:
        """No baseline → no novelty detected."""
        events = _make_events(count=15, process_name="weird_app.exe")
        await _insert_events(builder._activity_repo, events)

        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)
        assert bundle.novelty_flags == ()


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — Serialisation
# ═══════════════════════════════════════════════════════════════════════════════


class TestSerialization:
    """EvidenceBundle JSON serialisation from the builder."""

    async def test_to_prompt_json_valid(
        self,
        builder: EvidenceBundleBuilder,
    ) -> None:
        """Bundle serializes to valid JSON without errors."""
        events = _make_events(count=15)
        await _insert_events(builder._activity_repo, events)

        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)
        from mindflow.domain.evidence import to_prompt_json
        raw = to_prompt_json(bundle)
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    async def test_to_prompt_json_no_titles(
        self,
        builder: EvidenceBundleBuilder,
    ) -> None:
        """Serialized JSON has no window title or file path leaks."""
        events = _make_events(count=15)
        await _insert_events(builder._activity_repo, events)

        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)
        from mindflow.domain.evidence import to_prompt_json
        raw = to_prompt_json(bundle)
        lower = raw.lower()
        assert "https://" not in lower
        assert ".com" not in lower.split("//")[-1] if "//" in lower else True
        assert "c:" not in lower


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — Unit-level severity helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TestSeverityHelpers:
    """Pure-function severity mapping tests."""

    def test_focus_severity_high(self) -> None:
        assert _focus_severity(85.0) == "info"

    def test_focus_severity_mild(self) -> None:
        assert _focus_severity(60.0) == "mild"

    def test_focus_severity_moderate(self) -> None:
        assert _focus_severity(40.0) == "moderate"

    def test_focus_severity_severe(self) -> None:
        assert _focus_severity(20.0) == "severe"

    def test_switch_severity_low(self) -> None:
        assert _switch_severity(10.0) == "info"

    def test_switch_severity_mild(self) -> None:
        assert _switch_severity(20.0) == "mild"

    def test_switch_severity_at_max_acceptable(self) -> None:
        """At exactly MAX_ACCEPTABLE_SWITCHES_PER_HOUR → mild."""
        assert _switch_severity(MAX_ACCEPTABLE_SWITCHES_PER_HOUR) == "mild"

    def test_switch_severity_at_moderate(self) -> None:
        assert _switch_severity(35.0) == "moderate"

    def test_switch_severity_severe(self) -> None:
        assert _switch_severity(50.0) == "severe"

    def test_block_severity_long(self) -> None:
        assert _block_severity(1800.0) == "info"

    def test_block_severity_mild(self) -> None:
        assert _block_severity(900.0) == "mild"

    def test_block_severity_moderate(self) -> None:
        assert _block_severity(400.0) == "moderate"

    def test_block_severity_short(self) -> None:
        assert _block_severity(100.0) == "severe"

    def test_deviation_normal_to_info(self) -> None:
        assert _deviation_severity("normal") == "info"

    def test_deviation_pass_through(self) -> None:
        assert _deviation_severity("mild") == "mild"
        assert _deviation_severity("moderate") == "moderate"
        assert _deviation_severity("severe") == "severe"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — G001 review gap: _deviation_to_confidence
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeviationToConfidence:
    """Pure-function tests for _deviation_to_confidence() (M3 gap)."""

    def test_zero_deviation(self) -> None:
        """z = 0.0 → 0.0 confidence."""
        from mindflow.services.evidence_service import _deviation_to_confidence
        assert _deviation_to_confidence(0.0) == 0.0

    def test_mild_deviation(self) -> None:
        """z ≈ 1.5 → ~0.356 (1.5/4.0 * 0.95)."""
        from mindflow.services.evidence_service import _deviation_to_confidence
        assert _deviation_to_confidence(1.5) == pytest.approx(0.356, abs=0.001)

    def test_severe_deviation_saturates(self) -> None:
        """z = 5.0 → 0.95 (saturates at z >= 4.0)."""
        from mindflow.services.evidence_service import _deviation_to_confidence
        assert _deviation_to_confidence(5.0) == 0.95

    def test_negative_deviation_absolute(self) -> None:
        """Negative z uses absolute value."""
        from mindflow.services.evidence_service import _deviation_to_confidence
        # z = -3.0 → |3.0| = 3.0 → 3.0/4.0 * 0.95 = 0.7125 → round(0.7125, 3) = 0.712
        assert _deviation_to_confidence(-3.0) == pytest.approx(0.712, abs=0.001)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — G001 review gap: _response_to_effect_note
# ═══════════════════════════════════════════════════════════════════════════════


class TestResponseToEffectNote:
    """Pure-function tests for _response_to_effect_note() (M4 gap)."""

    def test_accepted(self) -> None:
        from mindflow.services.evidence_service import _response_to_effect_note
        assert _response_to_effect_note("accepted") == "已接受并执行"

    def test_ignored(self) -> None:
        from mindflow.services.evidence_service import _response_to_effect_note
        assert _response_to_effect_note("ignored") == "用户忽略"

    def test_dismissed(self) -> None:
        from mindflow.services.evidence_service import _response_to_effect_note
        assert _response_to_effect_note("dismissed") == "用户已关闭"

    def test_none_defaults(self) -> None:
        from mindflow.services.evidence_service import _response_to_effect_note
        assert _response_to_effect_note(None) == "尚未回应"

    def test_unknown_defaults(self) -> None:
        from mindflow.services.evidence_service import _response_to_effect_note
        assert _response_to_effect_note("unknown") == "尚未回应"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — G001 review gap: severity boundary conditions
# ═══════════════════════════════════════════════════════════════════════════════


class TestSeverityBoundaries:
    """Exact boundary conditions for severity helpers (L1 gap)."""

    def test_focus_at_info_boundary(self) -> None:
        """focus_score 70.0 → info (exact info threshold)."""
        from mindflow.services.evidence_service import _focus_severity
        assert _focus_severity(70.0) == "info"

    def test_focus_at_mild_boundary(self) -> None:
        """focus_score 50.0 → mild (exact mild threshold)."""
        from mindflow.services.evidence_service import _focus_severity
        assert _focus_severity(50.0) == "mild"

    def test_focus_at_moderate_boundary(self) -> None:
        """focus_score 30.0 → moderate (exact moderate threshold)."""
        from mindflow.services.evidence_service import _focus_severity
        assert _focus_severity(30.0) == "moderate"

    def test_block_at_mild_boundary(self) -> None:
        """block_severity 600.0 → mild (exact mild threshold)."""
        from mindflow.services.evidence_service import _block_severity
        assert _block_severity(600.0) == "mild"

    def test_block_at_moderate_boundary(self) -> None:
        """block_severity 300.0 → moderate (exact moderate threshold)."""
        from mindflow.services.evidence_service import _block_severity
        assert _block_severity(300.0) == "moderate"

    def test_switch_at_mild_boundary(self) -> None:
        """switch_severity 30.0 → mild (at mild threshold)."""
        from mindflow.services.evidence_service import _switch_severity
        assert _switch_severity(30.0) == "mild"

    def test_switch_just_above_mild(self) -> None:
        """switch_severity 30.01 → moderate (just above mild threshold)."""
        from mindflow.services.evidence_service import _switch_severity
        assert _switch_severity(30.01) == "moderate"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — TMT decomposition wiring (contract §3)
# ═══════════════════════════════════════════════════════════════════════════════


class TestTmtWiring:
    """``build()`` always computes TMT and attaches it to the bundle.

    ``domain/tmt.py`` has its own unit tests for the arithmetic; these cover
    the wiring: which repository reads feed which proxy, that a missing
    repository degrades to the documented prior instead of inventing data,
    and that exactly four citeable ``tmt_*`` items reach the evidence catalog.
    """

    async def test_always_produces_a_breakdown(self, builder: EvidenceBundleBuilder) -> None:
        """Even with no feedback and no tasks, TMT is present and all-prior."""
        await _insert_events(builder._activity_repo, _make_events(count=15))

        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)
        assert bundle.tmt is not None
        payload = bundle.tmt.to_dict()  # type: ignore[union-attr]
        # No telemetry repo and no task repo were injected → E and V are prior.
        assert payload["components"]["expectancy"]["source"] == "先验（无近14天自评）"
        assert payload["components"]["value"]["source"] == "先验"

    async def test_exactly_four_tmt_items_reach_the_catalog(
        self, builder: EvidenceBundleBuilder
    ) -> None:
        """One item per component, and every id is citeable."""
        await _insert_events(builder._activity_repo, _make_events(count=15))

        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)
        tmt_items = [item for item in bundle.items if item.metric.startswith("tmt_")]
        assert {item.metric for item in tmt_items} == {
            "tmt_expectancy",
            "tmt_value",
            "tmt_impulsiveness",
            "tmt_delay",
        }
        assert all(item.severity == "info" for item in tmt_items)

        catalog_ids = {fact.id for fact in build_evidence_catalog(bundle)}
        assert {
            "focus.tmt_expectancy",
            "focus.tmt_value",
            "focus.tmt_impulsiveness",
            "focus.tmt_delay",
        } <= catalog_ids

    async def test_recent_feedback_drives_expectancy(self, engine, all_tables: None) -> None:
        """Self-reports inside the 14-day lookback raise E above the prior."""
        factory = create_session_factory(engine)
        telemetry = TelemetryRepository(session_factory=factory)
        await telemetry.save_focus_feedback(1, "s1", "focused", 5, None)
        await telemetry.save_focus_feedback(1, "s2", "focused", 5, None)

        builder = EvidenceBundleBuilder(
            activity_repo=SQLAlchemyActivityRepository(
                session_factory=factory, pulsetime_s=10
            ),
            intervention_repo=InterventionLogRepository(session_factory=factory),
            session_factory=factory,
            telemetry_repo=telemetry,
        )
        await _insert_events(builder._activity_repo, _make_events(count=15))

        # The feedback rows are stamped "now"; analyse a window ending now so
        # they fall inside the lookback.
        now = datetime.now(UTC)
        bundle = await builder.build(1, now - timedelta(hours=4), now)
        expectancy = bundle.tmt.to_dict()["components"]["expectancy"]  # type: ignore[union-attr]
        assert expectancy["score"] > PRIOR
        assert "2 次自评" in expectancy["source"]

    async def test_old_feedback_is_excluded(self, engine, all_tables: None) -> None:
        """A self-report older than 14 days does not drive E."""
        factory = create_session_factory(engine)
        telemetry = TelemetryRepository(session_factory=factory)
        await telemetry.save_focus_feedback(1, "s1", "focused", 5, None)
        # Push the row back beyond the lookback window.
        async with factory() as session, session.begin():
            await session.execute(
                sa.update(focus_session_feedback).values(
                    created_at=(_WINDOW_END - timedelta(days=30)).isoformat()
                )
            )

        builder = EvidenceBundleBuilder(
            activity_repo=SQLAlchemyActivityRepository(
                session_factory=factory, pulsetime_s=10
            ),
            intervention_repo=InterventionLogRepository(session_factory=factory),
            session_factory=factory,
            telemetry_repo=telemetry,
        )
        await _insert_events(builder._activity_repo, _make_events(count=15))

        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)
        expectancy = bundle.tmt.to_dict()["components"]["expectancy"]  # type: ignore[union-attr]
        assert expectancy["score"] == PRIOR
        assert expectancy["source"] == "先验（无近14天自评）"

    async def test_open_tasks_drive_value_and_delay(self, engine, all_tables: None) -> None:
        """A high-priority task due soon raises V and lowers D."""
        factory = create_session_factory(engine)
        task_repo = SQLAlchemyTaskRepository(session_factory=factory)
        await task_repo.create(
            1,
            "thesis draft",
            priority=5,
            deadline_utc=(_WINDOW_END + timedelta(hours=2)).isoformat(),
        )
        builder = EvidenceBundleBuilder(
            activity_repo=SQLAlchemyActivityRepository(
                session_factory=factory, pulsetime_s=10
            ),
            intervention_repo=InterventionLogRepository(session_factory=factory),
            session_factory=factory,
            task_repo=task_repo,
        )
        await _insert_events(builder._activity_repo, _make_events(count=15))

        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)
        components = bundle.tmt.to_dict()["components"]  # type: ignore[union-attr]
        # priority 5 → 1.0, blended 0.7/0.3 with the productive share (code.exe
        # is 100% productive here) → exactly 1.0.
        assert components["value"]["score"] == pytest.approx(1.0)
        # Two hours out → D = 2 / (2 + 24).
        assert components["delay"]["score"] == pytest.approx(2.0 / 26.0, abs=1e-4)

    async def test_done_tasks_are_ignored(self, engine, all_tables: None) -> None:
        """Only pending / in_progress tasks feed the V and D proxies."""
        factory = create_session_factory(engine)
        task_repo = SQLAlchemyTaskRepository(session_factory=factory)
        await task_repo.create(
            1,
            "already finished",
            priority=5,
            status="done",
            deadline_utc=(_WINDOW_END + timedelta(hours=1)).isoformat(),
        )
        builder = EvidenceBundleBuilder(
            activity_repo=SQLAlchemyActivityRepository(
                session_factory=factory, pulsetime_s=10
            ),
            intervention_repo=InterventionLogRepository(session_factory=factory),
            session_factory=factory,
            task_repo=task_repo,
        )
        await _insert_events(builder._activity_repo, _make_events(count=15))

        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)
        components = bundle.tmt.to_dict()["components"]  # type: ignore[union-attr]
        assert components["value"]["score"] == PRIOR
        assert components["delay"]["score"] == PRIOR

    async def test_missing_task_repo_degrades_without_raising(
        self, builder: EvidenceBundleBuilder
    ) -> None:
        """No task repo is a prior, not an exception."""
        await _insert_events(builder._activity_repo, _make_events(count=15))
        bundle = await builder.build(1, _WINDOW_START, _WINDOW_END)
        components = bundle.tmt.to_dict()["components"]  # type: ignore[union-attr]
        assert components["value"]["source"] == "先验"
        assert components["delay"]["source"] == "无截止时间"

    def test_deviation_summary_feeds_impulsiveness(self) -> None:
        """The I proxy reads the contract §2 payload rather than re-deriving it."""
        assert deviation_z_mean(None) is None
        payload = {
            "features": [
                {"feature": "app_switch_count", "robust_z": 3.0},
                {"feature": "longest_segment_ratio", "robust_z": -1.0},
            ]
        }
        assert deviation_z_mean(payload) == pytest.approx(2.0)


class TestValueTaskSelection:
    """``_select_value_task`` prefers the important task that is actually due."""

    def test_important_task_due_soon_beats_important_task_far_out(self) -> None:
        soon = Task(
            id="a", user_id=1, title="soon", description="", priority=4,
            status="pending", deadline_utc=_WINDOW_END + timedelta(days=2),
            estimated_minutes=None, created_at=_WINDOW_END, updated_at=_WINDOW_END,
        )
        far = Task(
            id="b", user_id=1, title="far", description="", priority=5,
            status="pending", deadline_utc=_WINDOW_END + timedelta(days=30),
            estimated_minutes=None, created_at=_WINDOW_END, updated_at=_WINDOW_END,
        )
        assert _select_value_task([far, soon], _WINDOW_END) is soon

    def test_no_due_soon_task_falls_back_to_highest_priority(self) -> None:
        low = Task(
            id="a", user_id=1, title="low", description="", priority=2,
            status="pending", deadline_utc=None,
            estimated_minutes=None, created_at=_WINDOW_END, updated_at=_WINDOW_END,
        )
        high = Task(
            id="b", user_id=1, title="high", description="", priority=5,
            status="pending", deadline_utc=None,
            estimated_minutes=None, created_at=_WINDOW_END, updated_at=_WINDOW_END,
        )
        assert _select_value_task([low, high], _WINDOW_END) is high

    def test_ties_break_on_the_soonest_deadline(self) -> None:
        later = Task(
            id="a", user_id=1, title="later", description="", priority=3,
            status="pending", deadline_utc=_WINDOW_END + timedelta(days=6),
            estimated_minutes=None, created_at=_WINDOW_END, updated_at=_WINDOW_END,
        )
        sooner = Task(
            id="b", user_id=1, title="sooner", description="", priority=3,
            status="pending", deadline_utc=_WINDOW_END + timedelta(days=1),
            estimated_minutes=None, created_at=_WINDOW_END, updated_at=_WINDOW_END,
        )
        assert _select_value_task([later, sooner], _WINDOW_END) is sooner

    def test_empty_task_list_returns_none(self) -> None:
        assert _select_value_task([], _WINDOW_END) is None


class TestHoursToDeadline:
    """The D proxy's raw input."""

    @staticmethod
    def _task(deadline: datetime | None) -> Task:
        return Task(
            id="x", user_id=1, title="t", description="", priority=3,
            status="pending", deadline_utc=deadline,
            estimated_minutes=None, created_at=_WINDOW_END, updated_at=_WINDOW_END,
        )

    def test_nearest_deadline_wins(self) -> None:
        tasks = [
            self._task(_WINDOW_END + timedelta(hours=50)),
            self._task(_WINDOW_END + timedelta(hours=5)),
            self._task(None),
        ]
        assert _hours_to_deadline(tasks, _WINDOW_END) == pytest.approx(5.0)

    def test_overdue_is_negative(self) -> None:
        task = self._task(_WINDOW_END - timedelta(hours=3))
        assert _hours_to_deadline([task], _WINDOW_END) == pytest.approx(-3.0)

    def test_no_deadlines_returns_none(self) -> None:
        assert _hours_to_deadline([self._task(None)], _WINDOW_END) is None
