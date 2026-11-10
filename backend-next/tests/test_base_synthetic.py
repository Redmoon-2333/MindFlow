"""Contract and leakage tests for the event-derived synthetic base corpus."""

from __future__ import annotations

import copy
import random
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from mindflow.domain.events import ActivityEvent
from mindflow.domain.feature_schema import V2_FEATURE_NAMES
from mindflow.domain.features import count_confirmed_switches
from mindflow.domain.task_context import TASK_TYPE_CODES
from mindflow.services import telemetry_features
from mindflow.train import base_synthetic
from mindflow.train.base_catalog import APP_CATALOG, ROLES, SCENARIOS
from mindflow.train.base_synthetic import (
    SyntheticCorpus,
    audit_base_corpus,
    generate_base_corpus,
)


@pytest.fixture(scope="module")
def corpus() -> SyntheticCorpus:
    return generate_base_corpus(seed=134, personas_per_role=3, days=7, windows_per_day=24)


def test_catalog_is_broad_and_internally_consistent() -> None:
    assert len(ROLES) >= 16
    assert len(APP_CATALOG) - 1 >= 70  # The unknown tool is not a real app.
    assert len(SCENARIOS) >= 20
    assert len({role.key for role in ROLES}) == len(ROLES)
    assert len({app.name for app in APP_CATALOG.values()}) == len(APP_CATALOG)
    for role in ROLES:
        assert set(role.apps) <= APP_CATALOG.keys()
        assert role.specialty in {scenario.key for scenario in SCENARIOS}
        assert any(not APP_CATALOG[key].domain for key in role.apps)
    for scenario in SCENARIOS:
        assert set(scenario.apps) <= APP_CATALOG.keys()
    assert set(TASK_TYPE_CODES.values()) == {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}


def test_deterministic_without_mutating_global_random_state() -> None:
    state = random.getstate()
    kwargs = {"seed": 42, "personas_per_role": 3, "days": 1, "windows_per_day": 3}
    first = generate_base_corpus(**kwargs)
    assert random.getstate() == state
    assert first == generate_base_corpus(**kwargs)
    assert first != generate_base_corpus(**{**kwargs, "seed": 43})
    assert audit_base_corpus(first)["passed"]


def test_small_corpus_contract_and_audit(corpus: SyntheticCorpus) -> None:
    result = audit_base_corpus(corpus)
    assert result["passed"], result["checks"]
    assert all(type(value) is bool for value in result["checks"].values())
    assert len(corpus.windows) == len(ROLES) * 3 * 7 * 24
    assert corpus.metadata["synthetic_only"] is True
    assert corpus.metadata["contains_real_user_data"] is False
    assert corpus.metadata["coverage"] == result["coverage"]
    assert result["coverage"]["label_counts"].keys() == {"0", "1", "unlabelled"}
    for row in corpus.windows:
        assert set(row["features"]) == set(V2_FEATURE_NAMES)
        assert row["feature_schema_version"] == 4
        assert row["features"]["task_type_code"] in TASK_TYPE_CODES.values()
        assert row["label_source"] == "synthetic"
        assert row["features"]["input_active_ratio"] <= (
            row["features"]["active_seconds_ratio"] + 1e-5
        )
        start, end = map(datetime.fromisoformat, (row["window_start_utc"], row["window_end_utc"]))
        assert start.tzinfo == UTC and end - start == timedelta(minutes=5)


def test_every_window_calls_real_rollup_with_simulated_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = telemetry_features.build_v2_feature_window

    def recording_builder(
        events: list[ActivityEvent],
        buckets: list[dict[str, Any]],
        segments: list[dict[str, Any]],
        start: datetime,
        end: datetime,
        **kwargs: Any,
    ) -> dict[str, int | float]:
        nonlocal calls
        calls += 1
        assert events and all(isinstance(event, ActivityEvent) for event in events)
        assert sum(event.duration_s for event in events) == 300
        assert all(
            a.timestamp_utc + timedelta(seconds=a.duration_s) == b.timestamp_utc
            for a, b in zip(events, events[1:], strict=False)
        )
        assert len(buckets) in {0, 10}
        for bucket in buckets:
            assert bucket["duration_s"] == 30
            stamp = datetime.fromisoformat(bucket["window_start_utc"])
            assert start <= stamp < end and stamp.second in {0, 30}
        assert sum(segment["duration_s"] for segment in segments) <= 300
        features = original(events, buckets, segments, start, end, **kwargs)
        assert features["app_switch_count"] == count_confirmed_switches(events)
        return features

    monkeypatch.setattr(base_synthetic, "build_v2_feature_window", recording_builder)
    generated = generate_base_corpus(personas_per_role=3, days=1, windows_per_day=2)
    assert calls == len(generated.windows)


def test_persona_calendar_and_session_splits(corpus: SyntheticCorpus) -> None:
    cutoff = datetime.fromisoformat(corpus.metadata["train_calendar_cutoff_utc"])
    persona_splits: dict[str, set[str]] = defaultdict(set)
    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    calibration_future = 0
    for row in corpus.windows:
        persona_splits[row["persona_id"]].add(row["split"])
        sessions[row["session_id"]].append(row)
        start = datetime.fromisoformat(row["window_start_utc"])
        end = datetime.fromisoformat(row["window_end_utc"])
        if row["split"] == "train":
            assert end <= cutoff
        elif row["split"] == "test_future":
            assert start >= cutoff
        elif row["split"] == "calibration" and start >= cutoff:
            calibration_future += 1
    for role in ROLES:
        assert persona_splits[f"{role.key}:p02"] == {"test_persona"}
        assert persona_splits[f"{role.key}:p01"] == {"calibration"}
        assert persona_splits[f"{role.key}:p00"] == {"train", "test_future"}
    assert calibration_future > 0  # Consumers must apply the advertised cutoff.
    for group in sessions.values():
        assert 2 <= len(group) <= 12
        assert len({row["split"] for row in group}) == 1
        assert len({row["persona_id"] for row in group}) == 1
        assert len({row["scenario"] for row in group}) == 1


def test_goal_labels_allow_low_input_high_switch_and_shared_apps(corpus: SyntheticCorpus) -> None:
    rows = corpus.windows
    assert any(
        row["label"] == 1
        and row["features"]["input_active_ratio"] < 0.03
        and row["scenario"] in {"lecture", "reference_reading", "audio_review", "video_review"}
        for row in rows
    )
    assert any(
        row["label"] == 1
        and row["features"]["app_switch_count"] >= 4
        and row["scenario"] == "collaboration"
        for row in rows
    )
    assert any(row["label"] == 1 and row["local_hour"] < 5 for row in rows)
    assert any(row["label"] == 1 and row["scenario"] == "game_testing" for row in rows)
    assert any(row["label"] == 0 and row["scenario"] == "goal_drift" for row in rows)
    assert len(corpus.metadata["coverage"]["dual_label_apps"]) >= 10
    assert len(corpus.metadata["coverage"]["timezones"]) >= 4
    assert len(corpus.metadata["coverage"]["local_hours"]) >= 16
    for row in rows:
        if row["scenario"] in {
            "away",
            "planned_break",
            "mixed_intent",
            "unknown_app",
            "input_missing",
            "meeting_unclear",
        }:
            assert row["label"] is None
        if row["scenario"] == "browser_closed":
            assert row["features"]["browser_ratio"] == 0
        if row["scenario"] == "unknown_app":
            assert row["features"]["task_type_code"] == TASK_TYPE_CODES["unknown"]


@pytest.mark.parametrize(
    ("field", "value", "check"),
    [
        ("keypress_rate_per_min", float("nan"), "finite_values"),
        ("mouse_distance_per_min", -1.0, "nonnegative_values"),
        ("input_active_ratio", 1.2, "unit_ranges"),
        ("hour_sin", -0.5, "unit_ranges"),
        ("task_type_code", 0.0, "task_codes"),
        ("app_switch_count", 99, "switch_ranges"),
        ("app_switch_count", 0.5, "confirmed_switches"),
    ],
)
def test_audit_rejects_invalid_features(
    corpus: SyntheticCorpus,
    field: str,
    value: float,
    check: str,
) -> None:
    row = copy.deepcopy(corpus.windows[0])
    row["features"][field] = value
    report = audit_base_corpus(SyntheticCorpus([row], corpus.metadata))
    assert report["checks"][check] is False
    assert report["passed"] is False


def test_audit_detects_holdout_and_session_leakage(corpus: SyntheticCorpus) -> None:
    # Same session copied into another split and persona.
    first = copy.deepcopy(next(row for row in corpus.windows if row["split"] == "test_persona"))
    second = copy.deepcopy(first)
    second["id"] += ":copy"
    second["split"] = "train"
    second["persona_id"] = first["role"] + ":p00"
    report = audit_base_corpus(SyntheticCorpus([first, second], corpus.metadata))
    assert report["checks"]["session_split_isolation"] is False
    first["split"] = "train"
    report = audit_base_corpus(SyntheticCorpus([first], corpus.metadata))
    assert report["checks"]["persona_split_isolation"] is False


def test_audit_detects_rest_labels_and_bad_timestamps(corpus: SyntheticCorpus) -> None:
    row = copy.deepcopy(next(row for row in corpus.windows if row["scenario"] == "planned_break"))
    row["label"] = 0
    row["window_start_utc"] = "broken"
    report = audit_base_corpus(SyntheticCorpus([row], corpus.metadata))
    assert report["checks"]["rest_missing_unlabelled"] is False
    assert report["checks"]["utc_five_minute_windows"] is False
    assert report["checks"]["scenario_labels"] is False


def test_audit_detects_duplicate_ids_and_future_train(corpus: SyntheticCorpus) -> None:
    row = copy.deepcopy(next(row for row in corpus.windows if row["split"] == "test_future"))
    row["split"] = "train"
    report = audit_base_corpus(SyntheticCorpus([row, copy.deepcopy(row)], corpus.metadata))
    assert report["checks"]["calendar_split"] is False
    assert report["checks"]["unique_ids"] is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"personas_per_role": 2},
        {"days": 0},
        {"windows_per_day": 1},
        {"windows_per_day": 289},
        {"seed": True},
        {"days": 1.5},
    ],
)
def test_invalid_recipe_arguments(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        generate_base_corpus(**kwargs)
