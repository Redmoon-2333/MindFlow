"""Regression tests for the phase-3.2 task-context feature upgrade (schema v4).

Covers the three layers of the feature end to end:

* ``domain/task_context.py`` — vocabulary, user-rule mapping, per-window
  summarisation (dominant / entropy / dominant ratio / unknown ratio).
* ``services/telemetry_features.py`` — the five v4 columns and the transition
  contract, including the "missing previous window is not a transition" choice.
* ``train/v2.py`` + ``infrastructure/repositories`` — the columns survive into
  the training matrix, and a non-current ``feature_schema_version`` is rejected
  rather than silently read.

Every numeric expectation below is computed by hand in the test (or from the
documented contract, e.g. ``log(2) / log(7)``) instead of being copied from the
implementation, so a future semantic change has to argue with the test.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from mindflow.domain.events import make_event
from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION, V2_FEATURE_NAMES
from mindflow.domain.task_context import (
    TASK_CONTEXT_CATEGORIES,
    TASK_CONTEXT_UNKNOWN,
    TASK_TYPE_CODES,
    TaskContextMapper,
    summarize_task_context,
)
from mindflow.infrastructure.repositories.activity import SQLAlchemyActivityRepository
from mindflow.infrastructure.repositories.baseline import BaselineRepository
from mindflow.infrastructure.repositories.preferences import PreferencesRepository
from mindflow.infrastructure.repositories.task_context import TaskContextRulesRepository
from mindflow.infrastructure.repositories.telemetry import TelemetryRepository
from mindflow.services.telemetry_features import (
    build_v2_feature_window,
    task_context_from_feature_window,
)
from mindflow.services.telemetry_service import TelemetryService
from mindflow.train.v2 import prepare_v2_training_data

START = datetime(2026, 7, 24, 8, 0, tzinfo=UTC)
END = START + timedelta(minutes=5)
WINDOW_SECONDS = (END - START).total_seconds()

TASK_COLUMNS = (
    "task_type_code",
    "task_type_entropy",
    "task_type_dominant_ratio",
    "task_context_transition",
    "task_unknown_ratio",
)


def _event(process: str, *, offset_s: float, duration_s: float) -> Any:
    """One foreground window snapshot inside the (START, END) fixture window."""
    return make_event(
        user_id=1,
        timestamp_utc=START + timedelta(seconds=offset_s),
        duration_s=duration_s,
        process_name=process,
        is_idle=False,
    )


def _window(**builder_kwargs: Any) -> dict[str, Any]:
    """Build the fixture window from two 150s events, one process each.

    ``first_process`` / ``second_process`` select the process names; everything
    else is forwarded to :func:`build_v2_feature_window`.
    """
    first = str(builder_kwargs.pop("first_process", "first.exe"))
    second = str(builder_kwargs.pop("second_process", "second.exe"))
    events = [
        _event(first, offset_s=0, duration_s=150),
        _event(second, offset_s=150, duration_s=150),
    ]
    return build_v2_feature_window(events, [], [], START, END, **builder_kwargs)


def _training_row(features: dict[str, Any], start: datetime = START) -> dict[str, Any]:
    """Wrap a built feature window into the row shape the trainer consumes."""
    return {
        "window_start_utc": start.isoformat(),
        "window_end_utc": (start + timedelta(minutes=5)).isoformat(),
        "feature_schema_version": features["feature_schema_version"],
        "features": features,
    }


def _feedback(start: datetime, label: str = "focus", score: int = 5) -> dict[str, Any]:
    return {
        "session_id": "session-1",
        "start_time": start.isoformat(),
        "end_time": (start + timedelta(minutes=30)).isoformat(),
        "label": label,
        "score": score,
        "task_type": "coding",
    }


def _window_row(features: dict[str, Any], start: datetime = START) -> dict[str, Any]:
    """The repository row shape for a built feature window."""
    return {
        "user_id": 1,
        "window_start_utc": start,
        "window_end_utc": start + timedelta(minutes=5),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features_json": json.dumps(features, ensure_ascii=False),
        "label": None,
    }


# ── 1. task_type_code is a real observation, not the constant 0.0 ───────────


def test_task_type_code_varies_with_the_observed_task_mix() -> None:
    """Two windows over identical activity but different mappings disagree."""
    coding = _window(process_task_contexts={"first.exe": "coding", "second.exe": "coding"})
    writing = _window(process_task_contexts={"first.exe": "writing", "second.exe": "writing"})

    assert coding["task_type_code"] == TASK_TYPE_CODES["coding"] == 1.0
    assert writing["task_type_code"] == TASK_TYPE_CODES["writing"] == 2.0
    # The pre-v4 payload hard-coded 0.0; 0.0 stays reserved for "column absent".
    assert coding["task_type_code"] != 0.0
    assert writing["task_type_code"] != 0.0
    assert coding["task_type_code"] != writing["task_type_code"]


def test_unmapped_window_reports_unknown_code_and_full_unknown_ratio() -> None:
    """No mapping at all => unknown (7.0), never a fabricated category or 0.0.

    ``task_type_dominant_ratio`` is 1.0 here on purpose: it always describes
    the dominant bucket, and the dominant bucket *is* unknown.  ``unknown`` is
    therefore identifiable by ``task_unknown_ratio == 1.0`` together with the
    high code — never by a zero dominance ratio.
    """
    features = _window(first_process="dark.exe", second_process="mystery.exe")

    assert features["task_type_code"] == TASK_TYPE_CODES["unknown"] == 7.0
    assert features["task_unknown_ratio"] == 1.0
    assert features["task_type_entropy"] == 0.0
    assert features["task_type_dominant_ratio"] == 1.0


def test_partially_mapped_window_splits_unknown_coverage_by_observed_seconds() -> None:
    """Only the unmapped process's seconds count as unknown."""
    features = _window(process_task_contexts={"first.exe": "coding"})

    assert features["task_type_code"] == TASK_TYPE_CODES["coding"]
    # Second event (150s of 300s) has no mapping entry.
    assert features["task_unknown_ratio"] == pytest.approx(0.5)
    assert features["task_type_dominant_ratio"] == pytest.approx(0.5)
    observed = features["task_unknown_ratio"] + features["task_type_dominant_ratio"]
    assert observed == pytest.approx(1.0)


def test_unknown_code_is_not_read_back_as_a_real_context() -> None:
    """The reader round-trips real codes and refuses to invent anything else."""
    assert task_context_from_feature_window({"task_type_code": 1.0}) == "coding"
    assert task_context_from_feature_window({"task_type_code": 7.0}) == TASK_CONTEXT_UNKNOWN
    for absent_or_legacy in ({}, {"task_type_code": 0.0}, {"task_type_code": "nope"}):
        assert task_context_from_feature_window(absent_or_legacy) == TASK_CONTEXT_UNKNOWN


# ── 2. dominant / entropy / dominant ratio: hand-computed contracts ─────────


def test_single_category_window_has_zero_entropy_and_full_dominance() -> None:
    features = _window(process_task_contexts={"first.exe": "coding", "second.exe": "coding"})

    assert features["task_type_code"] == TASK_TYPE_CODES["coding"]
    assert features["task_type_entropy"] == 0.0
    assert features["task_type_dominant_ratio"] == 1.0
    assert features["task_unknown_ratio"] == 0.0


def test_even_two_way_mix_matches_the_summarize_task_context_contract() -> None:
    """150s/150s => entropy ``log(2)/log(7)``, dominance 0.5 — the documented
    normalisation (Shannon entropy divided by ``log(len(vocabulary))``)."""
    mixed = _window(
        process_task_contexts={"first.exe": "coding", "second.exe": "entertainment"}
    )
    expected_entropy = math.log(2) / math.log(len(TASK_CONTEXT_CATEGORIES))
    summary = summarize_task_context({"coding": 150.0, "entertainment": 150.0})

    assert mixed["task_type_code"] in {
        TASK_TYPE_CODES["coding"],
        TASK_TYPE_CODES["entertainment"],
    }
    assert mixed["task_type_entropy"] == pytest.approx(expected_entropy, abs=1e-6)
    assert mixed["task_type_dominant_ratio"] == pytest.approx(0.5)
    assert mixed["task_unknown_ratio"] == 0.0
    # Cross-check: the builder's numbers are exactly the summary's numbers, so
    # the hand-computed normalisation above is the contract, not a coincidence.
    assert summary.entropy == pytest.approx(expected_entropy, abs=1e-12)
    assert summary.dominant_ratio == pytest.approx(0.5)
    assert summary.total_seconds == pytest.approx(WINDOW_SECONDS)
    assert mixed["task_type_entropy"] == pytest.approx(summary.entropy, abs=1e-6)


def test_uneven_mix_entropy_and_dominant_ratio_from_the_summary() -> None:
    """225s coding / 75s entertainment: dominance 0.75 and entropy
    ``-(0.75*log 0.75 + 0.25*log 0.25) / log(7)``."""
    summary = summarize_task_context({"coding": 225.0, "entertainment": 75.0})
    expected_entropy = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25)) / math.log(
        len(TASK_CONTEXT_CATEGORIES)
    )

    assert summary.dominant == "coding"
    assert summary.dominant_ratio == pytest.approx(0.75)
    assert summary.unknown_ratio == 0.0
    assert summary.entropy == pytest.approx(expected_entropy, abs=1e-12)


def test_summarize_task_context_unknown_bucket_participates() -> None:
    """The unknown share lowers certainty (entropy) and is reported separately;
    a tie with a real category is broken in vocabulary order (unknown last)."""
    summary = summarize_task_context({"coding": 150.0, TASK_CONTEXT_UNKNOWN: 150.0})

    assert summary.dominant == "coding"
    assert summary.dominant_ratio == pytest.approx(0.5)
    assert summary.unknown_ratio == pytest.approx(0.5)
    assert summary.entropy == pytest.approx(math.log(2) / math.log(7), abs=1e-12)


def test_summarize_task_context_without_observations_is_unknown_not_safe() -> None:
    summary = summarize_task_context({})

    assert summary.dominant == TASK_CONTEXT_UNKNOWN
    assert summary.entropy == 0.0
    assert summary.dominant_ratio == 0.0
    assert summary.unknown_ratio == 1.0
    assert summary.total_seconds == 0.0


def test_equal_mix_of_all_seven_contexts_normalises_to_one() -> None:
    """The normaliser's upper bound: an even spread over the whole vocabulary."""
    summary = summarize_task_context({name: 10.0 for name in TASK_CONTEXT_CATEGORIES})

    assert summary.entropy == pytest.approx(1.0, abs=1e-12)
    assert summary.dominant_ratio == pytest.approx(1.0 / len(TASK_CONTEXT_CATEGORIES))


# ── 3. task_context_transition ─────────────────────────────────────────────


def test_transition_fires_only_when_the_dominant_context_changes() -> None:
    coding = {"first.exe": "coding", "second.exe": "coding"}
    writing = {"first.exe": "writing", "second.exe": "writing"}

    same = _window(process_task_contexts=coding, previous_task_context="coding")
    changed = _window(process_task_contexts=writing, previous_task_context="coding")
    first_ever = _window(process_task_contexts=coding, previous_task_context=None)
    no_argument = _window(process_task_contexts=coding)

    assert same["task_context_transition"] == 0.0
    assert changed["task_context_transition"] == 1.0
    # Documented choice: a missing previous window (range start / first window
    # ever) is a gap in measurement, not an observed switch.
    assert first_ever["task_context_transition"] == 0.0
    assert no_argument["task_context_transition"] == 0.0


def test_transition_ignores_unknown_on_either_side() -> None:
    """"Could not tell" is not a switch: unknown -> known and known -> unknown
    are both 0.0 (``task_unknown_ratio`` already reports the gap)."""
    into_known = _window(
        process_task_contexts={"first.exe": "coding", "second.exe": "coding"},
        previous_task_context="unknown",
    )
    out_of_known = _window(
        first_process="dark.exe",
        second_process="mystery.exe",
        previous_task_context="coding",
    )
    both_unknown = _window(first_process="dark.exe", second_process="mystery.exe")

    assert into_known["task_type_code"] == TASK_TYPE_CODES["coding"]
    assert into_known["task_context_transition"] == 0.0
    assert out_of_known["task_type_code"] == TASK_TYPE_CODES["unknown"]
    assert out_of_known["task_unknown_ratio"] == 1.0
    assert out_of_known["task_context_transition"] == 0.0
    assert both_unknown["task_context_transition"] == 0.0
    # A real switch between two known contexts is the only 1.0 case.
    assert _window(
        process_task_contexts={"first.exe": "meeting", "second.exe": "meeting"},
        previous_task_context="coding",
    )["task_context_transition"] == 1.0


def test_transition_chain_is_seeded_from_the_built_window() -> None:
    """The rollup's own chaining: read the context back, feed it forward."""
    first = _window(process_task_contexts={"first.exe": "coding", "second.exe": "coding"})
    carried = task_context_from_feature_window(first)
    second = _window(
        process_task_contexts={"first.exe": "meeting", "second.exe": "meeting"},
        previous_task_context=carried,
    )

    assert carried == "coding"
    assert second["task_context_transition"] == 1.0


# ── 4. same mix, different unknown coverage ────────────────────────────────


def test_same_category_mix_with_more_unknown_coverage_is_distinguishable() -> None:
    """Both windows are an even two-way split of what was *observed*; the one
    with more unmapped time still reads differently."""
    fully_mapped = _window(
        process_task_contexts={"first.exe": "coding", "second.exe": "entertainment"}
    )
    partly_mapped = _window(
        first_process="a.exe",
        second_process="b.exe",
        process_task_contexts={"a.exe": "coding"},
    )
    even_split_entropy = math.log(2) / math.log(len(TASK_CONTEXT_CATEGORIES))

    assert fully_mapped["task_unknown_ratio"] == 0.0
    assert partly_mapped["task_unknown_ratio"] == pytest.approx(0.5)
    assert fully_mapped["task_type_dominant_ratio"] == pytest.approx(0.5)
    assert partly_mapped["task_type_dominant_ratio"] == pytest.approx(0.5)
    # Two-way split either way, so entropy agrees; the unknown indicator is what
    # separates "half coding half fun" from "half coding, half unmeasured".
    assert fully_mapped["task_type_entropy"] == pytest.approx(even_split_entropy, abs=1e-6)
    assert partly_mapped["task_type_entropy"] == pytest.approx(even_split_entropy, abs=1e-6)
    assert fully_mapped["task_unknown_ratio"] != partly_mapped["task_unknown_ratio"]


def test_unknown_only_dominates_when_strictly_larger() -> None:
    """Tie-breaking rule: vocabulary order, so a real category wins a tie."""
    tie = _window(
        first_process="p1.exe",
        second_process="p2.exe",
        process_task_contexts={"p1.exe": "social"},
    )
    mostly_unknown = _window(
        first_process="p1.exe",
        second_process="p2.exe",
        process_task_contexts={"p1.exe": "social", "p2.exe": "social"},
    )

    assert tie["task_unknown_ratio"] == pytest.approx(0.5)
    assert tie["task_type_code"] == TASK_TYPE_CODES["social"]
    assert tie["task_type_dominant_ratio"] == pytest.approx(0.5)
    assert mostly_unknown["task_type_code"] == TASK_TYPE_CODES["social"]
    assert mostly_unknown["task_unknown_ratio"] == 0.0


# ── 5. mapper: user rules beat the built-in defaults ───────────────────────


def test_category_rule_overrides_the_builtin_default() -> None:
    mapper = TaskContextMapper([
        {"match_type": "category", "match_value": "code", "task_context": "entertainment"},
    ])

    assert mapper.has_rules() is True
    assert mapper.context_for_category("code") == "entertainment"  # default is "coding"
    assert mapper.context_for_category("document") == "writing"  # untouched default
    assert mapper.context_for_category("other") == TASK_CONTEXT_UNKNOWN
    assert mapper.resolve("code") == "entertainment"


def test_domain_rule_maps_a_browser_domain_and_beats_the_category_rule() -> None:
    mapper = TaskContextMapper([
        {"match_type": "domain", "match_value": "arxiv.org", "task_context": "writing"},
        {"match_type": "category", "match_value": "browser_work", "task_context": "social"},
    ])

    # The domain rule wins for the mapped host and its subdomains...
    assert mapper.context_for_domain("arxiv.org") == "writing"
    assert mapper.context_for_domain("docs.arxiv.org") == "writing"
    assert mapper.resolve("browser_work", "arxiv.org") == "writing"
    # ...an unrelated domain falls through to the category rule...
    assert mapper.context_for_domain("news.example.com") is None
    assert mapper.resolve("browser_work", "news.example.com") == "social"
    # ...and no domain at all resolves through the category only.
    assert mapper.context_for_domain("") is None
    assert mapper.resolve("browser_work") == "social"


def test_defaults_are_used_when_no_user_rule_matches() -> None:
    mapper = TaskContextMapper([])

    assert mapper.has_rules() is False
    assert mapper.context_for_category("code") == "coding"
    assert mapper.context_for_category("document") == "writing"
    assert mapper.context_for_category("browser_work") == "reading"
    assert mapper.context_for_category("communication") == "meeting"
    assert mapper.context_for_category("entertainment") == "entertainment"
    assert mapper.context_for_category("social") == "social"
    assert mapper.context_for_category("other") == TASK_CONTEXT_UNKNOWN
    assert mapper.context_for_category(None) == TASK_CONTEXT_UNKNOWN


# ── 6. the task columns reach the training matrix ──────────────────────────


def test_task_columns_flow_into_the_training_matrix_without_being_dropped() -> None:
    features = _window(
        process_task_contexts={"first.exe": "coding", "second.exe": "coding"},
        previous_task_context="writing",
    )
    data = prepare_v2_training_data([_training_row(features)], [_feedback(START)])
    column = {name: index for index, name in enumerate(V2_FEATURE_NAMES)}
    row = data.features[0]

    assert data.features.shape == (1, 28) == (1, len(V2_FEATURE_NAMES))
    assert [name for name in V2_FEATURE_NAMES if name in TASK_COLUMNS] == list(TASK_COLUMNS)
    assert row[column["task_type_code"]] == TASK_TYPE_CODES["coding"]
    assert row[column["task_type_entropy"]] == 0.0
    assert row[column["task_type_dominant_ratio"]] == 1.0
    assert row[column["task_context_transition"]] == 1.0
    assert row[column["task_unknown_ratio"]] == 0.0
    # Not merely present — non-default (v2/v3 always reported 0.0 for both of
    # these on a single-category window with no predecessor).
    assert row[column["task_type_code"]] != 0.0
    assert row[column["task_context_transition"]] != 0.0
    assert data.labels.tolist() == [1]


def test_schema_declares_the_five_task_columns_at_the_end_of_the_vocabulary() -> None:
    assert len(V2_FEATURE_NAMES) == 28
    assert FEATURE_SCHEMA_VERSION == 4
    assert list(V2_FEATURE_NAMES)[-5:] == list(TASK_COLUMNS)
    # The v2-era task column keeps its historical position; the four v4 columns
    # are appended, so name-addressed readers are unaffected by the bump.
    assert V2_FEATURE_NAMES.index("task_type_code") == 23
    assert len(set(V2_FEATURE_NAMES)) == len(V2_FEATURE_NAMES)


# ── 7. a non-current schema version is rejected, never read ────────────────


async def test_stale_schema_version_windows_are_not_listed(
    engine, session_factory, create_tables,
) -> None:
    repository = TelemetryRepository(session_factory=session_factory)
    features = _window(
        process_task_contexts={"first.exe": "coding", "second.exe": "coding"}
    )
    stale = _window_row(features, start=START)
    stale["feature_schema_version"] = FEATURE_SCHEMA_VERSION - 1
    current = _window_row(features, start=START + timedelta(minutes=5))
    await repository.upsert_feature_windows([stale, current])

    listed = await repository.list_feature_windows(1)
    legacy = await repository.list_feature_windows(1, FEATURE_SCHEMA_VERSION - 1)
    latest = await repository.latest_feature_window(1)
    after_stale = await repository.last_feature_window_before(1, START + timedelta(minutes=5))

    assert [row["feature_schema_version"] for row in listed] == [FEATURE_SCHEMA_VERSION]
    assert [row["feature_schema_version"] for row in legacy] == [FEATURE_SCHEMA_VERSION - 1]
    assert latest is not None
    assert latest["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    # A stale window must not seed the next window's task_context_transition
    # either: the rollup's predecessor lookup is version-filtered too.
    assert after_stale is None
    # The half-open range read is version-filtered as well: the current-version
    # query skips the stale row, the explicit older-version query still finds it.
    assert await repository.list_feature_windows_in_range(1, START, END) == []
    stale_only = await repository.list_feature_windows_in_range(
        1, START, END, FEATURE_SCHEMA_VERSION - 1
    )
    assert [row["feature_schema_version"] for row in stale_only] == [
        FEATURE_SCHEMA_VERSION - 1
    ]


def test_trainer_drops_windows_from_a_stale_schema_version() -> None:
    """One stale (v3-shaped) window and two current ones: only the current
    windows reach the matrix, so the version check happens before any column
    is read."""
    features = _window(
        process_task_contexts={"first.exe": "coding", "second.exe": "coding"}
    )
    stale = _training_row(features)
    stale["feature_schema_version"] = FEATURE_SCHEMA_VERSION - 1
    kept_rows = [
        _training_row(features, start=START),
        _training_row(features, start=START + timedelta(minutes=10)),
    ]

    data = prepare_v2_training_data([stale, *kept_rows], [_feedback(START)])

    assert data.features.shape == (len(kept_rows), len(V2_FEATURE_NAMES))
    assert data.label_sources == ["explicit"] * len(kept_rows)
    assert len(data.explicit_mask) == len(kept_rows)


# ── 8. explicit task columns round-trip through the repository ─────────────


async def test_explicit_task_type_code_survives_the_repository_round_trip(
    engine, session_factory, create_tables,
) -> None:
    repository = TelemetryRepository(session_factory=session_factory)
    features = _window(
        process_task_contexts={"first.exe": "meeting", "second.exe": "meeting"}
    )
    assert features["task_type_code"] == TASK_TYPE_CODES["meeting"]

    inserted = await repository.upsert_feature_windows([_window_row(features)])
    rows = await repository.list_feature_windows(1)

    assert len(inserted) == 1
    assert len(rows) == 1
    stored = rows[0]
    # The explicit column carries the observed code (not NULL and not 0.0)...
    assert stored["f24"] == TASK_TYPE_CODES["meeting"]
    # ...the four v4 columns land in f25..f28...
    assert stored["f25"] == features["task_type_entropy"]
    assert stored["f26"] == features["task_type_dominant_ratio"]
    assert stored["f27"] == features["task_context_transition"]
    assert stored["f28"] == features["task_unknown_ratio"]
    # ...and the JSON payload stays readable for consumers that parse it.
    payload = json.loads(stored["features_json"])
    assert payload["task_type_code"] == TASK_TYPE_CODES["meeting"]
    assert set(V2_FEATURE_NAMES) <= set(payload)

    # A re-roll of the same window must not drop the column.
    await repository.upsert_feature_windows([_window_row(features)])
    again = await repository.list_feature_windows(1)

    assert len(again) == 1
    assert again[0]["f24"] == TASK_TYPE_CODES["meeting"]
    assert json.loads(again[0]["features_json"])["task_type_code"] == TASK_TYPE_CODES["meeting"]


# ── 9. malformed rule rows are ignored, never raised ───────────────────────


def test_mapper_ignores_malformed_rule_rows() -> None:
    """Stored rows are data, not code: every unusable shape is skipped without
    raising, and the built-in defaults still answer."""
    malformed: list[dict[str, Any]] = [
        {},
        {"match_type": None, "match_value": "code", "task_context": "coding"},
        {"match_type": "regex", "match_value": "code", "task_context": "coding"},
        {"match_type": "category", "match_value": "productivity", "task_context": "coding"},
        {"match_type": "category", "match_value": "", "task_context": "coding"},
        {"match_type": "category"},  # missing match_value
        {"match_type": "domain", "match_value": "https://", "task_context": "reading"},
        {"match_type": "domain", "match_value": "https:///no-host", "task_context": "reading"},
        {"match_type": "domain", "match_value": "   ", "task_context": "reading"},
        {"match_type": "domain", "match_value": None, "task_context": "reading"},
        {"match_type": "domain"},  # missing match_value
    ]

    mapper = TaskContextMapper(malformed)

    assert mapper.rules == ()
    assert mapper.has_rules() is False
    # The defaults still answer, and no malformed row leaked a bogus context in.
    assert mapper.context_for_category("code") == "coding"
    assert mapper.context_for_domain("https://") is None
    assert mapper.resolve("code", "https://") == "coding"
    assert mapper.resolve(None, None) == TASK_CONTEXT_UNKNOWN


def test_mapper_keeps_usable_rows_and_repairs_unusable_fields() -> None:
    mapper = TaskContextMapper([
        # unusable priority -> 0, unrecognised context -> unknown
        {"match_type": "category", "match_value": "DOCUMENT", "task_context": "nonsense"},
        # a full URL is reduced to its host on the way in (privacy boundary)
        {
            "match_type": "domain",
            "match_value": "https://Docs.Python.org/3/library/asyncio.html?q=1",
            "task_context": "reading",
            "priority": "high",
        },
    ])

    assert len(mapper.rules) == 2
    assert mapper.rules[0].match_value == "document"
    assert mapper.rules[0].task_context == TASK_CONTEXT_UNKNOWN
    assert mapper.rules[1].match_value == "docs.python.org"
    assert mapper.rules[1].priority == 0
    assert mapper.context_for_category("document") == TASK_CONTEXT_UNKNOWN
    assert mapper.context_for_domain("docs.python.org") == "reading"


# ── service wiring: the rollup resolves real processes and domains ─────────


def _service(
    session_factory: Any,
    tmp_path: Any,
    task_context_rules: TaskContextRulesRepository | None = None,
) -> TelemetryService:
    return TelemetryService(
        repository=TelemetryRepository(session_factory=session_factory),
        preferences_repository=PreferencesRepository(session_factory=session_factory),
        data_dir=tmp_path,
        activity_repository=SQLAlchemyActivityRepository(session_factory=session_factory),
        baseline_repository=BaselineRepository(session_factory=session_factory),
        session_factory=session_factory,
        task_context_rules_repository=task_context_rules
        or TaskContextRulesRepository(session_factory=session_factory),
    )


async def test_rollup_resolution_maps_real_processes_and_domains(
    engine, session_factory, create_tables, tmp_path,
) -> None:
    """``code.exe`` -> coding and a documentation domain -> reading, resolved
    through the real app classifier plus the task-context rules layer (the
    builder itself carries no app or domain list)."""
    service = _service(session_factory, tmp_path)
    events = [
        _event("code.exe", offset_s=0, duration_s=60),
        _event("explorer.exe", offset_s=60, duration_s=60),
    ]
    browser = [{"domain": "docs.python.org", "browser_name": "edge", "duration_s": 30}]

    process_contexts, domain_contexts = await service._resolve_task_contexts(1, events, browser)

    assert process_contexts["code.exe"] == "coding"
    assert process_contexts["explorer.exe"] in TASK_CONTEXT_CATEGORIES
    assert domain_contexts["docs.python.org"] == "reading"


async def test_rollup_resolution_honours_user_task_context_rules(
    engine, session_factory, create_tables, tmp_path,
) -> None:
    """A user rule mapping the ``code`` activity category wins over the default,
    and a domain rule maps a browser domain the built-in classifier cannot."""
    rules = TaskContextRulesRepository(session_factory=session_factory)
    await rules.replace_all(1, [
        {"match_type": "category", "match_value": "code", "task_context": "entertainment"},
        {"match_type": "domain", "match_value": "news.example.com", "task_context": "reading"},
    ])
    service = _service(session_factory, tmp_path, task_context_rules=rules)

    process_contexts, domain_contexts = await service._resolve_task_contexts(
        1,
        [_event("code.exe", offset_s=0, duration_s=300)],
        [{"domain": "news.example.com", "browser_name": "edge", "duration_s": 30}],
    )

    assert process_contexts["code.exe"] == "entertainment"
    assert domain_contexts["news.example.com"] == "reading"


async def test_previous_task_context_reads_the_last_stored_window(
    engine, session_factory, create_tables, tmp_path,
) -> None:
    """The transition seed comes from the newest window on disk, and a stored
    legacy payload reads as unknown instead of crashing the rollup."""
    repository = TelemetryRepository(session_factory=session_factory)
    service = _service(session_factory, tmp_path)

    assert await service._previous_task_context(1, START) is None

    # A v2-shaped payload (no task columns at all) must not be guessed at.
    legacy = _window_row({"idle_ratio": 0.1})
    await repository.upsert_feature_windows([legacy])
    assert await service._previous_task_context(1, END) == TASK_CONTEXT_UNKNOWN

    coded = _window(
        process_task_contexts={"first.exe": "coding", "second.exe": "coding"}
    )
    await repository.upsert_feature_windows([_window_row(coded, start=END)])
    assert await service._previous_task_context(1, END + timedelta(minutes=5)) == "coding"
