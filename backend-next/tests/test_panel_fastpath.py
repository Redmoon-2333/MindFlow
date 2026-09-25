"""Phase 2.2 regressions: the panel fast path.

The fast path is a *routing decision*, not a shortcut around validation:

* complete, high-quality, confident, conflict-free evidence → skip the LLM
  panel and answer from the deterministic rule engine (or a single expert);
* incomplete evidence coverage → ``insufficient_data`` with no LLM call;
* anything ambiguous → the full panel.

The feature flag is OFF by default, and every route still executes through the
existing chain, so the crisis gate, forbidden-word guard, schema checks and
evidence/citation validation are unchanged.  These tests pin both the decision
table and the graph-level wiring.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mindflow.graph import analysis_graph as analysis
from mindflow.graph.panel_fastpath import (
    DEFAULT_FAST_PATH_CONFIG,
    EXPECTED_EVIDENCE_SOURCES,
    FastPathConfig,
    FastPathSignals,
    decide_panel_route,
    evidence_sources_present,
    signals_from_assessment,
    signals_from_payload,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


def _evidence_row(metric: str, severity: str = "moderate", confidence: float = 0.9) -> dict:
    return {
        "metric": metric,
        "severity": severity,
        "confidence": confidence,
        "human_readable": metric,
    }


def _full_payload() -> str:
    """A payload that covers every expected evidence source with good quality."""
    return json.dumps({
        "window": {"start": "2026-09-19T00:00:00+00:00", "end": "2026-09-20T00:00:00+00:00"},
        "evidence": [
            _evidence_row("focus.focus_score"),
            _evidence_row("focus.switch_rate"),
            _evidence_row("summary.social_media_ratio"),
            _evidence_row("summary.start_delay_min"),
        ],
        "stable_summary": {"collapsed_items": 0, "collapsed_metrics": 0, "metrics": {}},
        "behavior_summary": {"duration_min": 120.0},
        "intervention_history": [],
        "novelty_flags": [],
        "evidence_catalog": [{"id": "focus.focus_score", "label": "专注度", "type": "metric"}],
    }, ensure_ascii=False)


def _events() -> list[Any]:
    """One activity event so the rule engine has a behavior summary to judge."""
    from mindflow.domain.events import make_event

    return [make_event(
        user_id=1, timestamp_utc=datetime(2026, 9, 19, 9, 0, tzinfo=UTC),
        duration_s=300.0, process_name="Code.exe",
    )]


def _confident_assessment() -> dict[str, Any]:
    return {
        "procrastination_types": ["impulsivity"],
        "type_confidence": {"impulsivity": 0.9, "task_aversion": 0.1},
        "cognitive_distortions": [],
        "cbt_technique": "stimulus_control",
        "response_text": "规则结论",
    }


def _bundle_stub(rows: list[tuple[str, str, float]]) -> SimpleNamespace:
    return SimpleNamespace(
        items=tuple(
            SimpleNamespace(metric=metric, severity=severity, confidence=confidence)
            for metric, severity, confidence in rows
        )
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Decision table
# ═══════════════════════════════════════════════════════════════════════════════


def test_high_quality_confident_evidence_takes_the_fast_path() -> None:
    decision = decide_panel_route(FastPathSignals(
        rule_confidence=0.9, confidence_margin=0.5,
        evidence_coverage=1.0, evidence_quality=0.9,
    ))
    assert decision.route == "rule_engine"
    assert decision.runs_full_panel is False


def test_single_expert_is_available_as_the_fast_path() -> None:
    decision = decide_panel_route(
        FastPathSignals(
            rule_confidence=0.9, confidence_margin=0.5,
            evidence_coverage=1.0, evidence_quality=0.9,
        ),
        FastPathConfig(prefer_single_expert=True),
    )
    assert decision.route == "single_expert"


def test_low_coverage_returns_insufficient_data() -> None:
    decision = decide_panel_route(FastPathSignals(
        rule_confidence=0.9, confidence_margin=0.5,
        evidence_coverage=0.5, evidence_quality=0.9,
        missing_sources=("social", "delay"),
    ))
    assert decision.route == "insufficient_data"
    assert "social" in decision.reason


def test_a_missing_collector_is_decisive_even_at_full_category_coverage() -> None:
    """Collector gaps come from the quality record; one absent category is not."""
    decision = decide_panel_route(FastPathSignals(
        rule_confidence=0.95, confidence_margin=0.9,
        evidence_coverage=1.0, evidence_quality=0.9,
        collector_gaps=("browser_tracking_disabled",),
    ))
    assert decision.route == "insufficient_data"
    assert "browser_tracking_disabled" in decision.reason


def test_one_absent_category_alone_does_not_force_insufficient_data() -> None:
    """Windows legitimately differ in content; only coverage/collectors decide."""
    decision = decide_panel_route(FastPathSignals(
        rule_confidence=0.95, confidence_margin=0.9,
        evidence_coverage=0.75, evidence_quality=0.9,
        missing_sources=("delay",),
    ))
    assert decision.route == "rule_engine"


@pytest.mark.parametrize(
    ("label", "signals"),
    [
        ("rule conflict", FastPathSignals(
            rule_confidence=0.8, confidence_margin=0.05,
            evidence_coverage=1.0, evidence_quality=0.9, rule_conflict=True,
        )),
        ("low rule confidence", FastPathSignals(
            rule_confidence=0.5, confidence_margin=0.4,
            evidence_coverage=1.0, evidence_quality=0.9,
        )),
        ("low evidence quality", FastPathSignals(
            rule_confidence=0.9, confidence_margin=0.4,
            evidence_coverage=1.0, evidence_quality=0.3,
        )),
        ("history disagreement", FastPathSignals(
            rule_confidence=0.9, confidence_margin=0.4,
            evidence_coverage=1.0, evidence_quality=0.9, history_disagreement=True,
        )),
    ],
)
def test_ambiguous_evidence_requires_the_full_panel(label: str, signals: FastPathSignals) -> None:
    decision = decide_panel_route(signals)
    assert decision.route == "full_panel", label
    assert decision.runs_full_panel is True
    assert decision.reason


def test_thresholds_are_configurable_not_magic_numbers() -> None:
    relaxed = FastPathConfig(min_rule_confidence=0.4, min_evidence_quality=0.2)
    decision = decide_panel_route(
        FastPathSignals(
            rule_confidence=0.5, confidence_margin=0.4,
            evidence_coverage=1.0, evidence_quality=0.3,
        ),
        relaxed,
    )
    assert decision.route == "rule_engine"


def test_default_thresholds_are_the_documented_ones() -> None:
    assert DEFAULT_FAST_PATH_CONFIG.min_evidence_coverage == 0.6
    assert DEFAULT_FAST_PATH_CONFIG.min_evidence_quality == 0.6
    assert DEFAULT_FAST_PATH_CONFIG.min_rule_confidence == 0.75
    assert DEFAULT_FAST_PATH_CONFIG.min_confidence_margin == 0.2
    assert DEFAULT_FAST_PATH_CONFIG.prefer_single_expert is False


# ═══════════════════════════════════════════════════════════════════════════════
# Signal extraction
# ═══════════════════════════════════════════════════════════════════════════════


def test_expected_sources_are_four_documented_categories() -> None:
    assert EXPECTED_EVIDENCE_SOURCES == ("focus", "switch", "social", "delay")


@pytest.mark.parametrize(
    ("metrics", "present", "missing"),
    [
        (["focus.focus_score", "focus.switch_rate", "summary.social_media_ratio",
          "summary.start_delay_min"], 4, 0),
        (["focus.focus_score"], 1, 3),
        ([], 0, 4),
    ],
)
def test_source_coverage_counts_categories(metrics, present, missing) -> None:
    got_present, got_missing = evidence_sources_present(metrics)
    assert len(got_present) == present
    assert len(got_missing) == missing


def test_payload_signals_read_the_compressed_payload() -> None:
    signals = signals_from_payload(_full_payload(), _confident_assessment())

    assert signals.evidence_coverage == 1.0
    assert signals.rule_confidence == 0.9
    assert signals.confidence_margin == pytest.approx(0.8)
    assert signals.missing_sources == ()
    assert signals.evidence_quality == pytest.approx(0.9)
    assert signals.rule_conflict is False


def test_payload_signals_count_collapsed_stable_metrics_as_present() -> None:
    """A metric collapsed into ``stable_summary`` is still observed evidence."""
    payload = json.dumps({
        "evidence": [],
        "stable_summary": {
            "collapsed_items": 3,
            "collapsed_metrics": 2,
            "metrics": {"focus.focus_score": {"count": 2, "mean": 0.5},
                        "focus.switch_rate": {"count": 1, "mean": 4.0}},
        },
    }, ensure_ascii=False)
    signals = signals_from_payload(payload, _confident_assessment())

    # Keyword matching: "focus.focus_score" counts for focus, and
    # "focus.switch_rate" counts for both focus and switch.
    present, _ = evidence_sources_present(["focus.focus_score", "focus.switch_rate"])
    assert present == frozenset({"focus", "switch"})
    assert signals.evidence_coverage == pytest.approx(0.5)


def test_full_coverage_with_a_collector_gap_is_insufficient() -> None:
    """A disabled collector is decisive even when every category has a metric."""
    payload = json.dumps({
        "evidence": [
            _evidence_row("focus.focus_score"),
            _evidence_row("focus.switch_rate"),
            _evidence_row("summary.social_media_ratio"),
            _evidence_row("summary.start_delay_min"),
        ],
        "stable_summary": {"metrics": {}},
        "collector_gaps": ["browser_tracking_disabled"],
    }, ensure_ascii=False)
    signals = signals_from_payload(payload, _confident_assessment())

    assert signals.evidence_coverage == 1.0
    assert signals.collector_gaps == ("browser_tracking_disabled",)
    decision = decide_panel_route(signals)
    assert decision.route == "insufficient_data"
    assert "browser_tracking_disabled" in decision.reason


def test_window_quality_gaps_also_count_as_collector_gaps() -> None:
    payload = json.dumps({
        "evidence": [
            _evidence_row("focus.focus_score"),
            _evidence_row("focus.switch_rate"),
            _evidence_row("summary.social_media_ratio"),
            _evidence_row("summary.start_delay_min"),
        ],
        "window_quality": {"gaps": ["input_telemetry_disabled"]},
    }, ensure_ascii=False)
    assert signals_from_payload(payload, _confident_assessment()).collector_gaps == (
        "input_telemetry_disabled",
    )


def test_malformed_payload_does_not_look_confident() -> None:
    signals = signals_from_payload("not json at all", _confident_assessment())
    assert signals.evidence_coverage == 0.0
    assert decide_panel_route(signals).route == "insufficient_data"


def test_bundle_signals_use_item_severity_for_quality() -> None:
    bundle = _bundle_stub([
        ("focus.focus_score", "severe", 0.8),
        ("focus.switch_rate", "info", 0.1),  # stable items carry no quality signal
        ("summary.social_media_ratio", "moderate", 1.0),
        ("summary.start_delay_min", "info", 0.2),
    ])
    signals = signals_from_assessment(bundle, _confident_assessment())

    assert signals.evidence_quality == pytest.approx(0.9)
    assert signals.evidence_coverage == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Graph wiring
# ═══════════════════════════════════════════════════════════════════════════════


def _settings(enabled: bool) -> MagicMock:
    return MagicMock(panel_fast_path_enabled=enabled)


class _RuleEngineStub:
    """Deterministic rule-engine double with a controllable conclusion."""

    def __init__(self, confidence: dict[str, float]) -> None:
        self._confidence = confidence
        self.calls = 0

    def assess(self, summary: Any) -> SimpleNamespace:
        self.calls += 1
        ordered = sorted(self._confidence.items(), key=lambda kv: kv[1], reverse=True)
        return SimpleNamespace(
            types=[name for name, _ in ordered],
            confidence=dict(self._confidence),
            recommended_technique="stimulus_control",
            rationale="规则引擎结论",
        )


async def _run_panel_node(
    monkeypatch: pytest.MonkeyPatch,
    *,
    flag: bool,
    payload: str,
    confidence: dict[str, float],
    panel_ainvoke: AsyncMock | None = None,
) -> tuple[dict[str, Any], AsyncMock]:
    monkeypatch.setattr(analysis, "get_settings", lambda: _settings(flag), raising=False)
    from mindflow.config import get_settings as real_get_settings

    monkeypatch.setattr(
        "mindflow.config.get_settings", lambda: _settings(flag),
    )
    assert real_get_settings is not None

    panel = SimpleNamespace(
        ainvoke=panel_ainvoke or AsyncMock(return_value={
            "moderator_verdict": {
                "types": ["impulsivity"],
                "confidence": {"impulsivity": 0.9},
                "recommended_technique": "stimulus_control",
                "rationale": "panel",
            },
            "critic_approved": True,
            "panel_terminal": "approved",
            "panel_rejected": False,
            "rejection_reason": "",
            "call_count": 6,
        })
    )
    runtime = analysis.AnalysisRunContext(
        panel_graph=panel,
        rule_engine=_RuleEngineStub(confidence),
    )
    state: dict[str, Any] = {
        "runtime": runtime,
        "user_id": 1,
        "target_date": date(2026, 9, 19),
        "run_id": "",
        "bundle_json": payload,
        "valid_metrics": ("focus.focus_score",),
        "events_domain": _events(),
    }
    update = await analysis.panel_graph_node(state)
    return update, panel.ainvoke


async def test_flag_off_always_runs_the_full_panel(monkeypatch: pytest.MonkeyPatch) -> None:
    update, ainvoke = await _run_panel_node(
        monkeypatch, flag=False, payload=_full_payload(),
        confidence={"impulsivity": 0.95, "task_aversion": 0.05},
    )

    ainvoke.assert_awaited_once()
    assert update["panel_succeeded"] is True
    assert update["source"] == "panel"


async def test_fast_path_skips_the_panel_and_delegates_to_the_rule_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    update, ainvoke = await _run_panel_node(
        monkeypatch, flag=True, payload=_full_payload(),
        confidence={"impulsivity": 0.95, "task_aversion": 0.05},
    )

    ainvoke.assert_not_awaited()
    assert update["panel_succeeded"] is False
    assert update["panel_fast_path_route"] == "rule_engine"
    assert update["panel_degradation_marker"] == "panel_fast_path_rule_engine"
    # The answer still comes from the validated chain, never from the panel.
    assert update.get("source") != "panel"


async def test_fast_path_insufficient_evidence_spends_no_llm_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.dumps({
        "evidence": [_evidence_row("focus.focus_score")],
        "stable_summary": {"metrics": {}},
        "collector_gaps": ["browser_tracking_disabled"],
    })
    update, ainvoke = await _run_panel_node(
        monkeypatch, flag=True, payload=payload,
        confidence={"impulsivity": 0.95},
    )

    ainvoke.assert_not_awaited()
    assert update["panel_fast_path_route"] == "insufficient_data"
    assert update["panel_degradation_marker"] == "panel_insufficient_evidence"
    assert update["fallback_to_rule_engine"] is True


async def test_fast_path_defers_to_the_panel_when_the_rules_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    update, ainvoke = await _run_panel_node(
        monkeypatch, flag=True, payload=_full_payload(),
        confidence={"impulsivity": 0.52, "task_aversion": 0.50},
    )

    ainvoke.assert_awaited_once()
    assert update["panel_succeeded"] is True
    assert update.get("panel_fast_path_route") is None


async def test_without_a_rule_engine_the_panel_always_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("mindflow.config.get_settings", lambda: _settings(True))
    panel_ainvoke = AsyncMock(return_value={
        "moderator_verdict": {
            "types": ["impulsivity"], "confidence": {"impulsivity": 0.9},
            "recommended_technique": "stimulus_control", "rationale": "panel",
        },
        "critic_approved": True, "panel_terminal": "approved",
        "panel_rejected": False, "rejection_reason": "", "call_count": 6,
    })
    runtime = analysis.AnalysisRunContext(
        panel_graph=SimpleNamespace(ainvoke=panel_ainvoke), rule_engine=None,
    )
    update = await analysis.panel_graph_node({
        "runtime": runtime, "user_id": 1, "target_date": date(2026, 9, 19),
        "run_id": "", "bundle_json": _full_payload(), "valid_metrics": (),
        "events_domain": _events(),
    })

    panel_ainvoke.assert_awaited_once()
    assert update["panel_succeeded"] is True


async def test_feedback_disagreement_probe_forces_the_full_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("mindflow.config.get_settings", lambda: _settings(True))
    panel_ainvoke = AsyncMock(return_value={
        "moderator_verdict": {
            "types": ["impulsivity"], "confidence": {"impulsivity": 0.9},
            "recommended_technique": "stimulus_control", "rationale": "panel",
        },
        "critic_approved": True, "panel_terminal": "approved",
        "panel_rejected": False, "rejection_reason": "", "call_count": 6,
    })

    async def _probe(user_id: int, top_type: str) -> bool:
        assert user_id == 1
        assert top_type == "impulsivity"
        return True

    runtime = analysis.AnalysisRunContext(
        panel_graph=SimpleNamespace(ainvoke=panel_ainvoke),
        rule_engine=_RuleEngineStub({"impulsivity": 0.95, "task_aversion": 0.05}),
        feedback_disagreement_probe=_probe,
    )
    update = await analysis.panel_graph_node({
        "runtime": runtime, "user_id": 1, "target_date": date(2026, 9, 19),
        "run_id": "", "bundle_json": _full_payload(),
        "valid_metrics": (), "events_domain": _events(),
    })

    panel_ainvoke.assert_awaited_once()
    assert update["panel_succeeded"] is True


def test_fast_path_route_is_part_of_the_graph_state_contract() -> None:
    """The trace field exists so every skipped panel is explainable."""
    assert "panel_fast_path_route" in analysis.AnalysisGraphState.__annotations__


def test_datetime_utc_import_is_used() -> None:
    """Guard against an unused import creeping in while this module evolves."""
    assert datetime.now(UTC).tzinfo is UTC
