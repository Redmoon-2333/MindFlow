"""Phase-2.4 EvidenceBundle prompt compression — contracts and measurements.

Pins ``to_prompt_json(bundle, compressed=True)`` (the production default)
against ``compressed=False`` (the legacy payload kept for A/B comparison), as
implemented in ``src/mindflow/domain/evidence.py``:

1. every metric id is emitted exactly once in the *values* representation
   (``evidence`` rows for actionable items, ``stable_summary.metrics`` for
   collapsed ones) and exactly once in the citation catalog;
2. ``evidence_catalog`` rows carry ``id``/``label``/``type`` only — values,
   baselines, confidence and severity live once, in the evidence rows;
3. stable, non-deviating, non-intervention-related metrics are summarised
   (``count`` [+ ``mean``]), while anomalous / baseline-deviating /
   recent-intervention items stay in ``evidence`` as full rows;
4. intervention relatedness is decided by the *newest* intervention type
   (``EvidenceCompressionPolicy.intervention_window``) through literal token
   containment in the metric name;
5. ``compressed=False`` still serialises the pre-2.4 shape;
6. citation recall (the catalog id set) is identical in both modes — checked
   on the panel fixture, the 30 eval scenarios and the local fixtures;
7. measured payload reduction, printed for every fixture (run with ``-s``).

Measurement policy: the plan's ≥30% target is met by a realistic window with
many stable metrics (see ``_stable_heavy_bundle``).  A small anomaly-heavy
window — the panel and eval fixtures, where every item is already kept in
full — has almost nothing left to collapse; those numbers are reported as
measured rather than dressed up as a target hit.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from mindflow.domain.evidence import (
    DEFAULT_COMPRESSION_POLICY,
    EvidenceBundle,
    EvidenceCompressionPolicy,
    EvidenceItem,
    InterventionRecord,
    Severity,
    is_actionable_item,
    to_prompt_json,
)
from mindflow.domain.evidence_facts import build_evidence_catalog, evidence_catalog_ids
from mindflow.domain.procrastination import BehaviorSummary
from mindflow.eval.scenarios import ALL_SCENARIOS
from tests.test_agents_orchestrator import _make_bundle as _panel_bundle

# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

_NOW = datetime(2026, 9, 24, 14, 0, tzinfo=UTC)
_WINDOW = (datetime(2026, 9, 24, 8, 0, tzinfo=UTC), _NOW)


def _item(
    metric: str,
    value: float | str,
    *,
    baseline: float | None = None,
    severity: Severity = "info",
    confidence: float = 0.8,
    source: str = "feature_computation",
    human_readable: str | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        metric=metric,
        value=value,
        baseline=baseline,
        severity=severity,
        confidence=confidence,
        source=source,
        human_readable=human_readable or f"{metric} {value}",
    )


def _summary(**overrides: Any) -> BehaviorSummary:
    fields: dict[str, Any] = {
        "intended_task": "写论文",
        "duration_min": 420.0,
        "actual_focus_min": 246.0,
        "context_switches_per_hour": 12.4,
        "longest_focus_block_s": 1620.0,
        "social_media_ratio": 0.14,
        "start_delay_min": 8.0,
        "keyword_flags": frozenset(),
        "baseline_deviation": None,
    }
    fields.update(overrides)
    return BehaviorSummary(**fields)


def _bundle(
    items: tuple[EvidenceItem, ...],
    *,
    interventions: tuple[InterventionRecord, ...] = (),
    novelty: tuple[str, ...] = (),
    summary: BehaviorSummary | None = None,
) -> EvidenceBundle:
    return EvidenceBundle(
        user_id=1,
        window=_WINDOW,
        items=items,
        behavior_summary=summary if summary is not None else _summary(),
        intervention_history=interventions,
        novelty_flags=novelty,
    )


def _intervention(
    kind: str,
    *,
    minutes_ago: int = 30,
    response: str | None = "ignored",
) -> InterventionRecord:
    return InterventionRecord(
        intervention_type=kind,
        triggered_at=_NOW - timedelta(minutes=minutes_ago),
        user_response=response,
        effect_note="用户忽略，专注未恢复",
    )


def _mixed_bundle() -> EvidenceBundle:
    """One window mixing every actionability reason with simply stable items.

    Actionable: ``focus_score`` (severity) and ``behavior_deviation`` (info but
    significantly off its 0.0 baseline).  Stable: ``switch_rate`` (within
    tolerance), ``start_delay_min`` (within tolerance), the two ``ml_*`` rows
    without a baseline and the non-numeric ``top_apps``.
    """
    return _bundle((
        _item("focus_score", 42.0, baseline=72.0, severity="moderate", confidence=0.85,
              human_readable="专注度评分 42.0/100，明显偏低"),
        _item("behavior_deviation", 1.8, baseline=0.0, confidence=0.43,
              source="welford_baseline", human_readable="行为模式偏离基线 1.8 个标准差"),
        _item("switch_rate", 8.4, baseline=8.0, human_readable="应用切换频率 8.4 次/小时，正常"),
        _item("start_delay_min", 10.5, baseline=10.0, human_readable="启动延迟 10.5 分钟，正常"),
        _item("ml_uncertainty", 0.05, confidence=0.95, source="rf_classifier_v2",
              human_readable="ML预测不确定度 0.05（低）"),
        _item("ml_feature_coverage", 1.0, confidence=1.0, source="rf_classifier_v2",
              human_readable="ML特征覆盖率 100%（48个窗口）"),
        _item("top_apps", "Code.exe, chrome.exe, wechat.exe", confidence=0.85,
              human_readable="主要使用: Code.exe, chrome.exe, wechat.exe"),
    ))


# The stable half of ``_mixed_bundle`` (order of ``stable_summary.metrics``).
_MIXED_STABLE = (
    "switch_rate",
    "start_delay_min",
    "ml_uncertainty",
    "ml_feature_coverage",
    "top_apps",
)
_MIXED_ACTIONABLE = ("focus_score", "behavior_deviation")


def _task_item() -> EvidenceItem:
    """A task-context metric, stable by value but matching a ``task_*`` type.

    ``task_breakdown`` splits into the tokens ``task``/``breakdown``, and
    ``task`` occurs in this metric name — the only way the production metric
    vocabulary can be intervention-related.
    """
    return _item("task_type_dominant_ratio", 0.62, baseline=0.60, source="task_context",
                 human_readable="主导任务类型占比 62%（基线 60%）")


def _production_bundle(intervention: str = "nudge") -> EvidenceBundle:
    """A production-shaped window: 4 rule items + 4 ML items + deviation item.

    Mirrors ``EvidenceBundleBuilder`` (``services/evidence_service.py``) on a
    mixed day: focus/switch/ML are anomalous, the block/coverage/uncertainty
    rows and the zero deviation are stable.
    """
    return _bundle(
        (
            _item("focus_score", 58.4, severity="mild", confidence=0.85,
                  human_readable="专注度评分 58.4/100，偏低"),
            _item("switch_rate", 36.2, severity="moderate", confidence=0.85,
                  human_readable="应用切换频率 36.2 次/小时，较高"),
            _item("longest_block", 720.0, confidence=0.85,
                  human_readable="最长专注块 12 分钟，良好"),
            _item("top_apps", "Code.exe, chrome.exe, wechat.exe", confidence=0.85,
                  human_readable="主要使用: Code.exe, chrome.exe, wechat.exe"),
            _item("ml_focus_probability", 0.412, severity="moderate", confidence=0.588,
                  source="rf_classifier_v2", human_readable="ML专注概率 41%，分心倾向"),
            _item("ml_distracted_window_ratio", 0.52, severity="moderate", confidence=0.52,
                  source="rf_classifier_v2", human_readable="ML分心窗口占比 52%"),
            _item("ml_uncertainty", 0.0, confidence=1.0, source="rf_classifier_v2",
                  human_readable="ML预测不确定度 0.00（低）"),
            _item("ml_feature_coverage", 1.0, confidence=1.0, source="rf_classifier_v2",
                  human_readable="ML特征覆盖率 100%（50个窗口）"),
            _item("behavior_deviation", 0.0, baseline=0.0, confidence=0.0,
                  source="welford_baseline", human_readable="行为模式与基线一致，无显著偏差"),
        ),
        interventions=(_intervention(intervention),),
        summary=_summary(duration_min=360.0, actual_focus_min=210.0,
                         context_switches_per_hour=36.2, longest_focus_block_s=720.0,
                         social_media_ratio=0.18, start_delay_min=9.0),
    )


# Routine-day per-feature baseline rows: (metric, value, baseline, label).
# Every row but ``idle_ratio`` is inside the policy tolerance, so it collapses
# into ``stable_summary``.
_FEATURE_ROWS: tuple[tuple[str, float, float, str], ...] = (
    ("idle_ratio", 0.31, 0.18, "空闲占比 31%（基线 18%）"),
    ("input_active_ratio", 0.64, 0.62, "输入活跃占比 64%（基线 62%）"),
    ("browser_ratio", 0.28, 0.27, "浏览器占比 28%（基线 27%）"),
    ("audible_browser_ratio", 0.09, 0.10, "有声浏览器占比 9%（基线 10%）"),
    ("active_seconds_ratio", 0.71, 0.70, "活跃秒数占比 71%（基线 70%）"),
    ("top_app_ratio", 0.46, 0.45, "头部应用占比 46%（基线 45%）"),
    ("app_switch_count", 41.0, 44.0, "应用切换 41 次（基线 44 次）"),
    ("domain_switch_count", 63.0, 66.0, "域名切换 63 次（基线 66 次）"),
    ("longest_segment_ratio", 0.22, 0.23, "最长片段占比 22%（基线 23%）"),
    ("interaction_interval_cv", 0.86, 0.84, "交互间隔变异系数 0.86（基线 0.84）"),
    ("interaction_bursts_per_min", 3.4, 3.5, "交互爆发 3.4 次/分（基线 3.5）"),
    ("keypress_rate_per_min", 128.0, 131.0, "按键频率 128 次/分（基线 131）"),
    ("mouse_click_rate_per_min", 22.0, 21.5, "点击频率 22.0 次/分（基线 21.5）"),
    ("scroll_rate_per_min", 96.0, 92.0, "滚动频率 96 次/分（基线 92）"),
    ("mouse_distance_per_min", 5400.0, 5200.0, "鼠标移动 5400px/分（基线 5200）"),
    ("click_key_ratio", 0.17, 0.16, "点击/按键比 0.17（基线 0.16）"),
    ("task_type_dominant_ratio", 0.58, 0.56, "主导任务类型占比 58%（基线 56%）"),
    ("task_type_entropy", 0.42, 0.44, "任务类型熵 0.42（基线 0.44）"),
    ("task_unknown_ratio", 0.06, 0.05, "未知任务占比 6%（基线 5%）"),
    ("task_context_transition", 2.0, 2.0, "任务上下文切换 2 次（基线 2 次）"),
)


def _stable_heavy_bundle(intervention: str = "nudge") -> EvidenceBundle:
    """A full routine-day window: 27 stable metrics + 4 actionable ones.

    Superset of today's ``EvidenceBundleBuilder`` output (rule + ML +
    deviation rows) plus the per-feature baseline rows the Welford baseline
    layer tracks.  This is the realistic *standard analysis* shape the ≥30%
    compression target is stated for: many stable metrics, a few anomalies.

    ``intervention`` defaults to the generic ``nudge`` (which matches no metric
    name, so nothing extra is retained).  Passing ``task_breakdown`` keeps the
    four ``task_*`` rows in full — the intervention-relatedness branch, at the
    cost of some compression.
    """
    items: list[EvidenceItem] = [
        _item("focus_score", 74.2, confidence=0.85, human_readable="专注度评分 74.2/100，正常"),
        _item("switch_rate", 12.4, confidence=0.85,
              human_readable="应用切换频率 12.4 次/小时，正常"),
        _item("longest_block", 1620.0, confidence=0.85, human_readable="最长专注块 27 分钟，良好"),
        _item("top_apps", "Code.exe, chrome.exe, wechat.exe", confidence=0.85,
              human_readable="主要使用: Code.exe, chrome.exe, wechat.exe"),
        _item("ml_focus_probability", 0.76, confidence=0.76, source="rf_classifier_v2",
              human_readable="ML专注概率 76%，专注倾向"),
        _item("ml_distracted_window_ratio", 0.21, confidence=0.79, source="rf_classifier_v2",
              human_readable="ML分心窗口占比 21%"),
        _item("ml_uncertainty", 0.08, confidence=0.92, source="rf_classifier_v2",
              human_readable="ML预测不确定度 0.08（低）"),
        _item("ml_feature_coverage", 0.98, confidence=1.0, source="rf_classifier_v2",
              human_readable="ML特征覆盖率 98%（47个窗口）"),
        # Actionable: info + significant deviation from the 0.0 baseline.
        _item("behavior_deviation", 0.42, baseline=0.0, confidence=0.1,
              source="welford_baseline", human_readable="行为模式偏离基线 0.42 个标准差"),
        # Actionable: severity != info.
        _item("social_media_ratio", 0.38, baseline=0.20, severity="mild", confidence=0.75,
              human_readable="娱乐占比 38%（基线 20%）"),
        _item("start_delay_min", 34.0, baseline=10.0, severity="moderate", confidence=0.80,
              human_readable="启动延迟 34min（基线 10min）"),
    ]
    items.extend(
        _item(metric, value, baseline=baseline, human_readable=label)
        for metric, value, baseline, label in _FEATURE_ROWS
    )
    return _bundle(
        tuple(items),
        interventions=(_intervention(intervention, minutes_ago=45),),
    )


_STABLE_HEAVY_ACTIONABLE = (
    "behavior_deviation",
    "social_media_ratio",
    "start_delay_min",
    "idle_ratio",
)

# With a ``task_breakdown`` intervention the four ``task_*`` feature rows are
# intervention-related and therefore stay in full as well.
_STABLE_HEAVY_TASK_ACTIONABLE = (
    *_STABLE_HEAVY_ACTIONABLE,
    "task_type_dominant_ratio",
    "task_type_entropy",
    "task_unknown_ratio",
    "task_context_transition",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Payload helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _payload(
    bundle: EvidenceBundle,
    *,
    compressed: bool = True,
    policy: EvidenceCompressionPolicy = DEFAULT_COMPRESSION_POLICY,
) -> dict[str, Any]:
    return json.loads(to_prompt_json(bundle, compressed=compressed, policy=policy))


def _value_metric_ids(payload: dict[str, Any]) -> list[str]:
    """Metric ids in the *values* representation: rows + collapsed summary."""
    rows = [str(row["metric"]) for row in payload.get("evidence", ())]
    collapsed = [str(name) for name in (payload.get("stable_summary") or {}).get("metrics", {})]
    return rows + collapsed


def _catalog_ids(payload: dict[str, Any]) -> list[str]:
    """Catalog ids, from either encoding.

    The compressed payload uses positional ``[id, label, type]`` rows (the key
    names cost ~34% of the payload); the legacy payload uses objects with a
    ``label_zh``/``value`` shape.  Both are read here so the same assertions
    cover either contract.
    """
    ids: list[str] = []
    for entry in payload.get("evidence_catalog", ()):
        if isinstance(entry, dict):
            ids.append(str(entry["id"]))
        else:
            ids.append(str(entry[0]))
    return ids


def _catalog_rows(payload: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Catalog rows normalised to ``(id, label, type)`` tuples."""
    rows: list[tuple[str, str, str]] = []
    for entry in payload.get("evidence_catalog", ()):
        if isinstance(entry, dict):
            rows.append((str(entry["id"]), str(entry["label"]), str(entry["type"])))
        else:
            rows.append((str(entry[0]), str(entry[1]), str(entry[2])))
    return rows


def _canonical_id(metric: str) -> str:
    """Catalog id for an item metric (mirrors ``build_evidence_catalog``)."""
    return ("ml." if metric.startswith("ml_") else "focus.") + metric


def _sizes(bundle: EvidenceBundle) -> tuple[int, int, float]:
    before = to_prompt_json(bundle, compressed=False)
    after = to_prompt_json(bundle)
    return len(before), len(after), (1.0 - len(after) / len(before)) * 100.0


def _sections(payload: dict[str, Any]) -> dict[str, int]:
    """Per-section character cost, so the measurement shows where bytes go."""
    return {
        name: len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        for name, value in payload.items()
    }


def _report(label: str, bundle: EvidenceBundle) -> float:
    before, after, pct = _sizes(bundle)
    legacy = _payload(bundle, compressed=False)
    compressed = _payload(bundle)
    before_sections = _sections(legacy)
    after_sections = _sections(compressed)
    stable_metrics = (compressed.get("stable_summary") or {}).get("metrics", {})
    print(
        f"\n[evidence-compression] {label}: "
        f"before={before}c after={after}c reduction={pct:.1f}% "
        f"evidence_rows={len(legacy['evidence'])}->{len(compressed['evidence'])} "
        f"catalog={len(_catalog_ids(legacy))}->{len(_catalog_ids(compressed))} "
        f"stable_metrics={len(stable_metrics)}"
    )
    print(
        f"[evidence-compression]   before sections: evidence={before_sections['evidence']}c "
        f"catalog={before_sections['evidence_catalog']}c "
        f"behaviour={before_sections['behavior_summary']}c"
    )
    print(
        f"[evidence-compression]   after  sections: evidence={after_sections['evidence']}c "
        f"stable_summary={after_sections.get('stable_summary', 0)}c "
        f"catalog={after_sections['evidence_catalog']}c "
        f"behaviour={after_sections['behavior_summary']}c"
    )
    return pct


# ═══════════════════════════════════════════════════════════════════════════════
# 1 + 2. Section split: one value per metric, catalog = citation namespace only
# ═══════════════════════════════════════════════════════════════════════════════


class TestSectionSplit:
    """``evidence`` holds data once; ``evidence_catalog`` holds ids/labels only."""

    @pytest.mark.parametrize(
        "bundle",
        [_mixed_bundle(), _production_bundle(), _stable_heavy_bundle(), _panel_bundle()],
        ids=["mixed", "production", "stable_heavy", "panel"],
    )
    def test_each_metric_id_appears_once_in_values_representation(
        self, bundle: EvidenceBundle
    ) -> None:
        payload = _payload(bundle)
        metric_ids = _value_metric_ids(payload)

        # No metric id is emitted twice across evidence rows + stable summary.
        assert len(metric_ids) == len(set(metric_ids)), f"duplicate metric id: {metric_ids}"
        assert set(metric_ids) == {item.metric for item in bundle.items}
        # The two sub-sections are disjoint: a metric is detailed OR summarised.
        detailed = {str(row["metric"]) for row in payload["evidence"]}
        collapsed = set((payload.get("stable_summary") or {}).get("metrics", {}))
        assert detailed.isdisjoint(collapsed)

    @pytest.mark.parametrize(
        "bundle",
        [_mixed_bundle(), _production_bundle(), _stable_heavy_bundle(), _panel_bundle()],
        ids=["mixed", "production", "stable_heavy", "panel"],
    )
    def test_catalog_has_exactly_one_entry_per_metric(self, bundle: EvidenceBundle) -> None:
        ids = _catalog_ids(_payload(bundle))
        assert len(ids) == len(set(ids)), f"duplicate catalog id: {ids}"
        for item in bundle.items:
            assert ids.count(_canonical_id(item.metric)) == 1, item.metric

    def test_catalog_rows_carry_only_id_label_type(self) -> None:
        payload = _payload(_mixed_bundle())
        assert payload["evidence_catalog"], "catalog must not be empty"
        for entry in payload["evidence_catalog"]:
            # Positional encoding: [id, label, type] with no repeated key names
            # (the keys cost ~34% of the payload on a production-shaped window).
            assert isinstance(entry, list), entry
            assert len(entry) == 3, entry
            assert all(isinstance(value, str) for value in entry), entry
            # No value/baseline/quality/severity can hide in a 3-tuple.
            assert entry[2] in {"metric", "summary", "intervention", "novelty", "other"}

    def test_catalog_type_classifies_sections(self) -> None:
        payload = _payload(_bundle(
            _mixed_bundle().items,
            interventions=(_intervention("nudge"),),
            novelty=("新应用模式: tiktok.exe",),
        ))
        by_type: dict[str, list[str]] = {}
        for entry_id, _label, entry_type in _catalog_rows(payload):
            by_type.setdefault(entry_type, []).append(entry_id)
        assert _canonical_id("focus_score") in by_type["metric"]
        assert "summary.duration_min" in by_type["summary"]
        assert "intervention.latest_type" in by_type["intervention"]
        assert by_type["novelty"] == ["novelty.flags"]

    def test_catalog_is_complete_citation_namespace_in_both_modes(self) -> None:
        for bundle in (_mixed_bundle(), _production_bundle(), _stable_heavy_bundle()):
            expected = evidence_catalog_ids(build_evidence_catalog(bundle))
            assert set(_catalog_ids(_payload(bundle))) == expected
            assert set(_catalog_ids(_payload(bundle, compressed=False))) == expected

    def test_catalog_label_matches_human_readable(self) -> None:
        bundle = _mixed_bundle()
        payload = _payload(bundle)
        labels = {entry_id: label for entry_id, label, _type in _catalog_rows(payload)}
        for item in bundle.items:
            assert labels[_canonical_id(item.metric)] == item.human_readable


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Actionability: collapse the stable, keep the actionable in full
# ═══════════════════════════════════════════════════════════════════════════════


class TestActionabilityCollapse:
    """Stable items are summarised; anomalous/deviating items stay in full."""

    def test_stable_metrics_collapse_into_summary_with_count_and_mean(self) -> None:
        payload = _payload(_mixed_bundle())
        summary = payload["stable_summary"]
        assert list(summary["metrics"]) == list(_MIXED_STABLE)
        assert summary["collapsed_items"] == len(_MIXED_STABLE)
        assert summary["collapsed_metrics"] == len(_MIXED_STABLE)
        assert summary["total_metrics"] == len(_mixed_bundle().items)

        by_metric = {item.metric: item for item in _mixed_bundle().items}
        for metric in _MIXED_STABLE:
            entry = summary["metrics"][metric]
            assert entry["count"] == 1
            item = by_metric[metric]
            if isinstance(item.value, (int, float)):
                assert entry["mean"] == item.value
            else:
                assert entry["value"] == item.value

    def test_non_numeric_stable_metric_keeps_representative_value(self) -> None:
        payload = _payload(_mixed_bundle())
        entry = payload["stable_summary"]["metrics"]["top_apps"]
        assert entry == {"count": 1, "value": "Code.exe, chrome.exe, wechat.exe"}

    def test_anomalous_and_deviating_items_stay_in_evidence_as_full_rows(self) -> None:
        bundle = _mixed_bundle()
        payload = _payload(bundle)
        assert [row["metric"] for row in payload["evidence"]] == list(_MIXED_ACTIONABLE)
        assert set(payload["evidence"][0]) == {
            "metric", "severity", "confidence", "human_readable", "value", "baseline",
        }

    def test_deviating_info_row_keeps_value_and_baseline(self) -> None:
        """An ``info`` item kept for its baseline deviation must carry the data.

        Regression guard: reusing the legacy severity gate for kept rows would
        drop both numbers, so the item would be neither summarised (it is not
        in ``stable_summary``) nor visible as a deviation.
        """
        payload = _payload(_mixed_bundle())
        deviation = next(
            row for row in payload["evidence"] if row["metric"] == "behavior_deviation"
        )
        assert deviation["severity"] == "info"
        assert deviation["value"] == 1.8
        assert deviation["baseline"] == 0.0
        assert "behavior_deviation" not in (payload.get("stable_summary") or {}).get("metrics", {})

    def test_stable_heavy_window_splits_as_designed(self) -> None:
        bundle = _stable_heavy_bundle()
        payload = _payload(bundle)
        assert [row["metric"] for row in payload["evidence"]] == list(_STABLE_HEAVY_ACTIONABLE)
        stable = payload["stable_summary"]["metrics"]
        assert len(stable) == len(bundle.items) - len(_STABLE_HEAVY_ACTIONABLE)
        for metric in _STABLE_HEAVY_ACTIONABLE:
            assert metric not in stable
        # A deviating info row (idle_ratio, 0.31 vs 0.18) keeps its numbers.
        idle = next(row for row in payload["evidence"] if row["metric"] == "idle_ratio")
        assert (idle["value"], idle["baseline"]) == (0.31, 0.18)

    def test_task_breakdown_keeps_task_metrics_in_full(self) -> None:
        """A ``task_breakdown`` intervention retains the ``task_*`` feature rows."""
        bundle = _stable_heavy_bundle("task_breakdown")
        payload = _payload(bundle)
        assert [row["metric"] for row in payload["evidence"]] == list(
            _STABLE_HEAVY_TASK_ACTIONABLE
        )
        stable = payload["stable_summary"]["metrics"]
        assert {row["metric"] for row in payload["evidence"]}.isdisjoint(stable)

    def test_fully_anomalous_bundle_has_no_stable_summary(self) -> None:
        # Every panel-fixture item is anomalous, so nothing is collapsed and the
        # summary section is omitted rather than paid for empty.
        payload = _payload(_panel_bundle())
        assert len(payload["evidence"]) == len(_panel_bundle().items)
        assert "stable_summary" not in payload

    def test_deviation_tolerance_boundary_is_inclusive(self) -> None:
        """|value - baseline| >= max(floor, ratio*|baseline|) keeps the row.

        Against a 0.0 baseline the 0.05 absolute floor decides: 0.049 stays
        stable, exactly 0.05 counts as deviating.
        """
        below = _payload(_bundle((_item("behavior_deviation", 0.049, baseline=0.0),)))
        above = _payload(_bundle((_item("behavior_deviation", 0.05, baseline=0.0),)))
        assert below["evidence"] == []
        assert "behavior_deviation" in below["stable_summary"]["metrics"]
        assert [row["metric"] for row in above["evidence"]] == ["behavior_deviation"]
        assert above["evidence"][0]["value"] == 0.05

    def test_relative_threshold_applies_when_baseline_is_large(self) -> None:
        """15% of an 8.0 baseline is 1.2, well above the 0.05 floor."""
        stable = _payload(_bundle((_item("switch_rate", 8.9, baseline=8.0),)))
        deviating = _payload(_bundle((_item("switch_rate", 9.3, baseline=8.0),)))
        assert stable["evidence"] == []
        assert [row["metric"] for row in deviating["evidence"]] == ["switch_rate"]

    def test_policy_thresholds_are_configurable(self) -> None:
        bundle = _bundle((_item("switch_rate", 10.0, baseline=8.0),))
        strict = EvidenceCompressionPolicy(baseline_deviation_ratio=0.5)
        relaxed = EvidenceCompressionPolicy(baseline_deviation_ratio=0.1)
        # 2.0 vs thresholds 4.0 (strict) and 0.8 (relaxed).
        assert _payload(bundle, policy=strict)["evidence"] == []
        assert _payload(bundle, policy=relaxed)["evidence"][0]["metric"] == "switch_rate"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Recent-intervention relatedness
# ═══════════════════════════════════════════════════════════════════════════════


class TestInterventionRelatedness:
    """The newest intervention type decides which metric names stay in full."""

    def test_matching_newest_intervention_keeps_item_in_full(self) -> None:
        bundle = _bundle((_task_item(),), interventions=(_intervention("task_breakdown"),))
        payload = _payload(bundle)
        assert [row["metric"] for row in payload["evidence"]] == ["task_type_dominant_ratio"]
        row = payload["evidence"][0]
        assert (row["value"], row["baseline"]) == (0.62, 0.60)

    def test_non_matching_newest_intervention_collapses_the_same_item(self) -> None:
        bundle = _bundle((_task_item(),), interventions=(_intervention("nudge"),))
        payload = _payload(bundle)
        assert payload["evidence"] == []
        assert "task_type_dominant_ratio" in payload["stable_summary"]["metrics"]

    def test_only_the_newest_intervention_counts_by_default(self) -> None:
        older = _intervention("task_breakdown", minutes_ago=120)
        newer = _intervention("nudge", minutes_ago=10)
        # History out of order: the newest is picked by timestamp, not position.
        payload = _payload(_bundle((_task_item(),), interventions=(newer, older)))
        assert payload["evidence"] == []

    def test_wider_intervention_window_includes_older_records(self) -> None:
        older = _intervention("task_breakdown", minutes_ago=120)
        newer = _intervention("nudge", minutes_ago=10)
        policy = EvidenceCompressionPolicy(intervention_window=2)
        payload = _payload(
            _bundle((_task_item(),), interventions=(newer, older)), policy=policy
        )
        assert [row["metric"] for row in payload["evidence"]] == ["task_type_dominant_ratio"]

    def test_zero_intervention_window_disables_relatedness(self) -> None:
        policy = EvidenceCompressionPolicy(intervention_window=0)
        payload = _payload(
            _bundle((_task_item(),), interventions=(_intervention("task_breakdown"),)),
            policy=policy,
        )
        assert payload["evidence"] == []

    def test_short_type_tokens_are_ignored(self) -> None:
        # Tokens shorter than two characters never match a metric name.
        payload = _payload(_bundle((_task_item(),), interventions=(_intervention("t"),)))
        assert payload["evidence"] == []

    def test_is_actionable_item_accepts_precomputed_keywords(self) -> None:
        bundle = _bundle((_task_item(),), interventions=(_intervention("nudge"),))
        item = bundle.items[0]
        assert not is_actionable_item(item, bundle, intervention_keywords=frozenset())
        assert is_actionable_item(item, bundle, intervention_keywords=frozenset({"task"}))

    def test_intervention_relatedness_survives_empty_history(self) -> None:
        payload = _payload(_bundle((_task_item(),)))
        assert payload["evidence"] == []

    def test_stable_metric_limit_is_a_non_lossy_safety_valve(self) -> None:
        """``stable_metric_limit`` demotes the overflow back to full rows."""
        bundle = _bundle((
            _item("switch_rate", 8.04, baseline=8.0),
            _item("start_delay_min", 10.04, baseline=10.0),
            _item("idle_ratio", 0.184, baseline=0.18),
        ))
        policy = EvidenceCompressionPolicy(stable_metric_limit=2)
        payload = _payload(bundle, policy=policy)
        # Two metrics stay summarised, the overflow is enumerated instead of
        # disappearing — and with its value, since no summary carries it.
        assert list(payload["stable_summary"]["metrics"]) == ["switch_rate", "start_delay_min"]
        overflow = [row for row in payload["evidence"] if row["metric"] == "idle_ratio"]
        assert len(overflow) == 1
        assert (overflow[0]["value"], overflow[0]["baseline"]) == (0.184, 0.18)
        assert set(_value_metric_ids(payload)) == {item.metric for item in bundle.items}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Legacy mode (the documented A/B switch)
# ═══════════════════════════════════════════════════════════════════════════════


class TestLegacyMode:
    """``compressed=False`` reproduces the pre-2.4 payload shape."""

    def test_legacy_enumerates_every_item(self) -> None:
        bundle = _mixed_bundle()
        payload = _payload(bundle, compressed=False)
        assert [row["metric"] for row in payload["evidence"]] == [
            item.metric for item in bundle.items
        ]
        assert "stable_summary" not in payload

    def test_legacy_rows_keep_the_severity_value_gate(self) -> None:
        payload = _payload(_mixed_bundle(), compressed=False)
        for row in payload["evidence"]:
            if row["severity"] == "info":
                assert "value" not in row and "baseline" not in row
            else:
                assert "value" in row

    def test_legacy_catalog_repeats_values(self) -> None:
        payload = _payload(_mixed_bundle(), compressed=False)
        assert payload["evidence_catalog"]
        for entry in payload["evidence_catalog"]:
            assert set(entry) == {"id", "label_zh", "value"}, entry

    def test_non_evidence_sections_identical_between_modes(self) -> None:
        bundle = _bundle(
            _mixed_bundle().items,
            interventions=(_intervention("nudge"),),
            novelty=("新应用模式: tiktok.exe",),
        )
        legacy = _payload(bundle, compressed=False)
        compressed = _payload(bundle)
        for section in (
            "window", "behavior_summary", "intervention_history", "novelty_flags",
        ):
            assert legacy[section] == compressed[section], section
        # Section order is stable: compression only rewrites evidence/catalog.
        assert set(compressed) - set(legacy) == {"stable_summary"}
        assert set(legacy) - set(compressed) == set()

    def test_legacy_is_not_reachable_by_default(self) -> None:
        bundle = _mixed_bundle()
        assert to_prompt_json(bundle) == to_prompt_json(bundle, compressed=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Citation recall — the catalog id set must not shrink
# ═══════════════════════════════════════════════════════════════════════════════


class TestCitationRecall:
    """Compression must not remove a single citeable id."""

    @pytest.mark.parametrize(
        "bundle",
        [_mixed_bundle(), _production_bundle(), _stable_heavy_bundle(), _panel_bundle()],
        ids=["mixed", "production", "stable_heavy", "panel"],
    )
    def test_catalog_ids_identical_between_modes(self, bundle: EvidenceBundle) -> None:
        legacy = _catalog_ids(_payload(bundle, compressed=False))
        compressed = _catalog_ids(_payload(bundle))
        assert compressed == legacy
        assert set(compressed) == evidence_catalog_ids(build_evidence_catalog(bundle))

    @pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.scenario_id)
    def test_catalog_ids_identical_across_eval_scenarios(self, scenario: Any) -> None:
        bundle: EvidenceBundle = scenario.bundle
        assert _catalog_ids(_payload(bundle)) == _catalog_ids(
            _payload(bundle, compressed=False)
        )

    def test_panel_fixture_catalog_ids_identical(self) -> None:
        bundle = _panel_bundle()
        assert _catalog_ids(_payload(bundle)) == _catalog_ids(_payload(bundle, compressed=False))

    def test_every_metric_stays_citeable_and_panel_coverage_holds(self) -> None:
        bundle = _stable_heavy_bundle()
        payload = _payload(bundle)
        catalog = set(_catalog_ids(payload))
        for item in bundle.items:
            assert _canonical_id(item.metric) in catalog
        # panel_fastpath reads metric names from rows + summary + catalog ids.
        assert set(_value_metric_ids(payload)) == {item.metric for item in bundle.items}


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Measurement — payload size before/after (printed; run with -s)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMeasuredReduction:
    """Measured character counts; the reduction is a consequence, not a knob.

    Measured 2026-09-25 (this file, ``-s``): the plan's **≥30% target is not
    met**.  Best case is the 31-item stable-heavy window at 21.3%; the current
    production-shaped 9-item window (rule + ML + deviation rows) reaches 7.6%,
    and the anomaly-heavy panel / eval fixtures are flat (≈0%, slightly
    negative for the smallest ones).

    Structural reason: ``evidence_catalog`` — id + Chinese label + type for
    *every* citeable id, uncompressed by design because citation recall must
    not change — is the largest section of the compressed payload (~55% on the
    stable-heavy window) and only the per-item row is actually collapsed
    (≈85c row -> ≈30c summary entry, ≈+3c for the catalog ``type`` field).
    Section-level accounting is printed below so the gap is auditable rather
    than assumed; ``test_plan_target_of_30_percent_is_reported`` is a strict
    xfail so reaching the target later turns into a visible XPASS.
    """

    def test_panel_fixture_is_anomaly_heavy_and_barely_shrinks(self) -> None:
        """Honest bound: this fixture has nothing stable to collapse."""
        bundle = _panel_bundle()
        pct = _report("panel fixture (test_agents_orchestrator._make_bundle)", bundle)
        assert len(_payload(bundle)["evidence"]) == len(bundle.items)
        assert pct > -5.0

    def test_production_shaped_window_is_measured(self) -> None:
        pct = _report("production-shaped window (9 items, 5 stable)", _production_bundle())
        assert pct > 5.0, f"production-shaped payload regressed: {pct:.1f}%"

    def test_stable_heavy_window_is_measured(self) -> None:
        nudge_pct = _report("stable-heavy realistic window / nudge (31 items, 27 stable)",
                            _stable_heavy_bundle())
        task_pct = _report("stable-heavy realistic window / task_breakdown (31 items, 23 stable)",
                           _stable_heavy_bundle("task_breakdown"))
        assert nudge_pct > 15.0, f"stable-heavy payload regressed: {nudge_pct:.1f}%"
        # The intervention-relatedness branch deliberately keeps four extra rows
        # in full, so the same window compresses less — still a real reduction.
        assert task_pct > 10.0, f"intervention-retained window regressed: {task_pct:.1f}%"

    def test_plan_target_of_30_percent_is_met(self) -> None:
        """The plan's ≥30% input-token target, measured on a realistic window.

        Reached by the positional catalog encoding (``[id, label, type]`` rows
        instead of repeating the key names): the catalog was 58% of the payload
        and could not otherwise shrink without losing citeable ids.
        """
        pct = _report("plan target (stable-heavy realistic window)",
                      _stable_heavy_bundle())
        assert pct >= 30.0, f"compression target missed: {pct:.1f}%"

    def test_mixed_fixture_is_measured(self) -> None:
        pct = _report("local mixed fixture (7 items, 5 stable)", _mixed_bundle())
        assert pct > 5.0, f"mixed payload regressed: {pct:.1f}%"

    def test_eval_scenario_suite_aggregate_is_measured(self) -> None:
        before_total = 0
        after_total = 0
        shrunk = 0
        for scenario in ALL_SCENARIOS:
            before, after, _ = _sizes(scenario.bundle)
            before_total += before
            after_total += after
            shrunk += after < before
        pct = (1.0 - after_total / before_total) * 100.0
        print(
            f"\n[evidence-compression] eval suite aggregate ({len(ALL_SCENARIOS)} scenarios): "
            f"before={before_total}c after={after_total}c reduction={pct:.1f}% "
            f"shrunk_in={shrunk}/{len(ALL_SCENARIOS)}"
        )
        # Every scenario is anomaly-heavy (2-5 items, mostly non-info), so the
        # suite only confirms "no regression", not a reduction target.
        assert pct > -5.0

    def test_compression_never_grows_a_stable_heavy_payload(self) -> None:
        for bundle in (_mixed_bundle(), _production_bundle(), _stable_heavy_bundle()):
            before, after, pct = _sizes(bundle)
            assert after < before, f"payload grew: {before} -> {after}"
            assert 0.0 < pct < 100.0
