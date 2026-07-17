"""Tests for EvidenceBundle domain contracts (domain/evidence.py).

Covers:
  - Frozen dataclass immutability (all three types).
  - Severity validation rejects invalid values.
  - Confidence range validation.
  - ``to_prompt_json()`` does NOT contain window titles or file paths.
  - ``metric_names()`` returns all and only the metrics in the bundle.
  - ``to_prompt_json()`` output is valid JSON.
  - ``InterventionRecord`` construction with None response.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from mindflow.domain.evidence import (
    EvidenceBundle,
    EvidenceItem,
    InterventionRecord,
    metric_names,
    to_prompt_json,
)
from mindflow.domain.procrastination import BehaviorSummary


def _utc(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=UTC)


# ═══════════════════════════════════════════════════════════════════════════════
# EvidenceItem
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvidenceItem:
    """Frozen dataclass construction and validation."""

    def test_construct_minimal(self) -> None:
        """All required fields set correctly."""
        item = EvidenceItem(
            metric="focus_score",
            value=45.2,
            baseline=72.0,
            severity="moderate",
            confidence=0.85,
            source="feature_computation",
            human_readable="专注度评分45.2/100，低于基线72.0",
        )
        assert item.metric == "focus_score"
        assert item.value == 45.2
        assert item.baseline == 72.0
        assert item.severity == "moderate"
        assert item.confidence == 0.85
        assert item.source == "feature_computation"
        assert item.human_readable == "专注度评分45.2/100，低于基线72.0"

    def test_construct_with_none_baseline(self) -> None:
        """Baseline may be None when no baseline is available."""
        item = EvidenceItem(
            metric="focus_score",
            value=45.2,
            baseline=None,
            severity="info",
            confidence=0.9,
            source="feature_computation",
            human_readable="专注度45.2/100",
        )
        assert item.baseline is None

    def test_frozen_immutable(self) -> None:
        """Cannot modify a frozen dataclass."""
        item = EvidenceItem(
            metric="switch_rate",
            value=35.0,
            baseline=15.0,
            severity="moderate",
            confidence=0.8,
            source="feature_computation",
            human_readable="切换频率35次/小时",
        )
        with pytest.raises(AttributeError):
            item.metric = "focus_score"  # type: ignore[misc]

    def test_invalid_severity_rejected(self) -> None:
        """Severity outside Literal["info","mild","moderate","severe"] raises."""
        with pytest.raises(ValueError, match="severity"):
            EvidenceItem(
                metric="focus_score",
                value=50.0,
                baseline=None,
                severity="critical",  # type: ignore[arg-type]
                confidence=0.9,
                source="test",
                human_readable="test",
            )

    def test_severity_case_sensitive(self) -> None:
        """Severity is case-sensitive: only lowercase variants are valid."""
        with pytest.raises(ValueError, match="severity"):
            EvidenceItem(
                metric="focus_score",
                value=50.0,
                baseline=None,
                severity="Mild",  # type: ignore[arg-type]
                confidence=0.9,
                source="test",
                human_readable="test",
            )

    def test_confidence_too_low_rejected(self) -> None:
        """Confidence below 0 raises."""
        with pytest.raises(ValueError, match="Confidence"):
            EvidenceItem(
                metric="focus_score",
                value=50.0,
                baseline=None,
                severity="info",
                confidence=-0.1,
                source="test",
                human_readable="test",
            )

    def test_confidence_too_high_rejected(self) -> None:
        """Confidence above 1 raises."""
        with pytest.raises(ValueError, match="Confidence"):
            EvidenceItem(
                metric="focus_score",
                value=50.0,
                baseline=None,
                severity="info",
                confidence=1.1,
                source="test",
                human_readable="test",
            )

    def test_confidence_boundaries_accepted(self) -> None:
        """Confidence at exactly 0 and 1 are valid."""
        item_low = EvidenceItem(
            metric="focus_score",
            value=50.0,
            baseline=None,
            severity="info",
            confidence=0.0,
            source="test",
            human_readable="test",
        )
        item_high = EvidenceItem(
            metric="focus_score",
            value=50.0,
            baseline=None,
            severity="info",
            confidence=1.0,
            source="test",
            human_readable="test",
        )
        assert item_low.confidence == 0.0
        assert item_high.confidence == 1.0

    def test_string_value_accepted(self) -> None:
        """Value may be a string for categorical metrics."""
        item = EvidenceItem(
            metric="hmm_state",
            value="procrastinating",
            baseline=None,
            severity="moderate",
            confidence=0.7,
            source="hmm",
            human_readable="当前HMM状态: 拖延中",
        )
        assert item.value == "procrastinating"

    def test_all_four_severities(self) -> None:
        """All four severity levels are constructable."""
        for sev in ("info", "mild", "moderate", "severe"):
            item = EvidenceItem(
                metric="test",
                value=0.0,
                baseline=None,
                severity=sev,  # type: ignore[arg-type]
                confidence=0.5,
                source="test",
                human_readable="test",
            )
            assert item.severity == sev


# ═══════════════════════════════════════════════════════════════════════════════
# InterventionRecord
# ═══════════════════════════════════════════════════════════════════════════════


class TestInterventionRecord:
    """InterventionRecord frozen dataclass."""

    def test_construct(self) -> None:
        """Basic construction with all fields."""
        now = _utc("2026-07-18T10:30:00")
        rec = InterventionRecord(
            intervention_type="nudge",
            triggered_at=now,
            user_response="accepted",
            effect_note="用户接受了建议",
        )
        assert rec.intervention_type == "nudge"
        assert rec.triggered_at == now
        assert rec.user_response == "accepted"
        assert rec.effect_note == "用户接受了建议"

    def test_none_response(self) -> None:
        """user_response may be None for unresponded interventions."""
        now = _utc("2026-07-18T10:30:00")
        rec = InterventionRecord(
            intervention_type="task_breakdown",
            triggered_at=now,
            user_response=None,
            effect_note="尚未回应",
        )
        assert rec.user_response is None

    def test_frozen(self) -> None:
        """Cannot modify a frozen InterventionRecord."""
        now = _utc("2026-07-18T10:30:00")
        rec = InterventionRecord(
            intervention_type="nudge",
            triggered_at=now,
            user_response=None,
            effect_note="test",
        )
        with pytest.raises(AttributeError):
            rec.intervention_type = "task_breakdown"  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════════
# EvidenceBundle
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvidenceBundle:
    """EvidenceBundle construction and immutability."""

    NOW = _utc("2026-07-18T12:00:00")
    WINDOW = (_utc("2026-07-18T08:00:00"), _utc("2026-07-18T12:00:00"))

    @staticmethod
    def _sample_summary() -> BehaviorSummary:
        return BehaviorSummary(
            intended_task="写论文",
            duration_min=240.0,
            actual_focus_min=120.0,
            context_switches_per_hour=25.0,
            longest_focus_block_s=600.0,
            social_media_ratio=0.3,
            start_delay_min=15.0,
            keyword_flags=frozenset(),
            baseline_deviation=None,
        )

    @staticmethod
    def _sample_items() -> tuple[EvidenceItem, ...]:
        return (
            EvidenceItem(
                metric="focus_score",
                value=45.2,
                baseline=72.0,
                severity="moderate",
                confidence=0.85,
                source="feature_computation",
                human_readable="专注度评分45.2/100，偏低",
            ),
            EvidenceItem(
                metric="switch_rate",
                value=35.0,
                baseline=15.0,
                severity="moderate",
                confidence=0.8,
                source="feature_computation",
                human_readable="切换频率35次/小时，偏高",
            ),
        )

    def test_construct(self) -> None:
        """Basic construction with all fields."""
        bundle = EvidenceBundle(
            user_id=1,
            window=self.WINDOW,
            items=self._sample_items(),
            behavior_summary=self._sample_summary(),
            intervention_history=(),
            novelty_flags=(),
        )
        assert bundle.user_id == 1
        assert bundle.window == self.WINDOW
        assert len(bundle.items) == 2
        assert bundle.behavior_summary.intended_task == "写论文"
        assert bundle.intervention_history == ()
        assert bundle.novelty_flags == ()

    def test_frozen(self) -> None:
        """Cannot modify a frozen EvidenceBundle."""
        bundle = EvidenceBundle(
            user_id=1,
            window=self.WINDOW,
            items=self._sample_items(),
            behavior_summary=self._sample_summary(),
            intervention_history=(),
            novelty_flags=(),
        )
        with pytest.raises(AttributeError):
            bundle.user_id = 2  # type: ignore[misc]

    def test_with_interventions(self) -> None:
        """Bundle may carry intervention history."""
        now = _utc("2026-07-18T10:30:00")
        rec = InterventionRecord(
            intervention_type="nudge",
            triggered_at=now,
            user_response="accepted",
            effect_note="已接受",
        )
        bundle = EvidenceBundle(
            user_id=1,
            window=self.WINDOW,
            items=self._sample_items(),
            behavior_summary=self._sample_summary(),
            intervention_history=(rec,),
            novelty_flags=(),
        )
        assert len(bundle.intervention_history) == 1
        assert bundle.intervention_history[0].intervention_type == "nudge"

    def test_with_novelty_flags(self) -> None:
        """Bundle may carry novelty flags."""
        bundle = EvidenceBundle(
            user_id=1,
            window=self.WINDOW,
            items=self._sample_items(),
            behavior_summary=self._sample_summary(),
            intervention_history=(),
            novelty_flags=("新应用模式: tiktok.exe",),
        )
        assert bundle.novelty_flags == ("新应用模式: tiktok.exe",)


# ═══════════════════════════════════════════════════════════════════════════════
# to_prompt_json
# ═══════════════════════════════════════════════════════════════════════════════


class TestToPromptJson:
    """Serialisation contract tests."""

    NOW = _utc("2026-07-18T12:00:00")
    WINDOW = (_utc("2026-07-18T08:00:00"), _utc("2026-07-18T12:00:00"))

    def _bundle(
        self,
        items: tuple[EvidenceItem, ...] | None = None,
        interventions: tuple[InterventionRecord, ...] = (),
        novelty: tuple[str, ...] = (),
    ) -> EvidenceBundle:
        return EvidenceBundle(
            user_id=1,
            window=self.WINDOW,
            items=items or (
                EvidenceItem(
                    metric="focus_score",
                    value=45.2,
                    baseline=72.0,
                    severity="moderate",
                    confidence=0.85,
                    source="feature_computation",
                    human_readable="专注度评分45.2/100",
                ),
            ),
            behavior_summary=BehaviorSummary(
                intended_task="写论文",
                duration_min=240.0,
                actual_focus_min=120.0,
                context_switches_per_hour=25.0,
                longest_focus_block_s=600.0,
                social_media_ratio=0.3,
                start_delay_min=15.0,
                keyword_flags=frozenset(),
                baseline_deviation=None,
            ),
            intervention_history=interventions,
            novelty_flags=novelty,
        )

    def test_returns_valid_json(self) -> None:
        """Output is valid JSON that can be parsed."""
        bundle = self._bundle()
        raw = to_prompt_json(bundle)
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_compact_format(self) -> None:
        """JSON uses compact separators (no extra whitespace)."""
        bundle = self._bundle()
        raw = to_prompt_json(bundle)
        # Compact JSON has no spaces after : or ,
        assert ", " not in raw, f"Found spaces in compact JSON: {raw[:100]}"

    def test_contains_human_readable_not_window_titles(self) -> None:
        """human_readable values are present; window titles must NOT be."""
        bundle = self._bundle()
        raw = to_prompt_json(bundle)
        parsed = json.loads(raw)
        for ev in parsed["evidence"]:
            assert "human_readable" in ev
        # Window title patterns that must NOT appear
        forbidden = ["https://", "www.", ".com", ".py", "C:", "D:"]
        for word in forbidden:
            # The only acceptable occurrence is the key name itself
            lower = raw.lower()
            assert word.lower() not in lower, f"Forbidden content '{word}' found in prompt JSON"

    def test_evidence_structure(self) -> None:
        """Each evidence item has the expected structure in JSON output."""
        bundle = self._bundle()
        parsed = json.loads(to_prompt_json(bundle))
        for ev in parsed["evidence"]:
            assert "metric" in ev
            assert "severity" in ev
            assert "confidence" in ev
            assert "human_readable" in ev

    def test_info_severity_excludes_raw_values(self) -> None:
        """Info-level items omit value and baseline in the JSON (token efficiency)."""
        item = EvidenceItem(
            metric="baseline_building",
            value=0.0,
            baseline=None,
            severity="info",
            confidence=1.0,
            source="welford_baseline",
            human_readable="基线尚在建立中",
        )
        bundle = self._bundle(items=(item,))
        parsed = json.loads(to_prompt_json(bundle))
        ev = parsed["evidence"][0]
        assert "value" not in ev
        assert "baseline" not in ev

    def test_window_structure(self) -> None:
        """Window start/end are ISO strings."""
        bundle = self._bundle()
        parsed = json.loads(to_prompt_json(bundle))
        w = parsed["window"]
        assert "start" in w
        assert "end" in w

    def test_interventions_included(self) -> None:
        """Intervention history appears in the JSON."""
        now = _utc("2026-07-18T10:30:00")
        rec = InterventionRecord(
            intervention_type="nudge",
            triggered_at=now,
            user_response="accepted",
            effect_note="已接受",
        )
        bundle = self._bundle(interventions=(rec,))
        parsed = json.loads(to_prompt_json(bundle))
        assert len(parsed["intervention_history"]) == 1
        ih = parsed["intervention_history"][0]
        assert ih["type"] == "nudge"
        assert ih["user_response"] == "accepted"
        assert ih["effect_note"] == "已接受"

    def test_novelty_flags_included(self) -> None:
        """Novelty flags appear in the JSON."""
        bundle = self._bundle(novelty=("新应用模式: tiktok.exe",))
        parsed = json.loads(to_prompt_json(bundle))
        assert parsed["novelty_flags"] == ["新应用模式: tiktok.exe"]

    def test_behavior_summary_included(self) -> None:
        """Behavior summary is included as an object with metrics."""
        bundle = self._bundle()
        parsed = json.loads(to_prompt_json(bundle))
        bs = parsed["behavior_summary"]
        assert bs["duration_min"] == 240.0
        assert bs["actual_focus_min"] == 120.0
        assert bs["context_switches_per_hour"] == 25.0

    def test_behavior_summary_no_titles(self) -> None:
        """Behavior summary MUST NOT include intended_task (may be user-typed)."""
        # The intended_task is set to "写论文" but should not appear in output
        bundle = self._bundle()
        parsed = json.loads(to_prompt_json(bundle))
        bs = parsed["behavior_summary"]
        # intended_task should NOT be in the serialized output (privacy)
        assert "intended_task" not in bs, (
            "intended_task should not be in serialized behavior_summary (privacy)"
        )

    def test_ensure_no_path_or_title_leak(self) -> None:
        """Sanity: no file path patterns anywhere in the serialized JSON."""
        bundle = self._bundle()
        raw = to_prompt_json(bundle)
        lower = raw.lower()
        # Common path markers
        assert "\\" not in raw, "Backslash (path marker) found in prompt JSON"
        assert "/home/" not in lower, "/home/ path marker found"
        assert "c:" not in lower, "C: drive marker found"
        assert "d:" not in lower, "D: drive marker found"

    def test_extra_keys_in_evidence(self) -> None:
        """Evidence items should not leak unexpected keys like 'source' in the LLM-facing JSON."""
        bundle = self._bundle()
        parsed = json.loads(to_prompt_json(bundle))
        for ev in parsed["evidence"]:
            # The output should only contain these keys for non-info items:
            # metric, severity, confidence, human_readable [, value [, baseline]]
            allowed_keys = {
                "metric", "severity", "confidence", "human_readable", "value", "baseline",
            }
            for key in ev:
                assert key in allowed_keys, f"Unexpected key '{key}' in evidence output"

    def test_empty_interventions_omitted_not_null(self) -> None:
        """Empty intervention_history should be an empty list, not null."""
        bundle = self._bundle()
        parsed = json.loads(to_prompt_json(bundle))
        assert parsed["intervention_history"] == []

    def test_empty_novelty_flags(self) -> None:
        """Empty novelty_flags should be an empty list."""
        bundle = self._bundle()
        parsed = json.loads(to_prompt_json(bundle))
        assert parsed["novelty_flags"] == []


# ═══════════════════════════════════════════════════════════════════════════════
# metric_names
# ═══════════════════════════════════════════════════════════════════════════════


class TestMetricNames:
    """metric_names() contract tests."""

    WINDOW = (_utc("2026-07-18T08:00:00"), _utc("2026-07-18T12:00:00"))

    def _bundle(self, items: tuple[EvidenceItem, ...]) -> EvidenceBundle:
        return EvidenceBundle(
            user_id=1,
            window=self.WINDOW,
            items=items,
            behavior_summary=BehaviorSummary(
                intended_task="test",
                duration_min=60.0,
                actual_focus_min=30.0,
                context_switches_per_hour=10.0,
                longest_focus_block_s=300.0,
                social_media_ratio=0.2,
                start_delay_min=5.0,
                keyword_flags=frozenset(),
                baseline_deviation=None,
            ),
            intervention_history=(),
            novelty_flags=(),
        )

    def test_single_item(self) -> None:
        """Single metric returns a frozenset with that metric name."""
        item = EvidenceItem(
            metric="focus_score",
            value=45.0,
            baseline=None,
            severity="info",
            confidence=0.9,
            source="test",
            human_readable="test",
        )
        bundle = self._bundle(items=(item,))
        names = metric_names(bundle)
        assert names == frozenset({"focus_score"})

    def test_multiple_items(self) -> None:
        """Multiple items return all metric names."""
        items = (
            EvidenceItem(
                metric="focus_score", value=45.0, baseline=None,
                severity="info", confidence=0.9, source="test", human_readable="t",
            ),
            EvidenceItem(
                metric="switch_rate", value=30.0, baseline=None,
                severity="info", confidence=0.9, source="test", human_readable="t",
            ),
            EvidenceItem(
                metric="longest_block", value=600.0, baseline=None,
                severity="info", confidence=0.9, source="test", human_readable="t",
            ),
        )
        bundle = self._bundle(items=items)
        names = metric_names(bundle)
        assert names == frozenset({"focus_score", "switch_rate", "longest_block"})

    def test_empty_items(self) -> None:
        """Empty items returns empty frozenset."""
        bundle = self._bundle(items=())
        names = metric_names(bundle)
        assert names == frozenset()

    def test_duplicate_metrics_deduplicated(self) -> None:
        """Duplicate metric names are deduplicated in the frozenset."""
        items = (
            EvidenceItem(
                metric="focus_score", value=45.0, baseline=None,
                severity="info", confidence=0.9, source="test", human_readable="t",
            ),
            EvidenceItem(
                metric="focus_score", value=50.0, baseline=None,
                severity="info", confidence=0.8, source="other", human_readable="t",
            ),
        )
        bundle = self._bundle(items=items)
        names = metric_names(bundle)
        assert names == frozenset({"focus_score"})
        assert len(names) == 1

    def test_returns_frozenset(self) -> None:
        """Return type is frozenset[str], not set."""
        item = EvidenceItem(
            metric="focus_score", value=45.0, baseline=None,
            severity="info", confidence=0.9, source="test", human_readable="t",
        )
        bundle = self._bundle(items=(item,))
        names = metric_names(bundle)
        assert isinstance(names, frozenset)
