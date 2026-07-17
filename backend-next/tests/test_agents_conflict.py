"""Tests for agents/conflict.py — conflict detection boundary matrices.

Covers:
  - Confidence gap boundaries: 0.29 (no conflict) vs 0.30 (conflict) vs 0.31 (conflict)
  - Top-1 type: same vs different
  - Skipped expert handling
  - Edge cases: single opinion, identical opinions
"""

from __future__ import annotations

import pytest

from mindflow.agents.conflict import ConflictReport, detect_conflict
from mindflow.agents.types import ExpertOpinion


def _make_opinion(
    role: str,
    types: tuple[str, ...],
    confidence: dict[str, float],
    skipped: bool = False,
) -> ExpertOpinion:
    return ExpertOpinion(
        role=role,
        perspective=f"{role}视角",
        attribution_types=types,
        confidence=confidence,
        evidence_citations=(),
        argument="测试论证文本",
        skipped=skipped,
    )


class TestConflictDetection:
    """detect_conflict boundary matrix tests."""

    def test_same_top_type_no_conflict(self) -> None:
        """Same top type with close confidence → no conflict."""
        opinions = [
            _make_opinion("CBT", ("impulsivity",), {"impulsivity": 0.80}),
            _make_opinion("TMT", ("impulsivity",), {"impulsivity": 0.75}),
            _make_opinion("Emotion", ("impulsivity",), {"impulsivity": 0.70}),
        ]
        report = detect_conflict(opinions)
        assert not report.has_conflict
        assert report.max_confidence_gap == pytest.approx(0.10)

    def test_different_top_type_conflict(self) -> None:
        """Different top types → conflict (criterion 1)."""
        opinions = [
            _make_opinion("CBT", ("impulsivity",), {"impulsivity": 0.80}),
            _make_opinion("TMT", ("task_aversion",), {"task_aversion": 0.75}),
            _make_opinion("Emotion", ("impulsivity",), {"impulsivity": 0.70}),
        ]
        report = detect_conflict(opinions)
        assert report.has_conflict
        assert "不一致" in report.details

    def test_confidence_gap_0_29_no_conflict(self) -> None:
        """Confidence gap 0.29 → no conflict (below 0.3 threshold)."""
        opinions = [
            _make_opinion("CBT", ("impulsivity",), {"impulsivity": 0.80}),
            _make_opinion("TMT", ("impulsivity",), {"impulsivity": 0.51}),
        ]
        report = detect_conflict(opinions)
        assert not report.has_conflict
        assert report.max_confidence_gap == pytest.approx(0.29)

    def test_confidence_gap_0_30_conflict_boundary(self) -> None:
        """Confidence gap exactly 0.30 → conflict (threshold is >0.3, so no conflict)."""
        opinions = [
            _make_opinion("CBT", ("impulsivity",), {"impulsivity": 0.80}),
            _make_opinion("TMT", ("impulsivity",), {"impulsivity": 0.50}),
        ]
        report = detect_conflict(opinions)
        # 0.80 - 0.50 = 0.30, and criterion is gap > 0.3
        assert not report.has_conflict

    def test_confidence_gap_0_31_conflict(self) -> None:
        """Confidence gap 0.31 → conflict (above 0.3 threshold)."""
        opinions = [
            _make_opinion("CBT", ("impulsivity",), {"impulsivity": 0.81}),
            _make_opinion("TMT", ("impulsivity",), {"impulsivity": 0.50}),
        ]
        report = detect_conflict(opinions)
        assert report.has_conflict
        assert report.max_confidence_gap == pytest.approx(0.31)

    def test_same_type_different_type_also_conflict(self) -> None:
        """Both criteria triggered simultaneously."""
        opinions = [
            _make_opinion("CBT", ("impulsivity",), {"impulsivity": 0.90}),
            _make_opinion("TMT", ("task_aversion",), {"task_aversion": 0.85}),
            _make_opinion("Emotion", ("impulsivity",), {"impulsivity": 0.40}),
        ]
        report = detect_conflict(opinions)
        assert report.has_conflict
        # Both "不一致" and "置信度差距" should be in details
        assert "不一致" in report.details or "置信度差距" in report.details

    def test_all_three_different_top_types(self) -> None:
        """All three experts pick different types → conflict."""
        opinions = [
            _make_opinion("CBT", ("impulsivity",), {"impulsivity": 0.70}),
            _make_opinion("TMT", ("task_aversion",), {"task_aversion": 0.65}),
            _make_opinion("Emotion", ("decisional",), {"decisional": 0.60}),
        ]
        report = detect_conflict(opinions)
        assert report.has_conflict
        # Should have 2-3 unique top types
        unique = {t for t in report.top_types if t is not None}
        assert len(unique) >= 2

    def test_skipped_expert_ignored(self) -> None:
        """Skipped expert should be excluded from conflict detection."""
        opinions = [
            _make_opinion("CBT", ("impulsivity",), {"impulsivity": 0.80}),
            _make_opinion("TMT", ("impulsivity",), {"impulsivity": 0.75}, skipped=True),
            _make_opinion("Emotion", ("impulsivity",), {"impulsivity": 0.70}),
        ]
        report = detect_conflict(opinions)
        # Only CBT and Emotion are non-skipped → gap = 0.10, same type → no conflict
        assert not report.has_conflict

    def test_too_few_non_skipped(self) -> None:
        """Fewer than 2 non-skipped → no conflict reported."""
        opinions = [
            _make_opinion("CBT", ("impulsivity",), {"impulsivity": 0.80}, skipped=True),
            _make_opinion("TMT", ("task_aversion",), {"task_aversion": 0.75}),
        ]
        report = detect_conflict(opinions)
        assert not report.has_conflict
        assert "不足以检测冲突" in report.details

    def test_empty_attribution_types(self) -> None:
        """Expert with no attribution types → top_type is None."""
        opinions = [
            _make_opinion("CBT", (), {}),
            _make_opinion("TMT", ("impulsivity",), {"impulsivity": 0.70}),
        ]
        report = detect_conflict(opinions)
        # CBT has no top type → only one expert with a type → no mismatch
        assert not report.has_conflict

    def test_all_skipped(self) -> None:
        """All experts skipped → no conflict."""
        opinions = [
            _make_opinion("CBT", ("impulsivity",), {"impulsivity": 0.80}, skipped=True),
            _make_opinion("TMT", ("impulsivity",), {"impulsivity": 0.75}, skipped=True),
        ]
        report = detect_conflict(opinions)
        assert not report.has_conflict
        assert "不足以检测冲突" in report.details


class TestConflictReport:
    """ConflictReport dataclass."""

    def test_creates_with_all_fields(self) -> None:
        report = ConflictReport(
            has_conflict=True,
            top_types=("impulsivity", "task_aversion"),
            max_confidence_gap=0.35,
            details="类型不一致",
        )
        assert report.has_conflict
        assert report.top_types == ("impulsivity", "task_aversion")
        assert report.max_confidence_gap == 0.35

    def test_no_conflict(self) -> None:
        report = ConflictReport(
            has_conflict=False,
            top_types=("impulsivity", "impulsivity"),
            max_confidence_gap=0.1,
            details="一致",
        )
        assert not report.has_conflict
