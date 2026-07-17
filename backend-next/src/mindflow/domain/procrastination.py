"""Procrastination type classification: domain model and rule engine.

This module implements L3 fallback of the LLM degradation chain — a deterministic
rule-based classifier for 5 procrastination types (TMT-based typology by Steel 2007
and Rozental & Carlbring 2014). It is also the canonical specification of the
tagging system used throughout the application.

Design constraints (from ADR):
  - Zero framework dependencies (pure stdlib only — no pydantic in domain).
  - All thresholds are constructor-configurable for calibration.
  - rationale is Chinese and never contains "诊断/治疗/患者" (NF-S7).
  - Source is always Literal["rule_engine"] so consumers can distinguish L3 from
    LLM-based assessments.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal


class ProcrastinationType(StrEnum):
    """Five-factor procrastination typology based on TMT (Steel 2007).

    Members are lower-case strings matching the LLM schema field names in
    ProcrastinationAnalysis.procrastination_types.
    """

    TASK_AVERSION = "task_aversion"
    IMPULSIVITY = "impulsivity"
    DECISIONAL = "decisional"
    PERFECTIONISM = "perfectionism"
    EMOTIONAL_REGULATION = "emotional_regulation"


class CBTTechnique(StrEnum):
    """Evidence-based CBT techniques mapped from procrastination types.

    Maps directly to the ``cbt_technique`` field in the LLM output contract
    (see llm-cbt.md §3 ProcrastinationAnalysis), extended with mindfulness
    for emotional regulation support.
    """

    BEHAVIORAL_EXPERIMENT = "behavioral_experiment"
    COGNITIVE_RESTRUCTURING = "cognitive_restructuring"
    STIMULUS_CONTROL = "stimulus_control"
    GOAL_SETTING = "goal_setting"
    GRADED_EXPOSURE = "graded_exposure"
    MINDFULNESS = "mindfulness"


# Mapping from procrastination type to applicable CBT techniques,
# ordered by priority (primary technique first).
TYPE_TO_TECHNIQUES: Final[Mapping[ProcrastinationType, tuple[CBTTechnique, ...]]] = {
    ProcrastinationType.TASK_AVERSION: (
        CBTTechnique.GRADED_EXPOSURE,
        CBTTechnique.BEHAVIORAL_EXPERIMENT,
    ),
    ProcrastinationType.IMPULSIVITY: (CBTTechnique.STIMULUS_CONTROL,),
    ProcrastinationType.DECISIONAL: (CBTTechnique.GOAL_SETTING,),
    ProcrastinationType.PERFECTIONISM: (CBTTechnique.COGNITIVE_RESTRUCTURING,),
    ProcrastinationType.EMOTIONAL_REGULATION: (CBTTechnique.MINDFULNESS,),
}


@dataclass(frozen=True)
class BehaviorSummary:
    """Aggregated behavioral metrics for a single analysis window.

    All fields are pre-computed by the feature extraction layer before entering
    the rule engine. Duration fields use float to accommodate sub-minute windows.
    """

    intended_task: str | None
    duration_min: float
    actual_focus_min: float
    context_switches_per_hour: float
    longest_focus_block_s: float
    social_media_ratio: float
    start_delay_min: float
    keyword_flags: frozenset[str]
    baseline_deviation: float | None


@dataclass(frozen=True)
class ProcrastinationAssessment:
    """Rule engine output: classified types with confidence and CBT recommendation.

    Fields:
        types: 1-3 types sorted by confidence descending (highest first).
        confidence: Per-type confidence in [0, 1].
        recommended_technique: Primary CBT technique from the highest-confidence
            type, or None when no significant procrastination pattern was
            detected (top confidence < NO_SIGNIFICANT_THRESHOLD) — callers must
            not act on a technique in that case.
        rationale: Chinese explanation — never contains "诊断/治疗/患者" (NF-S7).
        source: Always "rule_engine" for this module.
    """

    types: tuple[ProcrastinationType, ...]
    confidence: Mapping[ProcrastinationType, float]
    recommended_technique: CBTTechnique | None
    rationale: str
    source: Literal["rule_engine"]


class RuleEngine:
    """Deterministic rule engine for procrastination type classification.

    Implements the rules defined in 03-requirements.md §3.4 as the L3 fallback
    of the LLM degradation chain. All thresholds are exposed as constructor
    parameters for calibration without subclassing.

    Threshold source:
      - impulsivity: longest_focus_block_s < 300s AND >12 switches/h
        (03 §3.4: "最长连续专注块 < 5 分钟 + 切换 > 12 次/小时")
      - decisional: start_delay_min > 30min AND post-start focus ratio > 0.4
        (03 §3.4: "从启动到开始 > 30 分钟 + 启动后恢复正常")
      - perfectionism: keyword_flags contains "self_criticism" or "redo_pattern"
        (03 §3.4: "含'不够好/重来/失败'回避模式 + 反复重做")
      - emotional_regulation: social_media_ratio > 0.55
        (03 §3.4: "社交媒体 > 55%")
      - task_aversion: catch-all when focus is low but no other type matches
        (03 §3.4: "兜底")
    """

    # When the maximum confidence across all detected types is below this value,
    # the assessment is considered "no significant procrastination pattern".
    NO_SIGNIFICANT_THRESHOLD: Final[float] = 0.2

    def __init__(
        self,
        impulsivity_min_switches: float = 12.0,
        impulsivity_max_focus_block_s: float = 300.0,
        decisional_min_delay_min: float = 30.0,
        decisional_min_focus_ratio: float = 0.4,
        perfectionism_keywords: frozenset[str] = frozenset({"self_criticism", "redo_pattern"}),
        emotional_regulation_min_ratio: float = 0.55,
        task_aversion_max_focus_ratio: float = 0.35,
        task_aversion_min_deviation: float = -0.5,
    ) -> None:
        self._impulsivity_min_switches = impulsivity_min_switches
        self._impulsivity_max_focus_block_s = impulsivity_max_focus_block_s
        self._decisional_min_delay_min = decisional_min_delay_min
        self._decisional_min_focus_ratio = decisional_min_focus_ratio
        self._perfectionism_keywords = perfectionism_keywords
        self._emotional_regulation_min_ratio = emotional_regulation_min_ratio
        self._task_aversion_max_focus_ratio = task_aversion_max_focus_ratio
        self._task_aversion_min_deviation = task_aversion_min_deviation

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(self, summary: BehaviorSummary) -> ProcrastinationAssessment:
        """Classify procrastination types from behavioral metrics.

        Args:
            summary: Aggregated behavioral metrics for the analysis window.

        Returns:
            A ProcrastinationAssessment with 1-3 types sorted by confidence.
            Returns a low-confidence assessment when no significant procrastination
            pattern is detected.

        Raises:
            No exceptions are raised for any input (guaranteed stable).
        """
        confidences: dict[ProcrastinationType, float] = {}

        self._check_impulsivity(summary, confidences)
        self._check_decisional(summary, confidences)
        self._check_perfectionism(summary, confidences)
        self._check_emotional_regulation(summary, confidences)

        if not confidences:
            self._fill_catch_all(summary, confidences)

        top_types = tuple(sorted(confidences, key=lambda t: confidences[t], reverse=True)[:3])
        top_type = top_types[0]
        top_confidence = confidences[top_type]

        # When max confidence is below threshold, it's "no significant pattern":
        # no technique is recommended so callers can't act on a phantom finding.
        if top_confidence < self.NO_SIGNIFICANT_THRESHOLD:
            rationale = (
                "未检测到显著的拖延模式，指标总体正常。"
                "当前行为数据未表现出与已知拖延类型强相关的模式。"
            )
            technique: CBTTechnique | None = None
        else:
            rationale = self._build_rationale(top_types, confidences)
            technique = TYPE_TO_TECHNIQUES[top_type][0]

        return ProcrastinationAssessment(
            types=top_types,
            confidence={t: confidences[t] for t in top_types},
            recommended_technique=technique,
            rationale=rationale,
            source="rule_engine",
        )

    # ------------------------------------------------------------------
    # Rule checks (private)
    # ------------------------------------------------------------------

    def _check_impulsivity(
        self,
        summary: BehaviorSummary,
        confidences: dict[ProcrastinationType, float],
    ) -> None:
        """Impulsivity: longest focus block < threshold AND high switch rate.

        Confidence is a linear map of switches per hour:
          switches == threshold (12) → 0.5
          switches >= 2x threshold (24) → 0.95
        """
        if (
            summary.longest_focus_block_s < self._impulsivity_max_focus_block_s
            and summary.context_switches_per_hour >= self._impulsivity_min_switches
        ):
            conf = self._linear_confidence(
                summary.context_switches_per_hour,
                self._impulsivity_min_switches,
                self._impulsivity_min_switches * 2,
            )
            confidences[ProcrastinationType.IMPULSIVITY] = conf

    def _check_decisional(
        self,
        summary: BehaviorSummary,
        confidences: dict[ProcrastinationType, float],
    ) -> None:
        """Decisional: delay > threshold AND focus recovers after starting.

        The recovery check (focus ratio > threshold) distinguishes decisional
        procrastination from general low focus — if the user works effectively
        once they finally start, the bottleneck is initiation, not sustained
        attention.

        Confidence is a linear map of start delay:
          delay == threshold (30 min) → 0.5
          delay >= 2x threshold (60 min) → 0.95
        """
        if summary.start_delay_min <= self._decisional_min_delay_min:
            return
        focus_ratio = self._safe_ratio(summary.actual_focus_min, summary.duration_min)
        if focus_ratio < self._decisional_min_focus_ratio:
            return
        conf = self._linear_confidence(
            summary.start_delay_min,
            self._decisional_min_delay_min,
            self._decisional_min_delay_min * 2,
        )
        confidences[ProcrastinationType.DECISIONAL] = conf

    def _check_perfectionism(
        self,
        summary: BehaviorSummary,
        confidences: dict[ProcrastinationType, float],
    ) -> None:
        """Perfectionism: keyword flags indicate self-criticism / redo pattern.

        Confidence depends on how many perfectionism keywords fired:
          1 keyword → 0.6
          2+ keywords → 0.85
        """
        matched = summary.keyword_flags & self._perfectionism_keywords
        if not matched:
            return
        conf = 0.85 if len(matched) >= 2 else 0.6
        confidences[ProcrastinationType.PERFECTIONISM] = conf

    def _check_emotional_regulation(
        self,
        summary: BehaviorSummary,
        confidences: dict[ProcrastinationType, float],
    ) -> None:
        """Emotional regulation: high social media ratio.

        Confidence is a linear map of social media ratio:
          ratio == threshold (0.55) → 0.5
          ratio >= 0.80 → 0.95
        """
        if summary.social_media_ratio <= self._emotional_regulation_min_ratio:
            return
        conf = self._linear_confidence(
            summary.social_media_ratio,
            self._emotional_regulation_min_ratio,
            0.80,
        )
        confidences[ProcrastinationType.EMOTIONAL_REGULATION] = conf

    def _fill_catch_all(
        self,
        summary: BehaviorSummary,
        confidences: dict[ProcrastinationType, float],
    ) -> None:
        """Catch-all: task_aversion when focus is low, no-significant otherwise.

        "Low focus" means:
          - actual_focus ratio < 0.35 of total duration, OR
          - baseline deviation is significantly negative (< -0.5)

        When focus is not particularly low and no other rule fired, return a
        low-confidence assessment (no significant procrastination).
        """
        focus_ratio = self._safe_ratio(summary.actual_focus_min, summary.duration_min)
        is_low_focus = focus_ratio < self._task_aversion_max_focus_ratio
        has_negative_deviation = (
            summary.baseline_deviation is not None
            and summary.baseline_deviation < self._task_aversion_min_deviation
        )

        if is_low_focus or has_negative_deviation:
            # Low focus with no specific pattern → task_aversion
            conf = max(0.4, 0.7 - (focus_ratio / self._task_aversion_max_focus_ratio) * 0.3)
            confidences[ProcrastinationType.TASK_AVERSION] = conf
        else:
            # No significant pattern — pick most "ambient" type at low confidence.
            # Impulsivity is chosen arbitrarily because it has the lowest false-positive
            # cost on the "no signal" path; callers should check confidence < 0.2.
            confidences[ProcrastinationType.IMPULSIVITY] = 0.15

    # ------------------------------------------------------------------
    # Confidence mapping helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _linear_confidence(value: float, threshold: float, saturation: float) -> float:
        """Linearly map a continuous value to [0.5, 0.95].

        Args:
            value: The observed metric value. Values below *threshold* are
                clamped up to it (guard for direct callers; rule checks
                already enforce the trigger condition).
            threshold: Value at which confidence = 0.5 (minimum trigger level).
            saturation: Value at which confidence plateaus at 0.95.

        Returns:
            Confidence in [0.5, 0.95].
        """
        if value >= saturation:
            return 0.95
        # Guard against value < threshold (callers already check this, but
        # handle gracefully to guarantee no exceptions).
        effective = max(value, threshold)
        return 0.5 + (effective - threshold) / (saturation - threshold) * 0.45

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        """Compute numerator/denominator safely, returning 0 for zero-denominator."""
        if denominator <= 0:
            return 0.0
        return numerator / denominator

    @staticmethod
    def _build_rationale(
        types: tuple[ProcrastinationType, ...],
        confidences: dict[ProcrastinationType, float],
    ) -> str:
        """Build a Chinese rationale from detected types and their confidences."""
        type_labels: dict[ProcrastinationType, str] = {
            ProcrastinationType.TASK_AVERSION: "任务畏惧型",
            ProcrastinationType.IMPULSIVITY: "冲动分心型",
            ProcrastinationType.DECISIONAL: "决策困难型",
            ProcrastinationType.PERFECTIONISM: "完美主义型",
            ProcrastinationType.EMOTIONAL_REGULATION: "情绪调节型",
        }
        parts: list[str] = []
        for t in types:
            label = type_labels.get(t, str(t))
            conf = confidences[t]
            parts.append(f"{label}(置信度{conf:.0%})")
        return "检测到拖延模式：" + "，".join(parts) + "。"
