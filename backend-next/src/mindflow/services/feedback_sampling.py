"""Active feedback sampling — ask for a label only when it carries information.

Phase 3.3 of the optimisation plan replaces indiscriminate feedback requests with
*targeted* ones.  A window is worth asking about only when the system is actually
uncertain or internally inconsistent:

  * the model's probability sits near the decision boundary;
  * the rule engine and the ML model disagree;
  * the panel's experts disagreed beyond a threshold;
  * the observed task context contradicts the behaviour pattern;
  * the recent data has drifted away from what the model was trained on.

Every decision carries the trigger reasons, so the value of active sampling can
be evaluated per feedback ("does a requested label carry more information than a
randomly collected one?") instead of being assumed.

This module is a pure decision function: it never calls a model, never writes to
the database, and never invents a label.  Callers (scheduler / UI) decide how to
surface the request; the reasons travel with it.

Design constraint: pure stdlib, mirroring ``agents/claims.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

# Trigger identifiers (stable strings — they are recorded per feedback request).
TRIGGER_MODEL_UNCERTAIN = "model_uncertain"
TRIGGER_RULE_ML_CONFLICT = "rule_ml_conflict"
TRIGGER_PANEL_DISAGREEMENT = "panel_disagreement"
TRIGGER_TASK_CONTEXT_MISMATCH = "task_context_mismatch"
TRIGGER_DATA_DRIFT = "data_drift"

ALL_TRIGGERS: tuple[str, ...] = (
    TRIGGER_MODEL_UNCERTAIN,
    TRIGGER_RULE_ML_CONFLICT,
    TRIGGER_PANEL_DISAGREEMENT,
    TRIGGER_TASK_CONTEXT_MISMATCH,
    TRIGGER_DATA_DRIFT,
)


@dataclass(frozen=True)
class FeedbackSamplingConfig:
    """Thresholds that decide whether a label is worth asking for.

    Attributes:
        uncertain_band_half_width: Probability band around 0.5 treated as
            "the model does not know" (0.15 → [0.35, 0.65]).
        rule_ml_min_confidence: Both the rule engine and the ML model must be at
            least this confident before "they disagree" is meaningful; below it
            the window is merely uncertain, not conflicting.
        panel_disagreement_threshold: Panel agreement strength below which the
            experts failed to agree (1.0 = perfect agreement).
        drift_psi_threshold: Population-stability index above which the recent
            data is considered drifted.
        max_reasons: Safety valve — never return more than this many reasons.
    """

    uncertain_band_half_width: float = 0.15
    rule_ml_min_confidence: float = 0.6
    panel_disagreement_threshold: float = 0.5
    drift_psi_threshold: float = 0.25
    max_reasons: int = len(ALL_TRIGGERS)


DEFAULT_FEEDBACK_SAMPLING_CONFIG = FeedbackSamplingConfig()

#: Tolerance for inclusive band comparisons (see ``_uncertain``).
_BAND_EPSILON = 1e-9


@dataclass(frozen=True)
class FeedbackSignals:
    """Everything the sampling decision needs, all optional.

    ``None`` means "signal unavailable" and never triggers on its own: an
    unknown value must not fabricate a feedback request.
    """

    model_probability: float | None = None
    model_top_type: str | None = None
    rule_top_type: str | None = None
    rule_confidence: float | None = None
    ml_confidence: float | None = None
    panel_agreement: float | None = None
    task_context_consistent: bool | None = None
    drift_psi: float | None = None


@dataclass(frozen=True)
class FeedbackRequestDecision:
    """Whether to ask the user for a label, and why."""

    should_request: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""

    @property
    def reason_set(self) -> frozenset[str]:
        return frozenset(self.reasons)


def _uncertain(signals: FeedbackSignals, config: FeedbackSamplingConfig) -> bool:
    probability = signals.model_probability
    if probability is None:
        return False
    half = config.uncertain_band_half_width
    # The band is inclusive; the epsilon keeps a boundary value such as 0.35
    # inside the band despite binary float representation (0.35 - 0.5 =
    # -0.15000000000000002).
    return abs(float(probability) - 0.5) <= half + _BAND_EPSILON


def _rule_ml_conflict(signals: FeedbackSignals, config: FeedbackSamplingConfig) -> bool:
    rule_type = signals.rule_top_type
    model_type = signals.model_top_type
    if not rule_type or not model_type or rule_type == model_type:
        return False
    rule_conf = signals.rule_confidence
    ml_conf = signals.ml_confidence
    if rule_conf is None or ml_conf is None:
        # A type mismatch without confidences is a weak signal: only treat it as
        # a conflict when the model itself is not also uncertain.
        return not _uncertain(signals, config)
    return (
        float(rule_conf) >= config.rule_ml_min_confidence
        and float(ml_conf) >= config.rule_ml_min_confidence
    )


def _panel_disagreement(signals: FeedbackSignals, config: FeedbackSamplingConfig) -> bool:
    agreement = signals.panel_agreement
    if agreement is None:
        return False
    return float(agreement) < config.panel_disagreement_threshold


def _task_context_mismatch(signals: FeedbackSignals) -> bool:
    return signals.task_context_consistent is False


def _data_drift(signals: FeedbackSignals, config: FeedbackSamplingConfig) -> bool:
    psi = signals.drift_psi
    if psi is None:
        return False
    return float(psi) > config.drift_psi_threshold


def decide_feedback_request(
    signals: FeedbackSignals,
    config: FeedbackSamplingConfig = DEFAULT_FEEDBACK_SAMPLING_CONFIG,
) -> FeedbackRequestDecision:
    """Decide whether this window deserves a feedback request.

    The five triggers are evaluated independently and all matching reasons are
    returned (capped by ``config.max_reasons``), so a single request can be
    explained as "uncertain AND drifted" rather than collapsing into one label.
    """
    reasons: list[str] = []
    if _uncertain(signals, config):
        reasons.append(TRIGGER_MODEL_UNCERTAIN)
    if _rule_ml_conflict(signals, config):
        reasons.append(TRIGGER_RULE_ML_CONFLICT)
    if _panel_disagreement(signals, config):
        reasons.append(TRIGGER_PANEL_DISAGREEMENT)
    if _task_context_mismatch(signals):
        reasons.append(TRIGGER_TASK_CONTEXT_MISMATCH)
    if _data_drift(signals, config):
        reasons.append(TRIGGER_DATA_DRIFT)

    reasons = reasons[: config.max_reasons]
    if not reasons:
        return FeedbackRequestDecision(
            should_request=False,
            reasons=(),
            description="模型与规则一致且证据充分，主动询问无额外信息增益",
        )

    return FeedbackRequestDecision(
        should_request=True,
        reasons=tuple(reasons),
        description="触发主动标注：" + "、".join(reasons),
    )


def information_gain_proxy(
    decisions: Sequence[FeedbackRequestDecision],
    baseline_uncertainty: float,
) -> dict[str, float]:
    """Report how targeted the sampling was, without waiting for labels.

    The plan asks to evaluate whether active sampling beats indiscriminate
    collection per feedback.  Until enough labels exist, the honest proxy is the
    *hit rate*: the share of requested windows that were actually ambiguous, and
    the request rate itself.

    Args:
        decisions: One decision per candidate window.
        baseline_uncertainty: Share of *all* windows that are ambiguous (the
            value indiscriminate sampling would achieve).

    Returns:
        ``request_rate``, ``triggered_uncertainty_rate`` and
        ``uncertainty_lift`` (triggered rate divided by the baseline).
    """
    total = len(decisions) or 1
    requested = [d for d in decisions if d.should_request]
    uncertain = [
        d for d in requested if TRIGGER_MODEL_UNCERTAIN in d.reason_set
    ]
    triggered_rate = len(uncertain) / len(requested) if requested else 0.0
    baseline = float(baseline_uncertainty)
    return {
        "request_rate": round(len(requested) / total, 4),
        "triggered_uncertainty_rate": round(triggered_rate, 4),
        "uncertainty_lift": round(triggered_rate / baseline, 4) if baseline > 0 else 0.0,
    }


def trigger_counts(
    decisions: Sequence[FeedbackRequestDecision],
) -> dict[str, int]:
    """Count how often each trigger fired (for the sampling report)."""
    counts = dict.fromkeys(ALL_TRIGGERS, 0)
    for decision in decisions:
        for reason in decision.reasons:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def signals_from_outputs(
    *,
    prediction: Mapping[str, object] | None = None,
    rule_assessment: Mapping[str, object] | None = None,
    panel_agreement: float | None = None,
    task_context_consistent: bool | None = None,
    drift_psi: float | None = None,
) -> FeedbackSignals:
    """Assemble :class:`FeedbackSignals` from the services' own outputs.

    ``prediction`` is a ``FocusPrediction``-shaped mapping and
    ``rule_assessment`` a rule-engine assessment mapping; both are read
    defensively so a partial payload yields "signal unavailable" rather than an
    exception.
    """
    probability: float | None = None
    model_type: str | None = None
    ml_confidence: float | None = None
    if prediction:
        raw_probability = prediction.get("focus_probability", prediction.get("probability"))
        if isinstance(raw_probability, (int, float)):
            probability = float(raw_probability)
        raw_type = prediction.get("top_type") or prediction.get("predicted_type")
        if raw_type:
            model_type = str(raw_type)
        raw_conf = prediction.get("confidence")
        if isinstance(raw_conf, (int, float)):
            ml_confidence = float(raw_conf)

    rule_type: str | None = None
    rule_confidence: float | None = None
    if rule_assessment:
        types = rule_assessment.get("procrastination_types") or ()
        if isinstance(types, (list, tuple)) and types:
            rule_type = str(types[0])
        confidence = rule_assessment.get("type_confidence") or {}
        if isinstance(confidence, Mapping) and rule_type is not None:
            value = confidence.get(rule_type)
            if isinstance(value, (int, float)):
                rule_confidence = float(value)

    return FeedbackSignals(
        model_probability=probability,
        model_top_type=model_type,
        rule_top_type=rule_type,
        rule_confidence=rule_confidence,
        ml_confidence=ml_confidence,
        panel_agreement=panel_agreement,
        task_context_consistent=task_context_consistent,
        drift_psi=drift_psi,
    )


__all__ = [
    "ALL_TRIGGERS",
    "DEFAULT_FEEDBACK_SAMPLING_CONFIG",
    "TRIGGER_DATA_DRIFT",
    "TRIGGER_MODEL_UNCERTAIN",
    "TRIGGER_PANEL_DISAGREEMENT",
    "TRIGGER_RULE_ML_CONFLICT",
    "TRIGGER_TASK_CONTEXT_MISMATCH",
    "FeedbackRequestDecision",
    "FeedbackSamplingConfig",
    "FeedbackSignals",
    "decide_feedback_request",
    "information_gain_proxy",
    "signals_from_outputs",
    "trigger_counts",
]
