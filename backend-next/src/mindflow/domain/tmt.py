"""Temporal Motivation Theory (TMT) decomposition from behavioural proxies.

Theory
------
Steel (2007, *Psychological Bulletin* 133(1):65-94) and Steel & König (2006,
*Academy of Management Review* 31(4):889-913) model the strength of a
motivation as::

    Motivation = (Expectancy × Value) / (1 + Impulsiveness × Delay)

* **E — Expectancy**: the subjective probability of succeeding at the task.
* **V — Value**: the subjective reward/utility of completing it.
* **I — Impulsiveness**: sensitivity to immediately available rewards.
* **D — Delay**: the temporal distance to the reward, discounted
  hyperbolically (hence the ``1 + I·D`` denominator — Mazur 1987).

This module does **not** measure any of those constructs.  It maps signals
MindFlow already records onto documented proxies in ``[0, 1]`` and combines
them with the formula above.  The result is an *approximation over behavioural
proxies — not a psychometric instrument and not a clinical assessment*.  Every
component therefore carries its ``source`` string, and the breakdown carries
the mandated ``note`` boundary statement.

Proxy table (all clamped to ``[0, 1]``)
--------------------------------------
=================  ===========================================================
Component          Proxy
=================  ===========================================================
Expectancy (E)     Self-reported focus scores (1-5) from the last 14 days,
                   shrunk toward the 0.5 prior with pseudo-count 3:
                   ``E = (Σ(score-1)/4 + 0.5·3) / (n + 3)``.  No ratings →
                   ``0.5`` with source ``"先验（无近14天自评）"``.
Value (V)          Priority (1 = lowest … 5 = highest) of the most important
                   open task due within 7 days, else of the most important open
                   task, normalised as ``(priority-1)/4``.  When the window
                   also has a productive-app share: ``0.7·priority + 0.3·share``.
                   No open tasks → ``0.5`` with source ``"先验"``.
Impulsiveness (I)  Mean robust z of the deviation summary's switch features
                   (``app_switch_count``, ``domain_switch_count`` and the
                   *negated* ``longest_segment_ratio``), squashed by
                   ``I = 1/(1+e^(-z̄))``.  No such feature present → ``0.5``.
Delay (D)          Hyperbolic distance to the nearest open-task deadline,
                   ``h / (h + 24)``: due now → 0, due in a day → 0.5, due in a
                   week → 0.875.  Overdue clamps to 0.  No deadline → ``0.5``
                   with source ``"无截止时间"``.
=================  ===========================================================

Pure stdlib: no framework, no I/O, no numpy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

#: Boundary text every breakdown carries (NF-S7 safe: no medical vocabulary).
TMT_NOTE: str = "基于行为代理变量的近似，不是心理测量或临床诊断"

#: Neutral prior for any component whose signal is unavailable.
PRIOR: float = 0.5

#: Pseudo-count of the 0.5 prior mixed into the expectancy estimate.  Three
#: virtual observations is enough that a single self-report cannot swing E to
#: an extreme, but few enough that a week of ratings dominates it.
_EXPECTANCY_PRIOR_WEIGHT: float = 3.0

#: Weight of the task-priority term inside V; the rest is the productive share.
_PRIORITY_WEIGHT: float = 0.7

#: Delay half-point: a deadline 24 h away maps to D = 0.5.
_DELAY_HALF_HOURS: float = 24.0

#: Symmetric bound on the logistic exponent.  ``exp(±60)`` is still finite in
#: a double, so an extreme z saturates I to 0 or 1 instead of raising
#: ``OverflowError``.
_LOGISTIC_LIMIT: float = 60.0

BarrierKind = Literal["expectancy", "value", "impulsiveness", "delay"]
MotivationLevel = Literal["low", "medium", "high"]

#: ``motivation`` thresholds.  The all-prior case (every factor at 0.5) gives
#: ``M = 0.5·0.5 / (1 + 0.5·0.5) = 0.2``, which lands in ``medium``: with no
#: signal there is no evidence of a motivation problem, and none against one.
LOW_LEVEL_THRESHOLD: float = 0.15
HIGH_LEVEL_THRESHOLD: float = 0.35

#: Deviation-summary features whose robust z feeds the impulsiveness proxy.
#: ``longest_segment_ratio`` is *inverted*: a long uninterrupted segment means
#: low impulsiveness, so its z enters with a negative sign.
_SWITCH_Z_FEATURES: tuple[str, ...] = ("app_switch_count", "domain_switch_count")
_LONGEST_SEGMENT_FEATURE: str = "longest_segment_ratio"


@dataclass(frozen=True)
class TmtComponent:
    """One TMT factor as a bounded score plus its provenance.

    Attributes:
        score: Normalised factor value in ``[0, 1]``.  For E and V higher is
            better; for I and D higher means more procrastination pressure.
        raw: The unnormalised proxy actually used, or ``None`` for the prior.
        label: Chinese display label.
        source: Human-readable provenance, e.g. ``"focus_session_feedback"``.
        evidence_ids: Evidence-catalog ids the value was computed from.
    """

    score: float
    raw: float | None
    label: str
    source: str
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            msg = f"TmtComponent.score must be in [0, 1], got {self.score}"
            raise ValueError(msg)


@dataclass(frozen=True)
class TmtComponents:
    """The four TMT factors (contract §3 ``components``)."""

    expectancy: TmtComponent
    value: TmtComponent
    impulsiveness: TmtComponent
    delay: TmtComponent

    def as_mapping(self) -> dict[str, TmtComponent]:
        """Return the four components keyed by their contract names."""
        return {
            "expectancy": self.expectancy,
            "value": self.value,
            "impulsiveness": self.impulsiveness,
            "delay": self.delay,
        }


@dataclass(frozen=True)
class TmtBreakdown:
    """Decomposed motivation for one analysis window (contract §3).

    Attributes:
        motivation: ``E·V / (1 + I·D)``.  With every factor in ``[0, 1]`` the
            numerator is in ``[0, 1]`` and the denominator in ``[1, 2]``, so the
            result is already normalised to ``[0, 1]``.
        motivation_level: ``low`` (< 0.15) / ``medium`` (< 0.35) / ``high``.
        dominant_barrier: Factor whose idealisation would raise motivation the
            most — i.e. the largest negative contribution to motivation.  I and
            D are scored as one joint ``I·D`` candidate (see
            :func:`_dominant_barrier`).
        components: The four factors.
        note: Boundary statement (see :data:`TMT_NOTE`).
    """

    motivation: float
    motivation_level: MotivationLevel
    dominant_barrier: BarrierKind
    components: TmtComponents
    note: str = TMT_NOTE

    def __post_init__(self) -> None:
        if not 0.0 <= self.motivation <= 1.0:
            msg = f"TmtBreakdown.motivation must be in [0, 1], got {self.motivation}"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict matching contract §3 (``tmt`` in the API responses)."""
        return {
            "motivation": self.motivation,
            "motivation_level": self.motivation_level,
            "dominant_barrier": self.dominant_barrier,
            "components": {
                key: {
                    "score": component.score,
                    "raw": component.raw,
                    "label": component.label,
                    "source": component.source,
                    "evidence_ids": list(component.evidence_ids),
                }
                for key, component in self.components.as_mapping().items()
            },
            "note": self.note,
        }


@dataclass(frozen=True)
class TmtInputs:
    """Everything :func:`compute_tmt` is allowed to look at.

    All fields are optional: the function degrades each missing signal to the
    neutral prior rather than inventing a value.  Callers populate them from
    data the evidence bundle already has (or from optional repositories).

    Attributes:
        feedback_scores: Self-reported focus scores (1-5) from the last 14 days.
        task_priority: Priority (1 = lowest … 5 = highest) of the task chosen
            for the value proxy — the most important one due within 7 days,
            else the most important open task.
        task_count: Number of open tasks, for the provenance string.
        productive_share: Share of active time in productive app categories,
            in ``[0, 1]``.
        deviation_z_mean: Mean robust z of the deviation summary's switch
            features (already sign-corrected), or ``None`` when absent.
        hours_to_deadline: Hours until the nearest open-task deadline (may be
            negative when overdue).
    """

    feedback_scores: Sequence[float] = ()
    task_priority: float | None = None
    task_count: int = 0
    productive_share: float | None = None
    deviation_z_mean: float | None = None
    hours_to_deadline: float | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# Deviation-summary helper
# ═══════════════════════════════════════════════════════════════════════════════


def deviation_z_mean(deviation: Mapping[str, Any] | None) -> float | None:
    """Mean sign-corrected robust z of the deviation summary's switch features.

    Reads the contract §2 ``deviation`` payload (a plain dict) and uses only
    the features that are actually present: ``app_switch_count`` and
    ``domain_switch_count`` enter positively (more switching → more
    impulsiveness), ``longest_segment_ratio`` negatively (a longer
    uninterrupted segment → less impulsiveness).

    Returns ``None`` when the payload is missing, malformed, or contains none
    of those features — the caller then falls back to the neutral prior.
    """
    if not isinstance(deviation, Mapping):
        return None
    features = deviation.get("features")
    if not isinstance(features, list):
        return None

    zs: list[float] = []
    for entry in features:
        if not isinstance(entry, Mapping):
            continue
        name = entry.get("feature")
        robust_z = entry.get("robust_z")
        if not isinstance(robust_z, (int, float)) or isinstance(robust_z, bool):
            continue
        if name in _SWITCH_Z_FEATURES:
            zs.append(float(robust_z))
        elif name == _LONGEST_SEGMENT_FEATURE:
            zs.append(-float(robust_z))
    if not zs:
        return None
    return sum(zs) / len(zs)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-component proxies
# ═══════════════════════════════════════════════════════════════════════════════


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _expectancy(inputs: TmtInputs) -> TmtComponent:
    """Beta-style shrunk success expectation from self-reported focus scores."""
    scores = [
        float(score)
        for score in inputs.feedback_scores
        if isinstance(score, (int, float))
        and not isinstance(score, bool)
        and 1.0 <= float(score) <= 5.0
    ]
    if not scores:
        return TmtComponent(
            score=PRIOR,
            raw=None,
            label="成功预期",
            source="先验（无近14天自评）",
            evidence_ids=("focus.tmt_expectancy",),
        )

    successes = sum((score - 1.0) / 4.0 for score in scores)
    score = _clamp(
        (successes + PRIOR * _EXPECTANCY_PRIOR_WEIGHT)
        / (len(scores) + _EXPECTANCY_PRIOR_WEIGHT)
    )
    return TmtComponent(
        score=score,
        raw=round(sum(scores) / len(scores), 4),
        label="成功预期",
        source=f"focus_session_feedback（近14天 {len(scores)} 次自评）",
        evidence_ids=("focus.tmt_expectancy",),
    )


def _value(inputs: TmtInputs) -> TmtComponent:
    """Task value from open-task priority, blended with the productive share."""
    if inputs.task_priority is None or inputs.task_count <= 0:
        return TmtComponent(
            score=PRIOR,
            raw=None,
            label="任务价值",
            source="先验",
            evidence_ids=("focus.tmt_value",),
        )

    priority_norm = _clamp((float(inputs.task_priority) - 1.0) / 4.0)
    if inputs.productive_share is not None:
        productive = _clamp(float(inputs.productive_share))
        score = _PRIORITY_WEIGHT * priority_norm + (1.0 - _PRIORITY_WEIGHT) * productive
        source = f"tasks.priority + 生产性应用占比（{inputs.task_count} 个未完成任务）"
    else:
        score = priority_norm
        source = f"tasks.priority（{inputs.task_count} 个未完成任务）"

    return TmtComponent(
        score=_clamp(score),
        raw=round(float(inputs.task_priority), 4),
        label="任务价值",
        source=source,
        evidence_ids=("focus.tmt_value",),
    )


def _impulsiveness(inputs: TmtInputs) -> TmtComponent:
    """Logistic squash of the deviation summary's switch-feature z-scores."""
    if inputs.deviation_z_mean is None:
        return TmtComponent(
            score=PRIOR,
            raw=None,
            label="冲动性",
            source="先验",
            evidence_ids=("focus.tmt_impulsiveness",),
        )

    z_mean = float(inputs.deviation_z_mean)
    bounded = max(-_LOGISTIC_LIMIT, min(_LOGISTIC_LIMIT, z_mean))
    score = 1.0 / (1.0 + math.exp(-bounded))
    return TmtComponent(
        score=_clamp(score),
        raw=round(z_mean, 4),
        label="冲动性",
        source="deviation_summary（切换频率/最长分段稳健 z）",
        evidence_ids=("focus.tmt_impulsiveness",),
    )


def _delay(inputs: TmtInputs) -> TmtComponent:
    """Hyperbolic distance to the nearest open-task deadline."""
    if inputs.hours_to_deadline is None:
        return TmtComponent(
            score=PRIOR,
            raw=None,
            label="奖赏延迟",
            source="无截止时间",
            evidence_ids=("focus.tmt_delay",),
        )
    hours = max(float(inputs.hours_to_deadline), 0.0)
    return TmtComponent(
        score=_clamp(hours / (hours + _DELAY_HALF_HOURS)),
        raw=round(hours, 4),
        label="奖赏延迟",
        source="最近未完成任务截止时间",
        evidence_ids=("focus.tmt_delay",),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Combination
# ═══════════════════════════════════════════════════════════════════════════════


def _motivation(components: TmtComponents) -> float:
    """``E·V / (1 + I·D)`` — already in ``[0, 1]`` because all factors are."""
    numerator = components.expectancy.score * components.value.score
    denominator = 1.0 + components.impulsiveness.score * components.delay.score
    return _clamp(numerator / denominator)


def _level(motivation: float) -> MotivationLevel:
    if motivation < LOW_LEVEL_THRESHOLD:
        return "low"
    if motivation >= HIGH_LEVEL_THRESHOLD:
        return "high"
    return "medium"


def _with_component(
    components: TmtComponents,
    key: BarrierKind,
    score: float,
) -> TmtComponents:
    """Copy *components* with one factor's score replaced (ideals are 0 or 1)."""
    mapped = components.as_mapping()
    current = mapped[key]
    mapped[key] = TmtComponent(
        score=score,
        raw=current.raw,
        label=current.label,
        source=current.source,
        evidence_ids=current.evidence_ids,
    )
    return TmtComponents(
        expectancy=mapped["expectancy"],
        value=mapped["value"],
        impulsiveness=mapped["impulsiveness"],
        delay=mapped["delay"],
    )


#: The value each factor would take if it were not a barrier at all.
_IDEAL_SCORES: dict[BarrierKind, float] = {
    "expectancy": 1.0,
    "value": 1.0,
    "impulsiveness": 0.0,
    "delay": 0.0,
}

#: Tolerance for treating I and D as the same size when splitting the joint
#: ``I·D`` gain (a float boundary, not a probability threshold).
_SCORE_EPSILON: float = 1e-9


def _idealise(
    components: TmtComponents,
    keys: Sequence[BarrierKind],
) -> TmtComponents:
    """Copy *components* with every factor in *keys* set to its ideal score."""
    idealised = components
    for key in keys:
        idealised = _with_component(idealised, key, _IDEAL_SCORES[key])
    return idealised


def _dominant_barrier(components: TmtComponents) -> BarrierKind:
    """The factor whose idealisation raises motivation the most.

    ``gain = M(components idealised) - M(components)``.  For E and V the
    idealisation is per factor; for I and D it is the *joint* candidate
    ``I·D → 0``, because ``M = E·V / (1 + I·D)`` is symmetric in the product:
    idealising I alone and idealising D alone yield identical gains, so the
    gain cannot tell the two apart on its own.

    When the joint gain wins, the reported barrier is whichever of I and D is
    the larger value — the more salient factor, and therefore the one the user
    can actually act on.  Scores within ``1e-9`` of each other are genuinely
    ambiguous under a symmetric product and resolve to ``impulsiveness``.

    E and V ties break E then V, so the choice is fully deterministic.
    """
    base = _motivation(components)
    gain_e = _motivation(_idealise(components, ("expectancy",))) - base
    gain_v = _motivation(_idealise(components, ("value",))) - base
    gain_id = (
        _motivation(_idealise(components, ("impulsiveness", "delay"))) - base
    )

    # ``max(gain_E, gain_V, gain_ID) is gain_ID``, evaluated literally.
    if gain_id >= max(gain_e, gain_v):
        i_score = components.impulsiveness.score
        d_score = components.delay.score
        if abs(i_score - d_score) < _SCORE_EPSILON:
            return "impulsiveness"
        return "impulsiveness" if i_score > d_score else "delay"
    return "expectancy" if gain_e >= gain_v else "value"


def compute_tmt(inputs: TmtInputs) -> TmtBreakdown:
    """Decompose motivation into E·V·I·D using behavioural proxies.

    Args:
        inputs: Available behavioural signals; missing ones fall back to the
            neutral prior ``0.5`` with an explicit ``source``.

    Returns:
        A :class:`TmtBreakdown` whose ``motivation`` is in ``[0, 1]`` and whose
        ``note`` states the behavioural-proxy boundary.
    """
    components = TmtComponents(
        expectancy=_expectancy(inputs),
        value=_value(inputs),
        impulsiveness=_impulsiveness(inputs),
        delay=_delay(inputs),
    )
    motivation = _motivation(components)
    return TmtBreakdown(
        motivation=round(motivation, 4),
        motivation_level=_level(motivation),
        dominant_barrier=_dominant_barrier(components),
        components=components,
        note=TMT_NOTE,
    )
