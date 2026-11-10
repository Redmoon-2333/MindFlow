"""Thompson-sampling intervention policy (JITAI-style, contract §4).

Theory
------
Just-in-time adaptive interventions (Nahum-Shani et al. 2018, *Annals of
Behavioral Medicine* 52(6):446-462; Klasnja et al. 2015, *Health Psychology*
34(Suppl):1220-1228) tailor the *type* and *timing* of support to the user's
momentary state.  Which support actually helps is rarely known a priori, so
the policy treats it as a multi-armed bandit and learns it online with
Thompson sampling (Thompson 1933; Chapelle & Li 2011, *Advances in Neural
Information Processing Systems* 24; Russo et al. 2018, *Foundations and
Trends in Machine Learning* 11(1):1-96).

Design
------
* **Candidate set is theory-gated.**  The bandit never *invents* an
  intervention: for each procrastination type it may only rank the types
  allowed by the CBT technique mapping (see :data:`ALLOWED_INTERVENTIONS`).
  Exploration therefore stays inside clinically defensible options.
* **Context bucket** = time-of-day band (local time) × predicted state
  (``focused`` / ``uncertain`` / ``distracted`` / ``unknown``).  Beta
  posteriors are kept per ``(intervention_type, bucket)``.
* **Posteriors are derived on the fly** from ``intervention_logs`` — no new
  table, no migration.  Each historical row contributes a fractional
  success/failure to its bucket, weighted by an exponential time decay
  (half-life :data:`HALF_LIFE_DAYS` days) so stale preferences fade.
* **Partial pooling.**  A bucket with fewer than :data:`MIN_BUCKET_OBSERVATIONS`
  observations is too thin to learn from, so it falls back to that
  intervention type's all-bucket counts scaled by
  :data:`POOLING_WEIGHT` (a fixed-effect shrinkage towards the global arm
  performance).
* **Prior** ``Beta(1, 1)`` — uniform and non-committal.  The type's *primary*
  candidate (first in the allowed tuple) gets ``Beta(2, 1)`` so that with no
  data at all the theory's first-line choice is the slight favourite rather
  than an arbitrary tie.  Both are documented constants, not tuning knobs
  hidden in the call site.

This module is pure logic plus a thin async loader; it never calls the LLM,
never broadcasts, and never writes to the database.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, cast

from loguru import logger

from mindflow.domain.intervention import (
    INTERVENTION_TYPE_LABELS,
    InterventionType,
)
from mindflow.domain.procrastination import ProcrastinationType

#: Theory-allowed intervention candidates per procrastination type, primary
#: first.  Justification (CBT technique mapping, ``TYPE_TO_TECHNIQUES``):
#:
#: * ``task_aversion`` — graded exposure / goal setting: break the avoided task
#:   into approachable steps; a plain reminder is the weaker fallback.
#: * ``impulsivity`` — stimulus control: remove the competing cue from the
#:   environment first, prompt second.
#: * ``decisional`` — behavioural experiment: a low-cost prompt lowers the
#:   cost of starting the decision; prioritisation is the alternative.
#: * ``perfectionism`` — goal setting: define "good enough" via step sizing,
#:   then prioritise what actually has to be perfect.
#: * ``emotional_regulation`` — mindfulness first (accept the feeling, then
#:   act); environment optimisation removes the escape route.
ALLOWED_INTERVENTIONS: Mapping[ProcrastinationType, tuple[InterventionType, ...]] = {
    ProcrastinationType.TASK_AVERSION: ("task_breakdown", "nudge"),
    ProcrastinationType.IMPULSIVITY: ("environment_optimization", "nudge"),
    ProcrastinationType.DECISIONAL: ("smart_prioritization", "nudge"),
    ProcrastinationType.PERFECTIONISM: ("task_breakdown", "smart_prioritization"),
    ProcrastinationType.EMOTIONAL_REGULATION: ("nudge", "environment_optimization"),
}

#: Prior pseudo-counts.  A type's primary candidate starts at ``(2, 1)``.
PRIOR_ALPHA: float = 1.0
PRIOR_BETA: float = 1.0
PRIMARY_PRIOR_ALPHA: float = 2.0

#: Time constant of the exponential decay.  A 14-day-old observation counts
#: half of a fresh one.
HALF_LIFE_DAYS: float = 14.0

#: A bucket needs this many decay-weighted observations before its own counts
#: are trusted; below it, the arm's pooled counts are used instead.
MIN_BUCKET_OBSERVATIONS: float = 3.0

#: Weight applied to the pooled all-bucket counts when a bucket is too thin.
POOLING_WEIGHT: float = 0.5

#: Chinese labels for the context bucket's state half.
STATE_LABELS: Mapping[str, str] = {
    "focused": "专注",
    "uncertain": "不确定",
    "distracted": "分心",
    "unknown": "未知",
}

#: Predicted focus probability at or above this is ``focused``.
FOCUSED_THRESHOLD: float = 0.6

#: Predicted focus probability at or below this is ``distracted``.  The band
#: in between is ``uncertain``: the model is not confident enough for the
#: bucket to claim a state it cannot support.
DISTRACTED_THRESHOLD: float = 0.4

StateBand = Literal["focused", "uncertain", "distracted", "unknown"]


# ═══════════════════════════════════════════════════════════════════════════════
# Context bucket
# ═══════════════════════════════════════════════════════════════════════════════


def time_band(local_hour: int) -> str:
    """Map a local hour (0-23) to one of four bands."""
    if 5 <= local_hour < 12:
        return "morning"
    if 12 <= local_hour < 18:
        return "afternoon"
    if 18 <= local_hour < 23:
        return "evening"
    return "night"


def state_band(focus_probability: float | None) -> StateBand:
    """Map a predicted focus probability to a coarse state band.

    ``None`` (no model / no data) is reported honestly as ``unknown`` instead
    of assuming either state.
    """
    if focus_probability is None:
        return "unknown"
    if focus_probability >= FOCUSED_THRESHOLD:
        return "focused"
    if focus_probability <= DISTRACTED_THRESHOLD:
        return "distracted"
    return "uncertain"


def context_bucket(local_hour: int, focus_probability: float | None) -> str:
    """Build the ``"<band>|<state>"`` bucket key used by the posteriors."""
    return f"{time_band(local_hour)}|{state_band(focus_probability)}"


def bucket_label(bucket: str) -> str:
    """Chinese label for a bucket key, e.g. ``afternoon|distracted``."""
    band, _, state = bucket.partition("|")
    band_zh = {
        "morning": "上午",
        "afternoon": "下午",
        "evening": "晚间",
        "night": "深夜",
    }.get(band, band)
    state_zh = STATE_LABELS.get(state, state)
    return f"{band_zh}·{state_zh}"


# ═══════════════════════════════════════════════════════════════════════════════
# Outcome attribution
# ═══════════════════════════════════════════════════════════════════════════════

#: Fractional (success, failure) contribution of one logged intervention.
Reward = tuple[float, float]

#: Reward granted for an ``ignored`` row that the reminder client, not the
#: user, auto-resolved.  It is weak evidence of failure — nobody actively
#: refused — but it is still evidence, so it is not dropped.
AUTO_IGNORED_FAILURE: float = 0.5


def reward_of(
    user_response: str | None,
    feedback_rating: str | None,
    response_source: str | None = None,
) -> Reward | None:
    """Map one logged intervention row to fractional (success, failure) counts.

    The explicit rating wins over the implicit response, because it is the
    only signal the user gave deliberately:

    * ``helpful`` → ``(1, 0)``; ``annoying`` → ``(0, 1)``;
      ``neutral`` → ``(0.5, 0.5)`` — a neutral rating is still a rating, so it
      contributes half a success and half a failure rather than nothing.
    * No rating: ``accepted`` → ``(1, 0)``; ``dismissed`` → ``(0, 1)``;
      ``ignored`` → ``(0, 1)``, or ``(0, 0.5)`` when ``response_source`` is
      ``auto`` (a reminder client timed out on the user's behalf).
    * No rating and no response → ``None``: the row carries no signal and is
      skipped.  Counting silence as failure would punish an arm for a user who
      simply left the window open.
    """
    if feedback_rating == "helpful":
        return (1.0, 0.0)
    if feedback_rating == "annoying":
        return (0.0, 1.0)
    if feedback_rating == "neutral":
        return (0.5, 0.5)

    if user_response == "accepted":
        return (1.0, 0.0)
    if user_response == "dismissed":
        return (0.0, 1.0)
    if user_response == "ignored":
        return (0.0, AUTO_IGNORED_FAILURE) if response_source == "auto" else (0.0, 1.0)
    return None


def _decay_weight(age: timedelta) -> float:
    """Exponential decay for one observation: ``0.5 ** (age / half-life)``."""
    age_days = max(age.total_seconds(), 0.0) / 86_400.0
    return math.pow(0.5, age_days / HALF_LIFE_DAYS)


# ═══════════════════════════════════════════════════════════════════════════════
# Posterior estimation
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ArmPosterior:
    """Beta posterior for one (intervention type, context bucket) arm."""

    intervention_type: InterventionType
    alpha: float
    beta: float
    #: Decay-weighted observations that fed alpha/beta (0.0 at cold start).
    observations: float = 0.0

    @property
    def posterior_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


@dataclass(frozen=True)
class Candidate:
    """One sampled candidate — the contract §4 ``candidates[]`` entry."""

    intervention_type: InterventionType
    label: str
    alpha: float
    beta: float
    posterior_mean: float
    sample: float
    chosen: bool = False


@dataclass(frozen=True)
class Selection:
    """Result of one selection step (the contract §4 ``rationale``)."""

    intervention_type: InterventionType
    selection: Literal["thompson_sampling", "rule"]
    context_bucket: str
    candidates: tuple[Candidate, ...]
    note: str


def _bucket_hour(bucket: str) -> str:
    return bucket.partition("|")[0]


def _row_bucket(row: Mapping[str, Any], *, local_hour: int | None) -> str:
    """Recover the bucket a historical row was served in.

    The bucket is read back from ``context_json.rationale.context_bucket`` when
    the row was logged by this policy.  Legacy rows have no rationale, so the
    bucket is reconstructed from the trigger timestamp — using the *stored*
    context rationale when present, otherwise the time band only (state stays
    ``unknown`` because no prediction was persisted).
    """
    context = row.get("context_json")
    if isinstance(context, dict):
        rationale = context.get("rationale")
        if isinstance(rationale, dict):
            stored = rationale.get("context_bucket")
            if isinstance(stored, str) and "|" in stored:
                return stored
    if local_hour is None:
        return "unknown|unknown"
    return f"{time_band(local_hour)}|unknown"


def _parse_triggered_at(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def posteriors_from_logs(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidates: Sequence[InterventionType],
    bucket: str,
    now: datetime,
    timezone_offset_hours: float = 0.0,
) -> dict[InterventionType, ArmPosterior]:
    """Derive one Beta posterior per candidate arm from intervention logs.

    Args:
        rows: Raw ``intervention_logs`` dicts (from ``query_range``).
        candidates: The theory-allowed intervention types to score.
        bucket: The ``"<band>|<state>"`` bucket being decided for.
        now: Reference time for the decay weight and for reconstructing local
            hours of legacy rows.
        timezone_offset_hours: The business timezone's UTC offset, used only
            to place legacy rows into a time band.

    Returns:
        A mapping from intervention type to its posterior, priors included.
    """
    bucket_counts: dict[InterventionType, list[float]] = {
        arm: [0.0, 0.0] for arm in candidates
    }
    pooled_counts: dict[InterventionType, list[float]] = {
        arm: [0.0, 0.0] for arm in candidates
    }

    for row in rows:
        raw_arm = row.get("intervention_type")
        if not isinstance(raw_arm, str) or raw_arm not in candidates:
            continue
        arm = cast(InterventionType, raw_arm)
        reward = reward_of(
            _as_str(row.get("user_response")),
            _as_str(row.get("feedback_rating")),
            _as_str(row.get("response_source")),
        )
        if reward is None:
            continue
        triggered = _parse_triggered_at(row.get("triggered_at"))
        if triggered is None:
            continue
        weight = _decay_weight(now - triggered)
        success, failure = reward
        pooled_counts[arm][0] += weight * success
        pooled_counts[arm][1] += weight * failure

        row_hour = (
            (triggered + timedelta(hours=timezone_offset_hours)).hour
            if timezone_offset_hours
            else triggered.hour
        )
        if _row_bucket(row, local_hour=row_hour) == bucket:
            bucket_counts[arm][0] += weight * success
            bucket_counts[arm][1] += weight * failure

    posteriors: dict[InterventionType, ArmPosterior] = {}
    for position, arm in enumerate(candidates):
        alpha0 = (
            PRIMARY_PRIOR_ALPHA if position == 0 else PRIOR_ALPHA
        )
        success, failure = bucket_counts[arm]
        observations = success + failure
        if observations < MIN_BUCKET_OBSERVATIONS:
            pooled_success, pooled_failure = pooled_counts[arm]
            success = POOLING_WEIGHT * pooled_success
            failure = POOLING_WEIGHT * pooled_failure
        posteriors[arm] = ArmPosterior(
            intervention_type=arm,
            alpha=alpha0 + success,
            beta=PRIOR_BETA + failure,
            observations=success + failure,
        )
    return posteriors


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def has_evidence(
    posteriors: Mapping[InterventionType, ArmPosterior],
) -> bool:
    """True when at least one arm has been observed at least once.

    Cold start is handled by the rule path: with no observation there is
    nothing to explore *from*, so the theory's primary candidate is the honest
    default rather than a coin flip between uninformed priors.
    """
    return any(posterior.observations > 0.0 for posterior in posteriors.values())


# ═══════════════════════════════════════════════════════════════════════════════
# Selection
# ═══════════════════════════════════════════════════════════════════════════════


def candidate_types_for(
    procrastination_types: Sequence[ProcrastinationType],
) -> tuple[InterventionType, ...]:
    """Union of the allowed candidates for the detected types, primary first.

    The first detected type's tuple leads, so the theory's first-line choice
    for the highest-confidence type stays first.  Duplicates are removed
    preserving order.
    """
    ordered: list[InterventionType] = []
    for ptype in procrastination_types:
        for arm in ALLOWED_INTERVENTIONS.get(ptype, ()):
            if arm not in ordered:
                ordered.append(arm)
    return tuple(ordered)


def rule_selection(
    procrastination_types: Sequence[ProcrastinationType],
    *,
    bucket: str,
    existing: Mapping[InterventionType, ArmPosterior] | None = None,
) -> Selection:
    """Deterministic choice: the primary candidate of the top type.

    Manual triggers with an explicit type must not be re-rolled by the bandit —
    the user asked for a specific kind of help.  The rationale still reports
    the current posteriors so the API payload has the same shape as an
    explored decision.
    """
    candidates = candidate_types_for(procrastination_types)
    if not candidates:
        raise ValueError("rule_selection requires at least one procrastination type")
    chosen = candidates[0]
    posteriors = existing or {}
    built: list[Candidate] = []
    for arm in candidates:
        posterior = posteriors.get(arm) or ArmPosterior(
            intervention_type=arm,
            alpha=PRIOR_ALPHA,
            beta=PRIOR_BETA,
        )
        built.append(
            Candidate(
                intervention_type=arm,
                label=INTERVENTION_TYPE_LABELS[arm],
                alpha=round(posterior.alpha, 4),
                beta=round(posterior.beta, 4),
                posterior_mean=round(posterior.posterior_mean, 4),
                sample=round(posterior.posterior_mean, 4),
                chosen=arm == chosen,
            )
        )
    return Selection(
        intervention_type=chosen,
        selection="rule",
        context_bucket=bucket,
        candidates=tuple(built),
        note="手动触发按拖延类型的主选方案执行，未使用汤普森采样探索",
    )


def thompson_select(
    procrastination_types: Sequence[ProcrastinationType],
    *,
    bucket: str,
    posteriors: Mapping[InterventionType, ArmPosterior],
    rng: random.Random,
) -> Selection:
    """Sample once per allowed arm and return the highest draw.

    Args:
        procrastination_types: Detected types (highest confidence first).
        bucket: The context bucket being decided for.
        posteriors: Posterior per allowed arm (from :func:`posteriors_from_logs`).
        rng: Injectable RNG — pass ``random.Random(seed)`` for determinism.

    Returns:
        The selection, with every candidate's draw recorded for the rationale.
    """
    candidates = candidate_types_for(procrastination_types)
    if not candidates:
        raise ValueError("thompson_select requires at least one procrastination type")

    draws: list[Candidate] = []
    for arm in candidates:
        posterior = posteriors.get(arm) or ArmPosterior(
            intervention_type=arm,
            alpha=PRIOR_ALPHA,
            beta=PRIOR_BETA,
        )
        draw = rng.betavariate(posterior.alpha, posterior.beta)
        draws.append(
            Candidate(
                intervention_type=arm,
                label=INTERVENTION_TYPE_LABELS[arm],
                alpha=round(posterior.alpha, 4),
                beta=round(posterior.beta, 4),
                posterior_mean=round(posterior.posterior_mean, 4),
                sample=round(draw, 4),
            )
        )

    best = max(draws, key=lambda c: c.sample)
    chosen_candidates = tuple(
        Candidate(
            intervention_type=c.intervention_type,
            label=c.label,
            alpha=c.alpha,
            beta=c.beta,
            posterior_mean=c.posterior_mean,
            sample=c.sample,
            chosen=c.intervention_type == best.intervention_type,
        )
        for c in draws
    )
    return Selection(
        intervention_type=best.intervention_type,
        selection="thompson_sampling",
        context_bucket=bucket,
        candidates=chosen_candidates,
        note=(
            f"在{bucket_label(bucket)}的上下文下，对"
            f"{len(chosen_candidates)}个理论允许的干预方案各采样一次并选择最高值"
        ),
    )


#: Chinese labels for the five procrastination types.  Duplicated from
#: ``frontend/src/lib/user-terms.ts`` on purpose: the services layer may not
#: import the agents layer, and the wire value must never reach the UI.
PROCRASTINATION_LABELS_ZH: Mapping[str, str] = {
    "task_aversion": "任务厌恶",
    "impulsivity": "冲动分心",
    "decisional": "决策拖延",
    "perfectionism": "完美主义",
    "emotional_regulation": "情绪调节困难",
}


def _label_for(ptype: object) -> str:
    return PROCRASTINATION_LABELS_ZH.get(str(ptype), str(ptype))


def _is_prior_only(candidate: Candidate) -> bool:
    """True when the candidate's Beta has never been updated by feedback.

    Two prior pairs exist: ``Beta(1, 1)`` for an ordinary arm, and
    ``Beta(2, 1)`` for a type's primary candidate (the cold-start favourite).
    A candidate still sitting on either is a default, not a finding.

    ``rule_selection`` and the fallback inside ``thompson_select`` both build
    unobserved arms with ``(1, 1)`` regardless of position, so the primary
    pair cannot be recognised from position alone — both are checked.
    """
    return (candidate.alpha, candidate.beta) in _PRIOR_PAIRS


_PRIOR_PAIRS: frozenset[tuple[float, float]] = frozenset(
    {
        (round(PRIOR_ALPHA, 4), round(PRIOR_BETA, 4)),
        (round(PRIMARY_PRIOR_ALPHA, 4), round(PRIOR_BETA, 4)),
    }
)


def selection_evidence(
    selection: Selection,
    procrastination_types: Sequence[ProcrastinationType],
) -> tuple[str, ...]:
    """Up to four human-readable Chinese evidence lines for the rationale.

    The lines avoid α/β and the word "后验": a reader deciding whether to trust
    a reminder cannot act on Beta parameters, and naming the machinery invites
    reading a prior as evidence.
    """
    type_labels = "、".join(_label_for(t) for t in procrastination_types[:3])
    lines: list[str] = [
        f"拖延类型：{type_labels or '未判定'}",
        f"情境：{bucket_label(selection.context_bucket)}",
    ]
    chosen = next(
        (c for c in selection.candidates if c.chosen),
        None,
    )
    if chosen is not None:
        if _is_prior_only(chosen):
            # The number is the prior's own mean. For a non-primary arm that is
            # exactly 50%; the primary arm's Beta(2, 1) prior averages 67%, and
            # printing 50% there would be a second wrong claim.
            lines.append(
                f"选择{chosen.label}：还没有你对这类提醒的反馈，"
                f"先按默认估计 {chosen.posterior_mean:.0%}"
            )
        else:
            lines.append(
                f"选择{chosen.label}：根据你过去的反馈，"
                f"估计有用的概率 {chosen.posterior_mean:.0%}"
            )
    for candidate in selection.candidates:
        if chosen is not None and candidate.intervention_type == chosen.intervention_type:
            continue
        lines.append(
            f"备选{candidate.label}：估计有用的概率 {candidate.posterior_mean:.0%}"
        )
        if len(lines) >= 4:
            break
    return tuple(lines[:4])


def rationale_payload(
    selection: Selection,
    procrastination_types: Sequence[ProcrastinationType],
) -> dict[str, Any]:
    """Build the JSON-serializable contract §4 ``rationale`` object."""
    return {
        "procrastination_type": (
            str(procrastination_types[0]) if procrastination_types else None
        ),
        "intervention_type": selection.intervention_type,
        "selection": selection.selection,
        "context_bucket": selection.context_bucket,
        "candidates": [
            {
                "intervention_type": c.intervention_type,
                "label": c.label,
                "alpha": c.alpha,
                "beta": c.beta,
                "posterior_mean": c.posterior_mean,
                "sample": c.sample,
                "chosen": c.chosen,
            }
            for c in selection.candidates
        ],
        "evidence": list(selection_evidence(selection, procrastination_types)),
        "note": selection.note,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Async loader (thin)
# ═══════════════════════════════════════════════════════════════════════════════

#: How far back the loader reads history.  Older rows have decayed to less than
#: 2% weight at the 14-day half-life, so reading further buys nothing.
LOOKBACK_DAYS: int = 28


async def load_history(
    repo: object,
    user_id: int,
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Read recent intervention logs for posterior derivation.

    Returns an empty list when the repository is unavailable or the read
    fails — a bandit without history simply uses its priors, which is the
    documented cold-start behaviour.  The failure is logged, never silent.
    """
    if repo is None:
        return []
    query_range = getattr(repo, "query_range", None)
    if query_range is None:
        return []
    reference = now or datetime.now(UTC)
    try:
        rows = await query_range(
            user_id, reference - timedelta(days=LOOKBACK_DAYS), reference
        )
    except Exception as exc:  # cold start: priors are the documented fallback
        logger.warning(
            "Intervention history unavailable for the bandit; using priors: {}", exc
        )
        return []
    return list(rows) if isinstance(rows, list) else []
