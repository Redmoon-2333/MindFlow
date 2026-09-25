"""Explicit labeling functions with provenance (phase 3.4).

Before this module the only weak supervision was one opaque helper
(``train.v2._weak_label``) whose decision could not be attributed, audited, or
measured.  Phase 3.4 turns every "looks like a label" heuristic into a named
**labeling function** (LF) with:

* a stable ``name`` and a written ``description`` of the signal it encodes;
* its raw per-row output (``1`` focus, ``0`` distract, ``-1`` abstain);
* its **coverage** (share of rows where it votes at all);
* its **conflict rate** (share of its votes contradicted by another LF);
* its **agreement rate with explicit feedback**, the only ground truth.

Nothing here is a label *model*: there is deliberately no Snorkel, no learned
weights, no dependency beyond numpy.  Each LF votes on its own and the report
records what each one did, so a future label model can be judged against this
baseline instead of replacing it silently.

The weak-label *composition* (``weak_label_from_functions``) is a faithful
decomposition of the previous ``_weak_label`` helper — same guard, same order,
same thresholds — so turning the heuristics into functions did not change any
label.  New functions that are not part of that composition are marked
``advisory``: they are measured, they are visible in the report, and they only
join the composition once the report shows their agreement justifies it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: Output vocabulary of a labeling function.
FOCUS = 1
DISTRACT = 0
ABSTAIN = -1

#: Rows where this guard fires are not labelled at all: a window that is idle
#: for more than this share says nothing about focus (the user was away).
IDLE_GUARD_THRESHOLD = 0.8

#: Long single-app dwell: one app held for most of the window, little idle,
#: and enough input activity to rule out a forgotten window.
LONG_DWELL_TOP_APP_RATIO = 0.9
LONG_DWELL_MAX_IDLE_RATIO = 0.1
LONG_DWELL_MIN_INPUT_ACTIVE_RATIO = 0.15

#: High switch frequency: a switch storm with almost no focused input.
HIGH_SWITCH_COUNT = 8
HIGH_SWITCH_MAX_INPUT_ACTIVE_RATIO = 0.2

#: Input-active with a high entertainment-domain share: the user is actively
#: typing/clicking, but what they are looking at is a video/streaming surface.
#: ``audible_browser_ratio`` is the privacy-preserving proxy available in the
#: feature schema; ``browser_ratio`` + ``top_domain_ratio`` covers silent
#: entertainment pages (social feeds, short video) that never emit audio.
ENTERTAINMENT_MIN_INPUT_ACTIVE_RATIO = 0.3
ENTERTAINMENT_MIN_AUDIBLE_BROWSER_RATIO = 0.2
ENTERTAINMENT_MIN_BROWSER_RATIO = 0.5
ENTERTAINMENT_MIN_TOP_DOMAIN_RATIO = 0.4

#: The gentler legacy signal kept for cold-start days; part of the composition
#: because it is what produced labels before phase 3.4.
SUSTAINED_TOP_APP_RATIO = 0.7
SUSTAINED_MIN_INPUT_ACTIVE_RATIO = 0.3
SUSTAINED_LOW_SWITCH_COUNT = 5
SUSTAINED_MIN_TOP_APP_RATIO_LOW_SWITCH = 0.5

#: Post-intervention response: the user reported their *state* after a
#: reminder.  Helpfulness alone (helpful/neutral/annoying) is a different
#: question and never becomes a state label — see
#: ``mindflow.domain.label_contract``.
POST_INTERVENTION_STATE_KEY = "state_after_intervention"
POST_INTERVENTION_HELPFULNESS_KEY = "intervention_response"
POST_INTERVENTION_FOCUS = "focus"
POST_INTERVENTION_DISTRACT = "distracted"
POST_INTERVENTION_MIXED = "mixed"


def _number(value: Any) -> float:
    """Finite float or 0.0 — a missing feature is never treated as evidence."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


# ── Individual labeling functions ────────────────────────────────────────


def lf_explicit_feedback(
    features: Mapping[str, Any], context: Mapping[str, Any]
) -> int:
    """The user's own verdict for the session covering this window.

    This function is the *ground truth* the others are measured against.  It
    abstains whenever no explicit feedback covers the window, and it never
    inspects features: a user's judgement is not a heuristic.
    """
    label = context.get("explicit_label")
    if label is None:
        return ABSTAIN
    try:
        value = int(label)
    except (TypeError, ValueError):
        return ABSTAIN
    return value if value in (FOCUS, DISTRACT) else ABSTAIN


def lf_inactive_window_guard(
    features: Mapping[str, Any], context: Mapping[str, Any]
) -> int:
    """Guard: vetoes a row instead of labelling it.

    Returns ``1`` when the window is mostly idle (> ``IDLE_GUARD_THRESHOLD``),
    meaning the user was not there.  The composition treats a firing guard as
    "no label", which is what the legacy heuristic did before anything else.
    """
    return 1 if _number(features.get("idle_ratio")) > IDLE_GUARD_THRESHOLD else 0


def lf_long_single_app_dwell(
    features: Mapping[str, Any], context: Mapping[str, Any]
) -> int:
    """Deep work: one app held for the window with low idle and real input."""
    if (
        _number(features.get("top_app_ratio")) > LONG_DWELL_TOP_APP_RATIO
        and _number(features.get("idle_ratio")) < LONG_DWELL_MAX_IDLE_RATIO
        and _number(features.get("input_active_ratio")) > LONG_DWELL_MIN_INPUT_ACTIVE_RATIO
    ):
        return FOCUS
    return ABSTAIN


def lf_high_switch_frequency(
    features: Mapping[str, Any], context: Mapping[str, Any]
) -> int:
    """Distraction: many app switches with almost no focused input."""
    if (
        _number(features.get("app_switch_count")) > HIGH_SWITCH_COUNT
        and _number(features.get("input_active_ratio")) < HIGH_SWITCH_MAX_INPUT_ACTIVE_RATIO
    ):
        return DISTRACT
    return ABSTAIN


def lf_input_active_entertainment_share(
    features: Mapping[str, Any], context: Mapping[str, Any]
) -> int:
    """Distraction: actively interacting with an entertainment surface.

    Advisory in v1: it is measured against explicit feedback but is not part of
    the weak-label composition yet, because there is not enough labelled
    entertainment data to justify changing labels on its word alone.
    """
    active = _number(features.get("input_active_ratio"))
    if active <= ENTERTAINMENT_MIN_INPUT_ACTIVE_RATIO:
        return ABSTAIN
    audible = _number(features.get("audible_browser_ratio"))
    if audible > ENTERTAINMENT_MIN_AUDIBLE_BROWSER_RATIO:
        return DISTRACT
    if (
        _number(features.get("browser_ratio")) > ENTERTAINMENT_MIN_BROWSER_RATIO
        and _number(features.get("top_domain_ratio")) > ENTERTAINMENT_MIN_TOP_DOMAIN_RATIO
    ):
        return DISTRACT
    return ABSTAIN


def lf_post_intervention_response(
    features: Mapping[str, Any], context: Mapping[str, Any]
) -> int:
    """The user's reported state after an intervention.

    Only a *state* answer counts.  A helpfulness rating ("the reminder was
    annoying") is deliberately not converted: the label contract keeps
    "was the user distracted" and "did the reminder help" as separate
    questions, so this function abstains when only helpfulness is present.
    """
    state = context.get(POST_INTERVENTION_STATE_KEY)
    if state is None:
        return ABSTAIN
    normalised = str(state).strip().lower()
    if normalised == POST_INTERVENTION_FOCUS:
        return FOCUS
    if normalised == POST_INTERVENTION_DISTRACT:
        return DISTRACT
    if normalised == POST_INTERVENTION_MIXED:
        return ABSTAIN
    return ABSTAIN


def lf_sustained_attention(
    features: Mapping[str, Any], context: Mapping[str, Any]
) -> int:
    """The gentler legacy signal: sustained use of one app with some input."""
    top = _number(features.get("top_app_ratio"))
    active = _number(features.get("input_active_ratio"))
    switches = _number(features.get("app_switch_count"))
    if (top > SUSTAINED_TOP_APP_RATIO and active > SUSTAINED_MIN_INPUT_ACTIVE_RATIO) or (
        switches < SUSTAINED_LOW_SWITCH_COUNT and top > SUSTAINED_MIN_TOP_APP_RATIO_LOW_SWITCH
    ):
        return FOCUS
    return ABSTAIN


@dataclass(frozen=True)
class LabelingFunctionSpec:
    """One auditable labeling function."""

    name: str
    kind: str
    description: str
    source: str
    #: Position in the weak-label composition; ``None`` means "not composed".
    priority: int | None
    vote: int | None
    advisory: bool
    function: Callable[[Mapping[str, Any], Mapping[str, Any]], int]


LABELING_FUNCTIONS: tuple[LabelingFunctionSpec, ...] = (
    LabelingFunctionSpec(
        name="explicit_feedback",
        kind="ground_truth",
        description=(
            "The user's own focus/distracted verdict for the session covering "
            "this window; the reference every other function is scored against."
        ),
        source="focus_session_feedback (score/label)",
        priority=None,
        vote=None,
        advisory=False,
        function=lf_explicit_feedback,
    ),
    LabelingFunctionSpec(
        name="inactive_window_guard",
        kind="guard",
        description=(
            "Vetoes a window that is mostly idle (idle_ratio > "
            f"{IDLE_GUARD_THRESHOLD}); an absent user is not evidence about focus."
        ),
        source="features.idle_ratio",
        priority=None,
        vote=None,
        advisory=False,
        function=lf_inactive_window_guard,
    ),
    LabelingFunctionSpec(
        name="long_single_app_dwell",
        kind="behavioural",
        description=(
            "Deep work: top_app_ratio > 0.9, idle_ratio < 0.1 and "
            "input_active_ratio > 0.15."
        ),
        source="features.top_app_ratio/idle_ratio/input_active_ratio",
        priority=0,
        vote=FOCUS,
        advisory=False,
        function=lf_long_single_app_dwell,
    ),
    LabelingFunctionSpec(
        name="high_switch_frequency",
        kind="behavioural",
        description=(
            "Distraction: app_switch_count > 8 with input_active_ratio < 0.2."
        ),
        source="features.app_switch_count/input_active_ratio",
        priority=1,
        vote=DISTRACT,
        advisory=False,
        function=lf_high_switch_frequency,
    ),
    LabelingFunctionSpec(
        name="sustained_attention",
        kind="behavioural",
        description=(
            "Gentle cold-start signal: (top_app_ratio > 0.7 and "
            "input_active_ratio > 0.3) or (app_switch_count < 5 and "
            "top_app_ratio > 0.5)."
        ),
        source="features.top_app_ratio/input_active_ratio/app_switch_count",
        priority=2,
        vote=FOCUS,
        advisory=False,
        function=lf_sustained_attention,
    ),
    LabelingFunctionSpec(
        name="input_active_entertainment_share",
        kind="behavioural",
        description=(
            "Distraction: input_active_ratio > 0.3 together with an "
            "entertainment-surface share (audible_browser_ratio > 0.2, or "
            "browser_ratio > 0.5 and top_domain_ratio > 0.4)."
        ),
        source="features.audible_browser_ratio/browser_ratio/top_domain_ratio",
        priority=None,
        vote=DISTRACT,
        advisory=True,
        function=lf_input_active_entertainment_share,
    ),
    LabelingFunctionSpec(
        name="post_intervention_response",
        kind="contextual",
        description=(
            "The user's reported state after a reminder "
            "(state_after_intervention); helpfulness ratings are never "
            "converted into a state label."
        ),
        source="focus_session_feedback.intervention_response",
        priority=None,
        vote=None,
        advisory=True,
        function=lf_post_intervention_response,
    ),
)

#: Order in which the composed weak label is resolved (see the module docstring).
COMPOSITION_ORDER: tuple[str, ...] = ("long_single_app_dwell", "high_switch_frequency",
                                      "sustained_attention")
GUARD_NAME = "inactive_window_guard"
GROUND_TRUTH_NAME = "explicit_feedback"

_SPECS_BY_NAME: dict[str, LabelingFunctionSpec] = {
    spec.name: spec for spec in LABELING_FUNCTIONS
}


def compute_labeling_function_outputs(
    features: Mapping[str, Any], context: Mapping[str, Any] | None = None
) -> dict[str, int]:
    """Raw output of every labeling function for one row."""
    resolved = dict(context or {})
    return {
        spec.name: spec.function(features, resolved) for spec in LABELING_FUNCTIONS
    }


def weak_label_from_functions(outputs: Mapping[str, int]) -> int:
    """Resolve the composed weak label from per-function outputs.

    Identical to the pre-3.4 ``_weak_label`` helper: the idle guard vetoes,
    then the first voting function in ``COMPOSITION_ORDER`` wins, and a row no
    function votes on stays unlabelled (``ABSTAIN``).
    """
    if outputs.get(GUARD_NAME, 0) == 1:
        return ABSTAIN
    for name in COMPOSITION_ORDER:
        vote = outputs.get(name, ABSTAIN)
        if vote in (FOCUS, DISTRACT):
            return vote
    return ABSTAIN


def weak_label(features: Mapping[str, Any]) -> int:
    """Composed weak label for a feature dict (no explicit feedback available)."""
    return weak_label_from_functions(compute_labeling_function_outputs(features))


# ── Provenance report ────────────────────────────────────────────────────


def build_labeling_function_report(
    features_rows: Sequence[Mapping[str, Any]],
    explicit_labels: Sequence[int | None],
    *,
    contexts: Sequence[Mapping[str, Any]] | None = None,
    universe: str = "kept_training_rows",
) -> dict[str, Any]:
    """Measure every labeling function over one training frame.

    Args:
        features_rows: feature dict per row of the frame.
        explicit_labels: the explicit feedback label per row (``None`` when the
            row has no real user label).  This is the only ground truth.
        contexts: optional per-row context (post-intervention answers).
        universe: human-readable description of the row set being measured.

    Returns:
        A JSON-serialisable report: per-function output counts, coverage,
        conflict rate and agreement with explicit feedback, plus a pairwise
        conflict matrix and the composition contract.
    """
    if len(features_rows) != len(explicit_labels):
        raise ValueError("Labeling-function provenance must align with rows")
    row_contexts: list[Mapping[str, Any]] = (
        list(contexts) if contexts is not None else [{} for _ in features_rows]
    )
    if len(row_contexts) != len(features_rows):
        raise ValueError("Labeling-function contexts must align with rows")

    total = len(features_rows)
    outputs_per_row: list[dict[str, int]] = []
    for index, features in enumerate(features_rows):
        context = dict(row_contexts[index])
        context["explicit_label"] = explicit_labels[index]
        outputs_per_row.append(compute_labeling_function_outputs(features, context))

    ground_truth_rows = sum(1 for label in explicit_labels if label in (FOCUS, DISTRACT))
    entries: list[dict[str, Any]] = []
    conflict_matrix: dict[str, dict[str, int]] = {}
    conflicting_rows = 0

    for spec in LABELING_FUNCTIONS:
        fires = 0
        abstains = 0
        votes: dict[int, int] = {FOCUS: 0, DISTRACT: 0}
        comparable = 0
        agreements = 0
        conflicts = 0
        conflicts_with: dict[str, int] = {}
        for index, outputs in enumerate(outputs_per_row):
            if spec.kind == "guard":
                # A guard's "vote" is a veto flag, not a label.
                fired = outputs.get(spec.name, 0) == 1
            else:
                fired = outputs.get(spec.name, ABSTAIN) in (FOCUS, DISTRACT)
            if not fired:
                abstains += 1
                continue
            fires += 1
            if spec.kind == "guard":
                continue
            vote = int(outputs[spec.name])
            votes[vote] = votes.get(vote, 0) + 1
            truth = explicit_labels[index]
            if truth in (FOCUS, DISTRACT):
                comparable += 1
                if truth == vote:
                    agreements += 1
            # A conflicting row is one where another function votes the
            # opposite label for the same row.
            row_conflict = False
            for other in LABELING_FUNCTIONS:
                if other.name == spec.name or other.kind == "guard":
                    continue
                if other.name == GROUND_TRUTH_NAME:
                    continue
                other_vote = outputs.get(other.name, ABSTAIN)
                if other_vote in (FOCUS, DISTRACT) and other_vote != vote:
                    conflicts_with[other.name] = conflicts_with.get(other.name, 0) + 1
                    row_conflict = True
            if row_conflict:
                conflicts += 1
        entries.append({
            "name": spec.name,
            "kind": spec.kind,
            "description": spec.description,
            "source": spec.source,
            "in_weak_composition": spec.priority is not None,
            "advisory": spec.advisory,
            "priority": spec.priority,
            "fires": fires,
            "abstains": abstains,
            "coverage": round(fires / total, 6) if total else 0.0,
            "votes": {
                "focus": votes.get(FOCUS, 0),
                "distracted": votes.get(DISTRACT, 0),
            },
            "comparable_rows": comparable,
            "agreement_rows": agreements,
            "agreement_rate": (
                round(agreements / comparable, 6) if comparable else None
            ),
            "conflict_rows": conflicts,
            "conflict_rate": round(conflicts / fires, 6) if fires else 0.0,
            "conflicts_with": conflicts_with,
        })
        conflict_matrix[spec.name] = conflicts_with

    # Count rows where at least two voting functions disagreed.
    voting_names = [
        spec.name for spec in LABELING_FUNCTIONS
        if spec.kind not in ("guard", "ground_truth")
    ]
    for outputs in outputs_per_row:
        distinct_votes = {
            outputs.get(name, ABSTAIN) for name in voting_names
        } - {ABSTAIN}
        if len(distinct_votes) > 1:
            conflicting_rows += 1

    return {
        "version": 1,
        "status": "reported",
        "universe": universe,
        "rows_considered": total,
        "ground_truth_rows": ground_truth_rows,
        "conflicting_rows": conflicting_rows,
        "functions": entries,
        "conflict_matrix": conflict_matrix,
        "composition": {
            "guard": GUARD_NAME,
            "order": list(COMPOSITION_ORDER),
            "composed": ["inactive_window_guard", *COMPOSITION_ORDER],
            "advisory": [
                spec.name for spec in LABELING_FUNCTIONS
                if spec.advisory and spec.name not in COMPOSITION_ORDER
            ],
            "ground_truth": GROUND_TRUTH_NAME,
            "note": (
                "The composed weak label is unchanged from the pre-3.4 "
                "heuristic; advisory functions are measured here and may only "
                "enter the composition once their reported agreement justifies "
                "it."
            ),
        },
        "note": (
            "Heuristic labeling functions only — no label model (Snorkel or "
            "similar) is used, and no weight is learned."
        ),
    }


__all__ = [
    "ABSTAIN",
    "COMPOSITION_ORDER",
    "DISTRACT",
    "FOCUS",
    "GROUND_TRUTH_NAME",
    "GUARD_NAME",
    "LABELING_FUNCTIONS",
    "LabelingFunctionSpec",
    "build_labeling_function_report",
    "compute_labeling_function_outputs",
    "lf_explicit_feedback",
    "lf_high_switch_frequency",
    "lf_inactive_window_guard",
    "lf_input_active_entertainment_share",
    "lf_long_single_app_dwell",
    "lf_post_intervention_response",
    "lf_sustained_attention",
    "weak_label",
    "weak_label_from_functions",
]
