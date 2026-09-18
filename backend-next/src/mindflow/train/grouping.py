"""Grouping and weighting plan for leak-free model evaluation.

Two structural risks motivate the grouping and weighting contract:

1. **Grouping by calendar date leaks sessions.** A feedback session that runs
   across midnight has windows on two dates; date-only ``GroupKFold`` puts
   those windows in different folds. Training and evaluation can then share
   the same session-level feedback, even when their rows are distinct.

2. **Window-level weighting inflates long sessions.** Each matched window
   carried weight 1.0, so longer sessions received more total influence even
   though each session represents a single user judgement.

This module computes, from the label sources alone:

* a **date-block group id** per sample: dates joined transitively through any
  session that spans them, so a cross-midnight session's dates always land in
  the same fold;
* a **session id** per explicit sample, so a session's windows can be weighted
  as one observation and kept out of calibration/test together;
* **session-balanced sample weights**: each explicit session contributes a
  fixed total, divided among its windows; auxiliary labels share a bounded
  budget instead of one fixed weight per window.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

#: Default total supervision weight granted to one feedback session. Kept at
#: 1.0 so the natural unit of supervision is "one user judgement".
SESSION_WEIGHT_TOTAL = 1.0

#: Auxiliary window labels may collectively contribute at most this multiple
#: of the explicit-feedback total. Prevents auto-annotated windows from
#: outvoting the labels the user actually gave.
AUXILIARY_BUDGET_RATIO = 1.0


@dataclass
class GroupingPlan:
    """Group ids and per-sample weights derived from label provenance."""

    #: One group id per sample (index-aligned with the training matrix).
    group_ids: list[str] = field(default_factory=list)
    #: Real feedback session id per sample; empty string when not explicit.
    session_ids: list[str] = field(default_factory=list)
    #: Supervision weight per sample.
    weights: list[float] = field(default_factory=list)
    #: Distinct dates covered by each group id.
    group_dates: dict[str, list[str]] = field(default_factory=dict)
    #: Sessions whose dates were merged into a single group (cross-midnight).
    merged_sessions: list[dict[str, object]] = field(default_factory=list)
    auxiliary_total_weight: float = 0.0
    explicit_total_weight: float = 0.0


class _UnionFind:
    """Minimal union-find over date strings."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self._parent.setdefault(item, item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression, so repeated lookups stay near O(1).
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            # Canonical date ids must not depend on session input order.
            self._parent[max(left_root, right_root)] = min(left_root, right_root)


def build_grouping_plan(
    *,
    dates: list[str],
    labels: list[int],
    sources: list[str],
    window_dates_by_session: dict[str, list[str]],
    session_id_by_sample: list[str],
    auxiliary_budget_ratio: float = AUXILIARY_BUDGET_RATIO,
    session_weight_total: float = SESSION_WEIGHT_TOTAL,
) -> GroupingPlan:
    """Build the evaluation grouping and weighting plan.

    Args:
        dates: capture date (``YYYY-MM-DD``) per training row.
        labels: binary label per row (already filtered to >= 0).
        sources: label provenance per row — ``"explicit"``, ``"window_label"``,
            or ``"weak"``.
        window_dates_by_session: for each feedback session id, every date its
            matched windows fall on. Provides the transitive date links.
        session_id_by_sample: real feedback session id per row ("" when the
            row is not explicit).
        auxiliary_budget_ratio: cap on auxiliary weight relative to the
            explicit total.
        session_weight_total: total weight granted to each feedback session.

    Returns:
        A :class:`GroupingPlan` with index-aligned group ids and weights.
    """
    dates = list(dates)
    sources = list(sources)
    n = len(dates)
    if len(labels) != n or len(sources) != n or len(session_id_by_sample) != n:
        raise ValueError("Grouping provenance must align with rows")

    # ── 1. Merge dates that one session spans ────────────────────────────
    uf = _UnionFind()
    for date in set(dates):
        uf.find(date)

    merged: list[dict[str, object]] = []
    for session_id, session_dates in window_dates_by_session.items():
        unique_dates = sorted({d for d in session_dates if d})
        if len(unique_dates) <= 1:
            continue
        first = unique_dates[0]
        for other in unique_dates[1:]:
            uf.union(first, other)
        merged.append({"session_id": session_id, "dates": unique_dates})

    # ── 2. Group id per sample (date block) ──────────────────────────────
    group_ids = [uf.find(d) for d in dates]
    group_dates: dict[str, list[str]] = defaultdict(list)
    for date, group in zip(dates, group_ids, strict=True):
        if date not in group_dates[group]:
            group_dates[group].append(date)
    for group in group_dates:
        group_dates[group].sort()

    # ── 3. Session-balanced weights ──────────────────────────────────────
    weights = session_balanced_weights(
        sources=sources,
        session_ids=session_id_by_sample,
        auxiliary_budget_ratio=auxiliary_budget_ratio,
        session_weight_total=session_weight_total,
    )
    explicit_total = sum(
        w for w, source in zip(weights, sources, strict=True) if source == "explicit"
    )
    auxiliary_total = sum(
        w for w, source in zip(weights, sources, strict=True) if source == "window_label"
    )
    return GroupingPlan(
        group_ids=group_ids,
        session_ids=list(session_id_by_sample),
        weights=weights,
        group_dates={group: sorted(set(ds)) for group, ds in group_dates.items()},
        merged_sessions=merged,
        auxiliary_total_weight=round(auxiliary_total, 6),
        explicit_total_weight=round(explicit_total, 6),
    )


def session_balanced_weights(
    *,
    sources: list[str],
    session_ids: list[str],
    auxiliary_budget_ratio: float = AUXILIARY_BUDGET_RATIO,
    session_weight_total: float = SESSION_WEIGHT_TOTAL,
    allow_auxiliary_only: bool = True,
) -> list[float]:
    """Compute weights using only the rows supplied to this actual fit.

    Auxiliary-only shadow training retains its nominal one-session budget.
    Internal calibration splits disable that exception: both sides obey the
    local explicit budget, including a zero budget with no explicit sessions.
    """
    # Each explicit session contributes `session_weight_total` in total,
    # split evenly across the windows that carry its label.
    windows_per_session: dict[str, int] = defaultdict(int)
    for source, session_id in zip(sources, session_ids, strict=True):
        if source == "explicit" and session_id:
            windows_per_session[session_id] += 1

    weights: list[float] = []
    explicit_total = 0.0
    auxiliary_rows: list[int] = []

    for index, (source, session_id) in enumerate(
        zip(sources, session_ids, strict=True)
    ):
        if source == "explicit" and session_id:
            count = windows_per_session[session_id] or 1
            weight = session_weight_total / count
            weights.append(weight)
            explicit_total += weight
        elif source == "window_label":
            weights.append(0.0)  # filled in below, after the budget is known
            auxiliary_rows.append(index)
        else:
            weights.append(0.0)  # weak heuristics never supervise the gate

    if auxiliary_rows:
        # Weak heuristic rows carry no weight (see above), so the only
        # auxiliary contributors are window labels.
        budget = explicit_total * auxiliary_budget_ratio
        if budget > 0:
            per_row = budget / len(auxiliary_rows)
            for index in auxiliary_rows:
                weights[index] = per_row
        elif allow_auxiliary_only:
            # No explicit supervision at all: give auxiliary labels a nominal
            # share so a shadow run still trains, but record it truthfully.
            per_row = session_weight_total / len(auxiliary_rows)
            for index in auxiliary_rows:
                weights[index] = per_row
    return weights


__all__ = [
    "AUXILIARY_BUDGET_RATIO",
    "SESSION_WEIGHT_TOTAL",
    "GroupingPlan",
    "build_grouping_plan",
    "session_balanced_weights",
]
