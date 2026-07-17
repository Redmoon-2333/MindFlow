"""Intervention effectiveness evaluation (C5 minimal design).

Provides two analyses:
  1. Before/after window comparison for a single intervention.
  2. Weekly aggregate effectiveness for reports.

Effectiveness is measured as the change in three metrics between the
30-minute window before an intervention and the 30-minute window after:
  - focus_score (domain.features)
  - switch_rate (switches per hour)
  - distraction_ratio (fraction of events in distraction apps)

This is a minimal deterministic implementation — no ML, no significance
testing, no user grouping.  The architecture doc §3.8 describes the
planned evolution for future waves.

Design constraints:
  - All computations are read-only — no DB writes.
  - Never raises; returns zeroed reports on missing data.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from loguru import logger

from mindflow.domain.features import focus_score, switch_rate_per_hour
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
)
from mindflow.infrastructure.repositories.intervention import (
    InterventionLogRepository,
)

_WINDOW_MINUTES: float = 30.0
"""Default window size for before/after comparison (30 min each side)."""


class EffectivenessReport:
    """Comparison of metrics before and after an intervention.

    Attributes:
        intervention_id: The intervention's UUID.
        before: Dict of metrics for the 30-minute window before.
        after: Dict of metrics for the 30-minute window after.
        deltas: Change in each metric (after - before).
        has_data: True if both windows had sufficient event data.
    """

    def __init__(
        self,
        intervention_id: str,
        before: dict[str, float],
        after: dict[str, float],
        deltas: dict[str, float],
        has_data: bool = False,
    ) -> None:
        self.intervention_id = intervention_id
        self.before = before
        self.after = after
        self.deltas = deltas
        self.has_data = has_data


class EffectivenessService:
    """Intervention effectiveness evaluation.

    Args:
        activity_repo: Repository for activity events.
        intervention_repo: Intervention log repository.
    """

    def __init__(
        self,
        activity_repo: SQLAlchemyActivityRepository,
        intervention_repo: InterventionLogRepository,
    ) -> None:
        self._activity_repo = activity_repo
        self._intervention_repo = intervention_repo

    async def compare_windows(
        self,
        intervention_id: str,
        window_minutes: float = _WINDOW_MINUTES,
    ) -> EffectivenessReport:
        """Compare metrics before vs after a single intervention.

        Args:
            intervention_id: The intervention's UUID.
            window_minutes: Size of each window in minutes (default 30).

        Returns:
            An ``EffectivenessReport`` with before/after metrics and deltas.
        """
        log = await self._intervention_repo.get_by_id(intervention_id)
        if log is None:
            return self._empty_report(intervention_id)

        user_id = log["user_id"]
        triggered_at_str = log.get("triggered_at", "")
        try:
            triggered_at = datetime.fromisoformat(triggered_at_str)
        except (ValueError, TypeError):
            logger.warning("Invalid triggered_at for intervention {}", intervention_id)
            return self._empty_report(intervention_id)

        # Ensure timezone-aware
        if triggered_at.tzinfo is None:
            triggered_at = triggered_at.replace(tzinfo=UTC)

        window = timedelta(minutes=window_minutes)

        before_start = triggered_at - window
        before_end = triggered_at
        after_start = triggered_at
        after_end = triggered_at + window

        try:
            before_events = await self._activity_repo.query_range(user_id, before_start, before_end)
            after_events = await self._activity_repo.query_range(user_id, after_start, after_end)

            before_metrics = self._compute_metrics(before_events)
            after_metrics = self._compute_metrics(after_events)

            has_data = len(before_events) >= 5 and len(after_events) >= 5

            deltas = {
                k: round(after_metrics.get(k, 0.0) - before_metrics.get(k, 0.0), 2)
                for k in before_metrics
            }

            return EffectivenessReport(
                intervention_id=intervention_id,
                before=before_metrics,
                after=after_metrics,
                deltas=deltas,
                has_data=has_data,
            )
        except Exception as exc:
            logger.error("Error comparing windows for {}: {}", intervention_id, exc)
            return self._empty_report(intervention_id)

    async def weekly_effectiveness(
        self,
        user_id: int,
        days: int = 7,
    ) -> dict[str, Any]:
        """Aggregate effectiveness stats for the last *days*.

        Returns a dict with:
          - ``total_interventions``: count in the window.
          - ``with_feedback``: count that have user response data.
          - ``acceptance_rate``: fraction of responded interventions that
            were accepted.
          - ``avg_focus_delta``: average focus_score change across all
            interventions with sufficient data.
          - ``avg_switch_delta``: average switch-rate change.
          - ``avg_distraction_delta``: average distraction-ratio change.

        Returns empty/zeroed data when no interventions exist in the window.
        """
        now = datetime.now(UTC)
        start = now - timedelta(days=days)

        try:
            logs = await self._intervention_repo.query_range(user_id, start, now)
        except Exception as exc:
            logger.error("Error fetching intervention history: {}", exc)
            return self._empty_weekly()

        if not logs:
            return self._empty_weekly()

        responded = [log for log in logs if log.get("user_response") is not None]
        accepted = [log for log in responded if log.get("user_response") == "accepted"]
        acceptance_rate = len(accepted) / len(responded) if responded else 0.0

        # Compute deltas for each intervention
        focus_deltas: list[float] = []
        switch_deltas: list[float] = []
        distraction_deltas: list[float] = []

        for log_entry in logs[:20]:  # Limit to 20 for performance
            lid = log_entry.get("id", "")
            report = await self.compare_windows(lid)
            if report.has_data:
                focus_deltas.append(report.deltas.get("focus_score", 0.0))
                switch_deltas.append(report.deltas.get("switch_rate", 0.0))
                distraction_deltas.append(report.deltas.get("distraction_ratio", 0.0))

        def _safe_avg(values: list[float]) -> float:
            return round(sum(values) / len(values), 2) if values else 0.0

        return {
            "total_interventions": len(logs),
            "with_feedback": len(responded),
            "acceptance_rate": round(acceptance_rate, 3),
            "avg_focus_delta": _safe_avg(focus_deltas),
            "avg_switch_delta": _safe_avg(switch_deltas),
            "avg_distraction_delta": _safe_avg(distraction_deltas),
        }

    # ── Internal helpers ──────────────────────────────────────────────

    def _compute_metrics(self, events: list[Any]) -> dict[str, float]:
        """Compute focus_score, switch_rate, and distraction_ratio for a window."""
        if not events:
            return {"focus_score": 0.0, "switch_rate": 0.0, "distraction_ratio": 0.0}

        score = focus_score(events)
        srate = switch_rate_per_hour(events)

        # Simple distraction ratio: fraction of events in apps that are
        # not the top-3 most-used work apps in this window.
        total = len(events)
        if total == 0:
            distraction_ratio = 0.0
        else:
            # Count events by app, find top-3, everything else is "distraction"
            app_counts: dict[str, int] = {}
            for ev in events:
                app = ev.data.process_name if hasattr(ev.data, "process_name") else ""
                app_counts[app] = app_counts.get(app, 0) + 1
            sorted_apps = sorted(app_counts.items(), key=lambda x: x[1], reverse=True)
            top_3_apps = set(a[0] for a in sorted_apps[:3])
            distraction_count = sum(c for a, c in sorted_apps if a not in top_3_apps)
            distraction_ratio = distraction_count / total

        return {
            "focus_score": round(score, 1),
            "switch_rate": round(srate, 2),
            "distraction_ratio": round(distraction_ratio, 3),
        }

    @staticmethod
    def _empty_report(intervention_id: str) -> EffectivenessReport:
        """Return a zeroed report for missing/invalid data."""
        zero = {"focus_score": 0.0, "switch_rate": 0.0, "distraction_ratio": 0.0}
        return EffectivenessReport(
            intervention_id=intervention_id,
            before=zero,
            after=zero,
            deltas={"focus_score": 0.0, "switch_rate": 0.0, "distraction_ratio": 0.0},
            has_data=False,
        )

    @staticmethod
    def _empty_weekly() -> dict[str, Any]:
        """Return a zeroed weekly summary."""
        return {
            "total_interventions": 0,
            "with_feedback": 0,
            "acceptance_rate": 0.0,
            "avg_focus_delta": 0.0,
            "avg_switch_delta": 0.0,
            "avg_distraction_delta": 0.0,
        }
