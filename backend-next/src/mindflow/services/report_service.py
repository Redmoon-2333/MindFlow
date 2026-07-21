"""Report service: daily and weekly report generation.

Generates idempotent daily reports and weekly trend summaries from the
focus session projection and raw event stream (Wave 5).

Daily reports include:
  - Total focus / distraction minutes
  - Focus score (from domain features)
  - Top applications by usage
  - Switch frequency
  - Chinese pattern summary (generated from session statistics)

Weekly reports aggregate 7 daily reports and compute week-over-week deltas.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from typing import Any

from loguru import logger

from mindflow.domain.features import (
    MAX_ACCEPTABLE_SWITCHES_PER_HOUR,
    app_usage_ranking,
    switch_rate_per_hour,
)
from mindflow.domain.features import (
    focus_score as compute_focus_score,
)
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
)
from mindflow.infrastructure.repositories.focus import (
    SQLAlchemyFocusSessionRepository,
)
from mindflow.infrastructure.repositories.report import (
    SQLAlchemyDailyReportRepository,
)
from mindflow.services.effectiveness_service import (
    EffectivenessService,
)


class ReportService:
    """Daily and weekly report generation.

    Args:
        activity_repo: Repository for activity events.
        focus_repo: Repository for focus sessions.
        report_repo: Repository for daily reports.
        effectiveness_svc: Optional effectiveness service for
            intervention impact data in weekly reports.
    """

    def __init__(
        self,
        activity_repo: SQLAlchemyActivityRepository,
        focus_repo: SQLAlchemyFocusSessionRepository,
        report_repo: SQLAlchemyDailyReportRepository,
        effectiveness_svc: EffectivenessService | None = None,
    ) -> None:
        self._activity_repo = activity_repo
        self._focus_repo = focus_repo
        self._report_repo = report_repo
        self._effectiveness_svc = effectiveness_svc

    # ── Daily report ─────────────────────────────────────────────────

    async def generate_daily_report(
        self,
        user_id: int,
        target_date: date,
    ) -> dict[str, Any]:
        """Generate an idempotent daily report for *user_id* on *target_date*.

        **Idempotency:** if a report already exists for this user+date, it
        is returned immediately without recomputation.

        **Steps:**
          1. Retrieve focus sessions for the day.
          2. Aggregate total focus / distraction minutes from sessions.
          3. Compute focus score and switch frequency from raw events
             (using ``domain.features`` functions, not session data).
          4. Rank applications by total duration.
          5. Generate a Chinese pattern summary.
          6. Persist via ``SQLAlchemyDailyReportRepository.upsert()``.

        Args:
            user_id: User identifier.
            target_date: The report date.

        Returns:
            The persisted report dict.
        """
        # Idempotency check
        existing = await self._report_repo.get_by_date(user_id, target_date)
        if existing is not None:
            logger.info("Daily report already exists for {} user {}", target_date, user_id)
            return existing

        # Sessions and same-day events are independent reads (no data
        # dependency) — fetch concurrently. Both run only after the
        # idempotency check above short-circuits.
        start_dt = datetime(target_date.year, target_date.month, target_date.day, tzinfo=UTC)
        end_dt = start_dt + timedelta(days=1) - timedelta(seconds=1)

        sessions, events = await asyncio.gather(
            self._focus_repo.get_by_date(user_id, target_date),
            self._activity_repo.query_range(user_id, start_dt, end_dt),
        )

        # Aggregate session-based metrics
        total_focus_min = 0.0
        total_distraction_min = 0.0
        for s in sessions:
            try:
                start_ts = datetime.fromisoformat(s["start_time"])
                end_ts = datetime.fromisoformat(s["end_time"])
                duration_min = (end_ts - start_ts).total_seconds() / 60.0
            except (ValueError, KeyError):
                duration_min = 0.0

            if s.get("session_type") == "focus":
                total_focus_min += duration_min
            elif s.get("session_type") == "distraction":
                total_distraction_min += duration_min

        # Compute event-based metrics via domain features
        score = compute_focus_score(events)
        switch_freq = switch_rate_per_hour(events)
        usage = app_usage_ranking(events)

        top_apps_data = [
            {"app": u.app_name, "minutes": round(u.total_duration_s / 60.0, 1)}
            for u in usage[:10]
        ]

        # Chinese pattern summary
        summary = _build_pattern_summary(
            total_focus_min=total_focus_min,
            total_distraction_min=total_distraction_min,
            focus_score=score,
            switch_frequency=switch_freq,
            session_count=len(sessions),
            top_apps=top_apps_data[:3],
        )

        report_data: dict[str, Any] = {
            "user_id": user_id,
            "date": target_date.isoformat(),
            "total_focus_min": round(total_focus_min, 1),
            "total_distraction_min": round(total_distraction_min, 1),
            "focus_score": score,
            "top_apps": top_apps_data,
            "switch_frequency": round(switch_freq, 2),
            "pattern_summary": summary,
        }

        result = await self._report_repo.upsert(report_data)
        logger.info(
            "Daily report generated for user {} on {} (focus={})",
            user_id,
            target_date,
            score,
        )
        return result

    # ── Weekly report ────────────────────────────────────────────────

    async def weekly_report(
        self,
        user_id: int,
        week_start: date,
    ) -> dict[str, Any]:
        """Generate a weekly summary with 7-day trend and week-over-week comparison.

        Args:
            user_id: User identifier.
            week_start: The Monday (or start day) of the target week.

        Returns:
            A dict with:
              - ``week_start`` / ``week_end``: ISO date range.
              - ``daily_reports``: list of individual daily report summaries.
              - ``averages``: weekly averages (focus_min, distraction_min,
                focus_score, switch_frequency).
              - ``trend``: week-over-week deltas vs the previous week.
              - ``week_number``: ISO week number.
        """
        week_end = week_start + timedelta(days=6)

        # Fetch or generate reports for each day
        daily: list[dict[str, Any]] = []
        current = week_start
        while current <= week_end:
            report = await self.generate_daily_report(user_id, current)
            daily.append(report)
            current += timedelta(days=1)

        if not daily:
            return {
                "week_start": week_start.isoformat(),
                "week_end": week_end.isoformat(),
                "daily_reports": [],
                "averages": {},
                "trend": {},
                "week_number": week_start.isocalendar()[1],
            }

        # Weekly averages
        n = len(
            [
                d
                for d in daily
                if d.get("total_focus_min", 0) > 0
                or d.get("total_distraction_min", 0) > 0
            ]
        )
        n = max(n, 1)
        avg_focus = sum(d.get("total_focus_min", 0.0) for d in daily) / n
        avg_distraction = sum(d.get("total_distraction_min", 0.0) for d in daily) / n
        avg_score = sum(d.get("focus_score", 0.0) for d in daily) / n
        avg_switch = sum(d.get("switch_frequency", 0.0) for d in daily) / n

        averages = {
            "avg_focus_min": round(avg_focus, 1),
            "avg_distraction_min": round(avg_distraction, 1),
            "avg_focus_score": round(avg_score, 1),
            "avg_switch_frequency": round(avg_switch, 2),
        }

        # Week-over-week comparison
        prev_start = week_start - timedelta(days=7)
        prev_reports = await self._report_repo.query_range(
            user_id, prev_start, prev_start + timedelta(days=6)
        )

        trend: dict[str, Any] = {}
        if prev_reports:
            pn = max(len([r for r in prev_reports if r.get("total_focus_min", 0) > 0]), 1)
            prev_avg_focus = sum(r.get("total_focus_min", 0.0) for r in prev_reports) / pn
            prev_avg_score = sum(r.get("focus_score", 0.0) for r in prev_reports) / pn

            focus_delta_pct = (
                round((avg_focus - prev_avg_focus) / prev_avg_focus * 100, 1)
                if prev_avg_focus > 0
                else 0.0
            )
            score_delta = round(avg_score - prev_avg_score, 1)

            direction = (
                "up" if focus_delta_pct > 0 else ("down" if focus_delta_pct < 0 else "stable")
            )
            trend = {
                "focus_min_delta_pct": focus_delta_pct,
                "focus_score_delta": score_delta,
                "direction": direction,
            }

        # Wave 7: Intervention effectiveness (optional — null when not wired)
        intervention_effectiveness: dict[str, Any] | None = None
        if self._effectiveness_svc is not None:
            try:
                intervention_effectiveness = await self._effectiveness_svc.weekly_effectiveness(
                    user_id, days=7
                )
            except Exception:
                logger.warning("Failed to fetch intervention effectiveness for weekly report")
                intervention_effectiveness = None

        return {
            "week_start": week_start.isoformat(),
            "week_end": week_end.isoformat(),
            "daily_reports": daily,
            "averages": averages,
            "trend": trend,
            "week_number": week_start.isocalendar()[1],
            "intervention_effectiveness": intervention_effectiveness,
        }

    # ── Scheduler convenience ─────────────────────────────────────────

    async def generate_daily_for_all(self, user_id: int = 1) -> dict[str, Any]:
        """Convenience wrapper for scheduled job — generates today's report."""
        return await self.generate_daily_report(user_id, date.today())


# ── Internal helpers ──────────────────────────────────────────────────


def _build_pattern_summary(
    total_focus_min: float,
    total_distraction_min: float,
    focus_score: float,
    switch_frequency: float,
    session_count: int,
    top_apps: list[dict[str, Any]],
) -> str:
    """Build a Chinese-language pattern summary for a daily report.

    The summary describes the day's focus quality, application usage,
    and provides actionable suggestions.
    """
    parts: list[str] = []

    # Focus quality
    if focus_score >= 80:
        parts.append("今日专注状态良好")
    elif focus_score >= 60:
        parts.append("今日专注状态中等")
    elif focus_score >= 40:
        parts.append("今日专注状态偏低")
    else:
        parts.append("今日专注状态不佳")

    parts.append(
        f"，专注{total_focus_min:.0f}分钟，分心{total_distraction_min:.0f}分钟"
    )

    # Switch frequency assessment
    if switch_frequency > MAX_ACCEPTABLE_SWITCHES_PER_HOUR:
        parts.append("，任务切换较为频繁")
    elif switch_frequency > MAX_ACCEPTABLE_SWITCHES_PER_HOUR / 2:
        parts.append("，切换频率适中")
    else:
        parts.append("，任务切换较少")

    # Top apps
    if top_apps:
        app_names = [a["app"] for a in top_apps[:3]]
        parts.append(f"。主要使用应用：{'、'.join(app_names)}")

    # Session count
    if session_count >= 6:
        parts.append("，专注块数量充足")
    elif session_count >= 3:
        parts.append("，专注块数量适中")
    else:
        parts.append("，连续专注时间偏少")

    # Suggestion
    if focus_score < 60:
        parts.append("。建议尝试番茄工作法，每25分钟休息5分钟以提高专注力")
    elif total_distraction_min > total_focus_min:
        parts.append("。分心时间超过专注时间，建议检查通知权限并减少多任务处理")

    return "".join(parts)
