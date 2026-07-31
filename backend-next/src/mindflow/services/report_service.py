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

from mindflow.domain.events import ActivityEvent
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
from mindflow.time_utils import (
    TimezoneLike,
    business_day_bounds_utc,
    business_today,
    resolve_timezone,
)


def _compute_daily_data_state(
    target_date: date,
    sessions: list[dict[str, Any]],
    events: list[ActivityEvent],
    today: date,
) -> str:
    """Classify one business day into the canonical daily data-state enum.

    ``future`` takes precedence (nothing can exist after business today);
    then ``no_activity`` (no sessions and no events), ``events_only``
    (events without sessions), ``neutral_only`` (sessions with neither a
    focus nor a distraction type), ``no_focus`` (distraction without
    focus), and finally ``ready`` once at least one focus session exists.
    """
    if target_date > today:
        return "future"
    if not sessions and not events:
        return "no_activity"
    if not sessions:
        return "events_only"
    has_focus = any(s.get("session_type") == "focus" for s in sessions)
    has_distraction = any(s.get("session_type") == "distraction" for s in sessions)
    if has_focus:
        return "ready"
    if has_distraction:
        return "no_focus"
    return "neutral_only"


def _decorate_daily_report(
    report: dict[str, Any],
    sessions: list[dict[str, Any]],
    events: list[ActivityEvent],
    target_date: date,
    timezone: TimezoneLike,
    today: date,
) -> dict[str, Any]:
    """Decorate a daily report dict with the canonical transient field set.

    One assembly point shared by the freshly generated and the cached
    paths: the DB stores ``total_focus_min`` / ``total_distraction_min``
    but every 200 response exposes ``total_focus_minutes`` /
    ``total_sessions`` / ``total_distractions`` / ``hourly_distribution``
    (from the live single-day read, never hard-coded zeros) plus the
    explicit ``data_state``.  ``top_apps`` is normalized to a list so the
    typed schema never sees null.
    """
    if report.get("top_apps") is None:
        report["top_apps"] = []
    if "total_focus_minutes" not in report:
        report["total_focus_minutes"] = report.get("total_focus_min", 0)
    report["total_sessions"] = len(sessions)
    report["total_distractions"] = sum(
        1 for s in sessions if s.get("session_type") == "distraction"
    )
    report["hourly_distribution"] = _compute_hourly_distribution(
        sessions, target_date, timezone
    )
    report["data_state"] = _compute_daily_data_state(
        target_date, sessions, events, today
    )
    return report


def _parse_session_interval(
    session: dict[str, Any],
) -> tuple[datetime, datetime] | None:
    """Parse a session's start/end into aware UTC datetimes, or None.

    Returns None for unparseable, missing, or inverted timestamps so the
    caller can treat the session as contributing zero minutes.
    """
    try:
        start_ts = datetime.fromisoformat(session["start_time"])
        end_ts = datetime.fromisoformat(session["end_time"])
    except (ValueError, TypeError, KeyError):
        return None
    if start_ts.tzinfo is None:
        start_ts = start_ts.replace(tzinfo=UTC)
    if end_ts.tzinfo is None:
        end_ts = end_ts.replace(tzinfo=UTC)
    if end_ts <= start_ts:
        return None
    return start_ts, end_ts


def _compute_hourly_distribution(
    sessions: list[dict[str, Any]],
    target_date: date,
    timezone: TimezoneLike,
) -> dict[str, float]:
    """Return ``{hour: rounded_focus_minutes}`` for one local business day.

    Only ``focus`` sessions contribute.  Each session is first clipped to
    the business-day UTC bounds (so a session spilling past local midnight
    stops at the day boundary), then its minutes are split across the local
    hour buckets it overlaps.  Hour boundaries follow the local wall clock,
    so DST-observed timezones (skipped/repeated hours) map sessions to the
    hour their local time actually shows.  All keys ``"0"``..``"23"`` are
    always present so chart consumers never see a missing hour.
    """
    day_start, day_end = business_day_bounds_utc(target_date, timezone)
    local_tz = resolve_timezone(timezone)
    bucket_minutes = [0.0] * 24

    for session in sessions:
        if session.get("session_type") != "focus":
            continue
        interval = _parse_session_interval(session)
        if interval is None:
            continue
        start_ts, end_ts = interval
        clipped_start = max(start_ts, day_start)
        clipped_end = min(end_ts, day_end)
        if clipped_end <= clipped_start:
            continue

        cursor = clipped_start
        while cursor < clipped_end:
            cursor_local = cursor.astimezone(local_tz)
            hour = cursor_local.hour
            # The next local-hour boundary from the current instant.  Using
            # the wall-clock floor keeps DST transitions exact (the skipped
            # spring hour never appears; the repeated fall hour advances by
            # its own UTC offset).
            hour_floor = cursor_local.replace(minute=0, second=0, microsecond=0)
            next_hour_utc = (hour_floor + timedelta(hours=1)).astimezone(UTC)
            chunk_end = min(clipped_end, next_hour_utc)
            bucket_minutes[hour] += (chunk_end - cursor).total_seconds() / 60.0
            cursor = chunk_end

    return {str(hour): round(bucket_minutes[hour], 1) for hour in range(24)}


def _future_daily_report(user_id: int, target_date: date) -> dict[str, Any]:
    """Build the typed empty payload for a business day that hasn't happened.

    Empty arrays/objects rather than fabricated bars; never persisted.
    """
    return {
        "id": "",
        "user_id": user_id,
        "date": target_date.isoformat(),
        "total_focus_min": 0.0,
        "total_distraction_min": 0.0,
        "focus_score": 0.0,
        "top_apps": [],
        "switch_frequency": 0.0,
        "pattern_summary": "",
        "created_at": None,
        "total_focus_minutes": 0.0,
        "total_sessions": 0,
        "total_distractions": 0,
        "hourly_distribution": {str(hour): 0.0 for hour in range(24)},
        "data_state": "future",
    }


def _compute_weekly_data_state(
    week_start: date,
    today: date,
    week_end: date,
    daily_reports: list[dict[str, Any]],
) -> str:
    """Classify one week into the canonical weekly data-state enum.

    ``future`` when the week starts after business today; ``no_activity``
    when every produced day is ``no_activity`` (takes precedence over the
    in-progress check); ``partial`` when the week is current/incomplete
    (today still inside the week) or any produced day is non-ready; else
    ``ready``.
    """
    if week_start > today:
        return "future"
    if not daily_reports:
        return "no_activity"
    if all(d.get("data_state") == "no_activity" for d in daily_reports):
        return "no_activity"
    if any(d.get("data_state") != "ready" for d in daily_reports) or today <= week_end:
        return "partial"
    return "ready"


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
        timezone: str = "local",
    ) -> None:
        self._activity_repo = activity_repo
        self._focus_repo = focus_repo
        self._report_repo = report_repo
        self._effectiveness_svc = effectiveness_svc
        self._timezone = timezone

    # ── Daily report ─────────────────────────────────────────────────

    async def generate_daily_report(
        self,
        user_id: int,
        target_date: date,
        *,
        refresh: bool = False,
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
            refresh: Recompute even when an existing report appears final.

        Returns:
            The persisted report dict.
        """
        today = business_today(self._timezone)
        # Nothing can exist for a business day after today — return the
        # typed empty payload without persisting a placeholder row.
        if target_date > today:
            return _future_daily_report(user_id, target_date)

        # Reports created before the business day ended are provisional.
        existing = await self._report_repo.get_by_date(user_id, target_date)
        if existing is not None and not refresh:
            created_at = existing.get("created_at")
            _, day_end = business_day_bounds_utc(target_date, self._timezone)
            try:
                created_at_dt = datetime.fromisoformat(created_at) if created_at else None
                if created_at_dt is not None and created_at_dt.tzinfo is None:
                    created_at_dt = created_at_dt.replace(tzinfo=UTC)
            except (TypeError, ValueError):
                created_at_dt = None
            if created_at_dt is not None and created_at_dt > day_end:
                logger.info(
                    "Daily report already exists for {} user {}", target_date, user_id
                )
                # Cached summaries still need the transient fields and the
                # explicit data state, computed from the live single-day
                # sessions + events read — never hard-coded zeros.
                sessions, events = await self._read_day_data(user_id, target_date)
                return _decorate_daily_report(
                    existing,
                    sessions,
                    events,
                    target_date,
                    self._timezone,
                    today,
                )

        sessions, events = await self._read_day_data(user_id, target_date)

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
        # One shared assembly point: the same transient field set (aliases,
        # hourly_distribution, data_state) the cached path decorates.
        return _decorate_daily_report(
            result, sessions, events, target_date, self._timezone, today
        )

    async def _read_day_data(
        self,
        user_id: int,
        target_date: date,
    ) -> tuple[list[dict[str, Any]], list[ActivityEvent]]:
        """Read one business day's sessions and events concurrently.

        Sessions and same-day events are independent reads (no data
        dependency) shared by fresh generation and cached decoration.
        """
        start_dt, end_dt = business_day_bounds_utc(target_date, self._timezone)
        sessions, events = await asyncio.gather(
            self._focus_repo.get_by_date(user_id, target_date),
            self._activity_repo.query_range(user_id, start_dt, end_dt),
        )
        return sessions, events

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
        today = business_today(self._timezone)
        generation_end = min(week_end, today)

        # Build ordered date list, then generate all reports concurrently
        days: list[date] = []
        current = week_start
        while current <= generation_end:
            days.append(current)
            current += timedelta(days=1)

        daily: list[dict[str, Any]] = await asyncio.gather(
            *(self.generate_daily_report(user_id, d) for d in days)
        )

        if not daily:
            return {
                "week_start": week_start.isoformat(),
                "week_end": week_end.isoformat(),
                "daily_reports": [],
                "averages": {},
                "trend": {},
                "week_number": week_start.isocalendar()[1],
                "intervention_effectiveness": None,
                "total_focus_minutes": 0.0,
                "total_sessions": 0,
                "total_distractions": 0,
                "avg_focus_score": 0.0,
                "daily_summary": [],
                "data_state": _compute_weekly_data_state(
                    week_start, today, week_end, daily
                ),
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
            # Frontend aliases
            "total_focus_minutes": round(avg_focus, 1),
            "total_sessions": len(daily),
            "total_distractions": sum(
                d.get("total_distractions", 0) for d in daily
            ),
            "avg_focus_score": round(avg_score, 1),
            "daily_summary": [
                {
                    "date": d.get("date"),
                    "focus_minutes": d.get("total_focus_minutes", d.get("total_focus_min", 0)),
                    "sessions": d.get("total_sessions", 0),
                    "distractions": d.get("total_distractions", 0),
                    "focus_score": d.get("focus_score"),
                }
                for d in daily
            ],
            "data_state": _compute_weekly_data_state(
                week_start, today, week_end, daily
            ),
        }

    # ── Scheduler convenience ─────────────────────────────────────────

    async def generate_daily_for_all(
        self,
        user_id: int = 1,
        *,
        target_date: date | None = None,
        refresh: bool = False,
    ) -> dict[str, Any]:
        """Generate the requested business day report for the desktop user."""
        resolved_date = target_date or business_today(self._timezone)
        return await self.generate_daily_report(
            user_id,
            resolved_date,
            refresh=refresh,
        )


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
