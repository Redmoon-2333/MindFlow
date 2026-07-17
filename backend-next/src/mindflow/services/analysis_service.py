"""Analysis service: focus session identification, pattern detection, profiling.

Ports session identification from the old ``mindflow/analyzer/patterns.py``
into the event-stream architecture (Wave 5).

Key differences from the old codebase:
  - Uses ``ActivityRepository.query_range()`` instead of raw SQLAlchemy ORM.
  - Sessions are grouped by consecutive same ``process_name`` events (after
    heartbeat merge).  Within a same-process group, the switch count is
    always 0 because heartbeat merge already combined consecutive ticks
    on the same application.  The old code exhibited the same behaviour.
  - UUIDv7 identifiers replace auto-increment integer PKs.
  - All timestamps are ISO8601-aware UTC (not naive).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from typing import Any

from loguru import logger

from mindflow.domain.events import ActivityEvent
from mindflow.domain.features import (
    _non_idle_events,
    app_usage_ranking,
)
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
)
from mindflow.infrastructure.repositories.focus import (
    SQLAlchemyFocusSessionRepository,
)

# ── Default session thresholds (preserved from old config) ────────────

_FOCUS_THRESHOLD_S: int = 300  # 5 minutes minimum for a focus block
_SWITCH_RATE_FOCUS_CUTOFF: float = 10.0
_SWITCH_RATE_DISTRACTION_CUTOFF: float = 30.0


class AnalysisService:
    """High-level analysis operations over the activity event stream.

    Args:
        activity_repo: Repository for reading activity events.
        focus_repo: Repository for writing / querying focus sessions.
        focus_threshold_s: Minimum duration (seconds) for a focus block.
        switch_rate_focus_cutoff: Switch rate below which a session
            is classified as ``focus``.
        switch_rate_distraction_cutoff: Switch rate above which a session
            is classified as ``distraction``.
    """

    def __init__(
        self,
        activity_repo: SQLAlchemyActivityRepository,
        focus_repo: SQLAlchemyFocusSessionRepository,
        focus_threshold_s: int = _FOCUS_THRESHOLD_S,
        switch_rate_focus_cutoff: float = _SWITCH_RATE_FOCUS_CUTOFF,
        switch_rate_distraction_cutoff: float = _SWITCH_RATE_DISTRACTION_CUTOFF,
    ) -> None:
        self._activity_repo = activity_repo
        self._focus_repo = focus_repo
        self._focus_threshold_s = focus_threshold_s
        self._switch_rate_focus_cutoff = switch_rate_focus_cutoff
        self._switch_rate_distraction_cutoff = switch_rate_distraction_cutoff

    # ── Focus session identification ──────────────────────────────────

    async def identify_focus_sessions(
        self,
        user_id: int,
        target_date: date,
    ) -> list[dict[str, Any]]:
        """Identify and persist focus sessions for *user_id* on *target_date*.

        Ported from ``patterns.identify_focus_sessions()``.

        **Algorithm:**
          1. Fetch all non-idle events for the day.
          2. Idempotency check — if sessions already exist for this date, skip.
          3. Group consecutive events by same ``process_name`` (after heartbeat
             merge, consecutive same-app rows represent long blocks).
          4. Groups with total duration >= ``_FOCUS_THRESHOLD_S`` (5 min) are
             promoted to focus sessions.
          5. Each session is classified as ``focus`` / ``distraction`` / ``neutral``
             based on switch rate within the window.  With the current heartbeat
             merge, switch rate inside a same-process group is always 0 → ``focus``.
          6. ``focus_score`` = min(total_duration / threshold * 100, 100).

        Args:
            user_id: User identifier.
            target_date: The calendar date to process.

        Returns:
            The list of persisted session dicts, or empty if skipped (idempotent).
        """
        start_dt = datetime(target_date.year, target_date.month, target_date.day, tzinfo=UTC)
        end_dt = start_dt + timedelta(days=1) - timedelta(seconds=1)

        events = await self._activity_repo.query_range(user_id, start_dt, end_dt)

        if len(events) < 2:
            logger.debug("Too few events on {} for user {}", target_date, user_id)
            return []

        # Fast-path idempotency check.  This is a performance optimisation
        # (skips computation when sessions already exist), not a correctness
        # guarantee.  The real guard against duplicate rows is the DELETE +
        # INSERT replace in SQLAlchemyFocusSessionRepository.save_sessions(),
        # which guarantees correctness even under concurrent callers.
        if await self._focus_repo.exists_for_date(user_id, target_date):
            logger.info("Sessions already exist for {} user {}", target_date, user_id)
            return []

        active = _non_idle(events)
        if len(active) < 2:
            return []

        sessions_data: list[dict[str, Any]] = []
        i = 0
        while i < len(active):
            current_proc = active[i].data.process_name
            j = i + 1
            while j < len(active) and active[j].data.process_name == current_proc:
                j += 1

            duration = sum(e.duration_s for e in active[i:j])

            if duration >= self._focus_threshold_s:
                window_events = active[i:j]
                local_switches = sum(
                    1
                    for k in range(1, len(window_events))
                    if window_events[k].data.process_name
                    != window_events[k - 1].data.process_name
                )
                local_hours = duration / 3600.0
                switch_rate = local_switches / local_hours if local_hours > 0 else 0.0

                if switch_rate < self._switch_rate_focus_cutoff:
                    session_type = "focus"
                elif switch_rate > self._switch_rate_distraction_cutoff:
                    session_type = "distraction"
                else:
                    session_type = "neutral"

                sessions_data.append({
                    "date": target_date.isoformat(),
                    "start_time": active[i].timestamp_utc.isoformat(),
                    "end_time": active[j - 1].timestamp_utc.isoformat(),
                    "session_type": session_type,
                    "dominant_app": current_proc,
                    "focus_score": round(min(duration / self._focus_threshold_s * 100.0, 100.0), 1),
                    "switch_count": local_switches,
                })

            i = j

        if not sessions_data:
            return []

        persisted = await self._focus_repo.save_sessions(user_id, sessions_data)
        logger.info(
            "Identified {} focus sessions for user {} on {}",
            len(persisted),
            user_id,
            target_date,
        )
        return persisted

    async def identify_all_today(self, user_id: int = 1) -> list[dict[str, Any]]:
        """Convenience wrapper for scheduled job — identifies sessions for today."""
        return await self.identify_focus_sessions(user_id, date.today())

    # ── Pattern detection ─────────────────────────────────────────────

    async def detect_patterns(
        self,
        user_id: int,
        days: int = 14,
    ) -> dict[str, Any]:
        """Detect distraction patterns from recent session history.

        Args:
            user_id: User identifier.
            days: Number of days of history to analyse (default 14).

        Returns:
            A dict with:
              - ``high_switch_periods``: list of (hour_label, switch_count).
              - ``trigger_apps``: apps that appear as ``dominant_app`` in
                distraction sessions, with counts.
              - ``heatmap``: 24x7 matrix of switch frequencies
                (list of 24 elements, each with 7 day-of-week buckets).
              - ``total_sessions``: total session count analysed.
              - ``distraction_ratio``: fraction of sessions that are distraction.
        """
        today = date.today()
        start = today - timedelta(days=days - 1)

        sessions = await self._focus_repo.query_range(user_id, start, today)

        if not sessions:
            return {
                "high_switch_periods": [],
                "trigger_apps": [],
                "heatmap": _empty_heatmap(),
                "total_sessions": 0,
                "distraction_ratio": 0.0,
            }

        # Hourly switch aggregation
        hourly_switches: dict[int, int] = defaultdict(int)
        trigger_apps: dict[str, int] = defaultdict(int)
        # 24 hours x 7 days
        heatmap: list[list[int]] = [[0] * 7 for _ in range(24)]
        distraction_count = 0

        for s in sessions:
            try:
                start_ts = datetime.fromisoformat(s["start_time"])
                hour = start_ts.hour
                dow = start_ts.weekday()  # 0=Monday, 6=Sunday
                heatmap[hour][dow] += s.get("switch_count", 0)

                if s.get("session_type") == "distraction":
                    distraction_count += 1
                    app = s.get("dominant_app")
                    if app:
                        trigger_apps[app] += 1

                hourly_switches[hour] += s.get("switch_count", 0)
            except (ValueError, KeyError):
                continue

        # Sort high-switch periods
        sorted_hours = sorted(hourly_switches.items(), key=lambda x: x[1], reverse=True)
        high_switch = [
            {"hour": h, "switch_count": c}
            for h, c in sorted_hours[:5]
            if c > 0
        ]

        sorted_triggers = sorted(trigger_apps.items(), key=lambda x: x[1], reverse=True)
        top_triggers = [
            {"app": app, "count": cnt}
            for app, cnt in sorted_triggers[:10]
        ]

        return {
            "high_switch_periods": high_switch,
            "trigger_apps": top_triggers,
            "heatmap": heatmap,
            "total_sessions": len(sessions),
            "distraction_ratio": round(
                distraction_count / len(sessions) if sessions else 0.0, 3
            ),
        }

    # ── Behavioural profile ───────────────────────────────────────────

    async def behavioral_profile(
        self,
        user_id: int,
        days: int = 30,
    ) -> dict[str, Any]:
        """Build a behavioural profile for *user_id* from recent events and sessions.

        Args:
            user_id: User identifier.
            days: Days of history to consider (default 30).

        Returns:
            A dict with:
              - ``peak_focus_hours``: list of hours (0-23) sorted by avg focus.
              - ``top_apps``: top productive apps by total duration.
              - ``avg_focus_block_min``: average focus session length in minutes.
              - ``distraction_triggers``: apps tied to distraction sessions.
              - ``total_events_analysed``: raw event count in the window.
              - ``profile_date``: ISO date of computation.
        """
        today = date.today()
        start = today - timedelta(days=days - 1)
        start_dt = datetime(start.year, start.month, start.day, tzinfo=UTC)
        end_dt = datetime(today.year, today.month, today.day, tzinfo=UTC) + timedelta(days=1)

        events = await self._activity_repo.query_range(user_id, start_dt, end_dt)
        sessions = await self._focus_repo.query_range(user_id, start, today)

        if not events:
            return {
                "peak_focus_hours": [],
                "top_apps": [],
                "avg_focus_block_min": 0.0,
                "distraction_triggers": [],
                "total_events_analysed": 0,
                "profile_date": today.isoformat(),
            }

        # Peak focus hours: analyse focus scores per hour from sessions
        hour_focus: dict[int, list[float]] = defaultdict(list)
        for s in sessions:
            if s.get("session_type") == "focus":
                try:
                    start_ts = datetime.fromisoformat(s["start_time"])
                    hour_focus[start_ts.hour].append(s.get("focus_score", 0) or 0)
                except (ValueError, KeyError):
                    continue

        avg_focus_per_hour = {
            h: sum(scores) / len(scores) for h, scores in hour_focus.items()
        }
        peak_hours = sorted(
            avg_focus_per_hour,
            key=lambda h: avg_focus_per_hour[h],
            reverse=True,
        )[:5]

        # Top apps
        usage = app_usage_ranking(events)
        top_apps = [
            {"app": u.app_name, "total_min": round(u.total_duration_s / 60.0, 1)}
            for u in usage[:10]
        ]

        # Average focus block length
        focus_sessions = [s for s in sessions if s.get("session_type") == "focus"]
        if focus_sessions:
            total_min = 0.0
            for s in focus_sessions:
                start_ts = datetime.fromisoformat(s["start_time"])
                end_ts = datetime.fromisoformat(s["end_time"])
                total_min += (end_ts - start_ts).total_seconds() / 60.0
            avg_block = round(total_min / len(focus_sessions), 1)
        else:
            avg_block = 0.0

        # Distraction triggers from sessions
        trigger_counts: dict[str, int] = defaultdict(int)
        for s in sessions:
            if s.get("session_type") == "distraction":
                app = s.get("dominant_app")
                if app:
                    trigger_counts[app] += 1
        sorted_triggers = sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True)
        distraction_triggers = [
            {"app": app, "count": cnt} for app, cnt in sorted_triggers[:5]
        ]

        peak_hours_list = [
            {"hour": h, "avg_score": round(avg_focus_per_hour[h], 1)}
            for h in peak_hours
        ]

        return {
            "peak_focus_hours": peak_hours_list,
            "top_apps": top_apps,
            "avg_focus_block_min": avg_block,
            "distraction_triggers": distraction_triggers,
            "total_events_analysed": len(events),
            "profile_date": today.isoformat(),
        }


# ── Internal helpers ──────────────────────────────────────────────────


def _non_idle(events: list[ActivityEvent]) -> list[ActivityEvent]:
    """Filter out idle events, preserving order.

    Thin alias over the domain helper — kept as a module-local name for
    call-site brevity; single implementation lives in domain.features
    (slop-scan dedup).
    """
    return _non_idle_events(events)


def _empty_heatmap() -> list[list[int]]:
    """Return a zero-filled 24x7 matrix."""
    return [[0] * 7 for _ in range(24)]
