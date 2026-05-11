from datetime import date, datetime

from sqlalchemy.orm import Session

from mindflow.models.schemas import ActivityLog, FocusSession, DailyReport
from mindflow.analyzer.features import (
    calculate_focus_score,
    get_top_apps,
    calculate_switch_frequency,
    query_day_activities,
)
from mindflow.config import settings


def identify_focus_sessions(db: Session, user_id: int, target_date: date) -> list[FocusSession]:
    start_dt = datetime.combine(target_date, datetime.min.time())
    end_dt = datetime.combine(target_date, datetime.max.time())

    activities = query_day_activities(db, user_id, target_date)

    if len(activities) < 2:
        return []

    existing = (
        db.query(FocusSession)
        .filter(
            FocusSession.user_id == user_id,
            FocusSession.start_time >= start_dt,
            FocusSession.start_time <= end_dt,
        )
        .count()
    )
    if existing > 0:
        return []

    focus_threshold = settings.focus_threshold_minutes * 60
    sessions: list[FocusSession] = []

    i = 0
    while i < len(activities):
        current_app = activities[i].process_name
        j = i + 1
        while j < len(activities) and activities[j].process_name == current_app:
            j += 1

        duration = sum(a.duration_seconds for a in activities[i:j])

        if duration >= focus_threshold:
            window_activities = activities[i:j]
            local_switches = sum(
                1 for k in range(1, len(window_activities))
                if window_activities[k].process_name != window_activities[k - 1].process_name
            )
            local_hours = duration / 3600.0
            switch_rate = local_switches / local_hours if local_hours > 0 else 0

            if switch_rate < 10:
                session_type = "focus"
            elif switch_rate > 30:
                session_type = "distraction"
            else:
                session_type = "neutral"

            session = FocusSession(
                user_id=user_id,
                start_time=activities[i].timestamp,
                end_time=activities[j - 1].timestamp,
                focus_score=min(duration / focus_threshold * 100.0, 100.0),
                session_type=session_type,
                dominant_app=current_app,
            )
            db.add(session)
            sessions.append(session)

        i = j

    if sessions:
        db.commit()
    return sessions


def generate_daily_report(db: Session, user_id: int, target_date: date) -> DailyReport:
    existing = (
        db.query(DailyReport)
        .filter(
            DailyReport.user_id == user_id,
            DailyReport.date == target_date,
        )
        .first()
    )
    if existing:
        return existing

    sessions = identify_focus_sessions(db, user_id, target_date)

    total_focus = 0.0
    total_distraction = 0.0
    for session in sessions:
        if session.end_time and session.start_time:
            duration_min = (session.end_time - session.start_time).total_seconds() / 60.0
        else:
            duration_min = 0.0
        if session.session_type == "focus":
            total_focus += duration_min
        elif session.session_type == "distraction":
            total_distraction += duration_min

    focus_score = calculate_focus_score(db, user_id, target_date)
    top_apps = get_top_apps(db, user_id, target_date, limit=10)
    switch_freq = calculate_switch_frequency(db, user_id, target_date)

    report = DailyReport(
        user_id=user_id,
        date=target_date,
        total_focus_minutes=round(total_focus, 1),
        total_distraction_minutes=round(total_distraction, 1),
        focus_score=focus_score,
        top_apps=top_apps,
        switch_frequency=round(switch_freq, 2),
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report
