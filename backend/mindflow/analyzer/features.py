from datetime import date, datetime, timedelta
from typing import Optional

from sqlalchemy import func, text
from sqlalchemy.orm import Session

from mindflow.models.schemas import ActivityLog
from mindflow.config import settings


def _get_day_activities(db: Session, user_id: int, target_date: date):
    start_dt = datetime.combine(target_date, datetime.min.time())
    end_dt = datetime.combine(target_date, datetime.max.time())
    return (
        db.query(ActivityLog)
        .filter(
            ActivityLog.user_id == user_id,
            ActivityLog.timestamp >= start_dt,
            ActivityLog.timestamp <= end_dt,
            ActivityLog.is_idle == 0,
        )
        .order_by(ActivityLog.timestamp.asc())
        .all()
    )


def calculate_switch_frequency(db: Session, user_id: int, target_date: date) -> float:
    activities = _get_day_activities(db, user_id, target_date)
    if len(activities) < 2:
        return 0.0

    switches = 0
    prev_app = activities[0].process_name
    for act in activities[1:]:
        if act.process_name != prev_app:
            switches += 1
        prev_app = act.process_name

    first_ts = activities[0].timestamp
    last_ts = activities[-1].timestamp
    total_hours = (last_ts - first_ts).total_seconds() / 3600.0
    if total_hours <= 0:
        return 0.0

    return switches / total_hours


def get_top_apps(db: Session, user_id: int, target_date: date, limit: int = 10) -> list[dict]:
    start_dt = datetime.combine(target_date, datetime.min.time())
    end_dt = datetime.combine(target_date, datetime.max.time())

    results = (
        db.query(
            ActivityLog.process_name,
            func.count(ActivityLog.id).label("sample_count"),
        )
        .filter(
            ActivityLog.user_id == user_id,
            ActivityLog.timestamp >= start_dt,
            ActivityLog.timestamp <= end_dt,
            ActivityLog.is_idle == 0,
        )
        .group_by(ActivityLog.process_name)
        .order_by(func.count(ActivityLog.id).desc())
        .limit(limit)
        .all()
    )

    return [
        {
            "app": row.process_name,
            "minutes": round(row.sample_count * settings.collect_interval_seconds / 60.0, 1),
        }
        for row in results
    ]


def calculate_focus_score(db: Session, user_id: int, target_date: date) -> float:
    activities = _get_day_activities(db, user_id, target_date)
    if len(activities) < 10:
        return 0.0

    app_durations: dict[str, float] = {}
    for act in activities:
        app_durations[act.process_name] = (
            app_durations.get(act.process_name, 0.0) + settings.collect_interval_seconds
        )

    if not app_durations:
        return 0.0

    total_duration = sum(app_durations.values())
    top_app_ratio = max(app_durations.values()) / total_duration if total_duration > 0 else 0

    switch_freq = calculate_switch_frequency(db, user_id, target_date)
    max_acceptable_switches = 30.0
    switch_penalty = min(switch_freq / max_acceptable_switches, 1.0)

    raw_score = (top_app_ratio * 60.0) + ((1.0 - switch_penalty) * 40.0)
    return round(min(max(raw_score, 0.0), 100.0), 1)
