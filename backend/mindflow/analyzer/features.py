from datetime import date, datetime

from sqlalchemy import func
from sqlalchemy.orm import Session

from mindflow.models.schemas import ActivityLog

MIN_ACTIVITY_THRESHOLD = 10
MIN_SWITCH_SAMPLES = 2
MAX_ACCEPTABLE_SWITCHES_PER_HOUR = 30.0
FOCUS_TOP_APP_WEIGHT = 60.0
FOCUS_SWITCH_WEIGHT = 40.0


def query_day_activities(db: Session, user_id: int, target_date: date) -> list[ActivityLog]:
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


def calculate_switch_frequency(
    db: Session, user_id: int, target_date: date,
    activities: list[ActivityLog] | None = None,
) -> float:
    if activities is None:
        activities = query_day_activities(db, user_id, target_date)
    if len(activities) < MIN_SWITCH_SAMPLES:
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
            func.sum(ActivityLog.duration_seconds).label("total_seconds"),
        )
        .filter(
            ActivityLog.user_id == user_id,
            ActivityLog.timestamp >= start_dt,
            ActivityLog.timestamp <= end_dt,
            ActivityLog.is_idle == 0,
        )
        .group_by(ActivityLog.process_name)
        .order_by(func.sum(ActivityLog.duration_seconds).desc())
        .limit(limit)
        .all()
    )

    return [
        {
            "app": row.process_name,
            "minutes": round(float(row.total_seconds or 0) / 60.0, 1),
        }
        for row in results
    ]


def calculate_focus_score(db: Session, user_id: int, target_date: date) -> float:
    activities = query_day_activities(db, user_id, target_date)
    if len(activities) < MIN_ACTIVITY_THRESHOLD:
        return 0.0

    app_durations: dict[str, float] = {}
    for act in activities:
        app_durations[act.process_name] = (
            app_durations.get(act.process_name, 0.0) + act.duration_seconds
        )

    if not app_durations:
        return 0.0

    total_duration = sum(app_durations.values())
    top_app_ratio = max(app_durations.values()) / total_duration if total_duration > 0 else 0

    switch_freq = calculate_switch_frequency(db, user_id, target_date, activities=activities)
    switch_penalty = min(switch_freq / MAX_ACCEPTABLE_SWITCHES_PER_HOUR, 1.0)

    raw_score = (top_app_ratio * FOCUS_TOP_APP_WEIGHT) + ((1.0 - switch_penalty) * FOCUS_SWITCH_WEIGHT)
    return round(min(max(raw_score, 0.0), 100.0), 1)
