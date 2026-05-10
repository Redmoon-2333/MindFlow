import time
from datetime import date, datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel

from mindflow.models.database import get_db
from mindflow.models.schemas import User, ActivityLog, DailyReport
from mindflow.config import settings
from mindflow.collector.tracker import get_active_window_info
from mindflow.collector.scheduler import collector
from mindflow.analyzer.features import calculate_focus_score, get_top_apps
from mindflow.analyzer.patterns import generate_daily_report


router = APIRouter(prefix="/api/v1")


def _ok(data=None, message: str = "success"):
    return {
        "code": 0,
        "message": message,
        "data": data,
        "timestamp": int(time.time()),
    }


def _err(code: int, message: str):
    return {
        "code": code,
        "message": message,
        "data": None,
        "timestamp": int(time.time()),
    }


def _get_default_user(db: Session) -> Optional[User]:
    return db.query(User).first()


@router.get("/status")
async def get_status(db: Session = Depends(get_db)):
    user = _get_default_user(db)
    activity_count = db.query(ActivityLog).count()
    return _ok({
        "collector_running": collector.is_running,
        "user_exists": user is not None,
        "total_activities": activity_count,
        "settings": {
            "collect_interval_seconds": settings.collect_interval_seconds,
            "idle_threshold_seconds": settings.idle_threshold_seconds,
            "focus_threshold_minutes": settings.focus_threshold_minutes,
        },
    })


@router.post("/collector/start")
async def collector_start():
    collector.start()
    return _ok({"collector_running": collector.is_running})


@router.post("/collector/stop")
async def collector_stop():
    collector.stop()
    return _ok({"collector_running": collector.is_running})


@router.get("/activities/today")
async def activities_today(db: Session = Depends(get_db)):
    user = _get_default_user(db)
    if not user:
        return _err(40001, "No user found")

    today = date.today()
    today_start = datetime.combine(today, datetime.min.time())
    today_end = datetime.combine(today, datetime.max.time())

    total = (
        db.query(ActivityLog)
        .filter(
            ActivityLog.user_id == user.id,
            ActivityLog.timestamp >= today_start,
            ActivityLog.timestamp <= today_end,
        )
        .count()
    )

    top_apps = get_top_apps(db, user.id, today, limit=10)
    focus_score = calculate_focus_score(db, user.id, today)

    return _ok({
        "date": today.isoformat(),
        "total_activities": total,
        "top_apps": top_apps,
        "focus_score": focus_score,
    })


@router.get("/activities/current")
async def activities_current():
    info = get_active_window_info()
    return _ok({
        "window": info,
        "collector_running": collector.is_running,
    })


@router.get("/focus/today")
async def focus_today(db: Session = Depends(get_db)):
    user = _get_default_user(db)
    if not user:
        return _err(40001, "No user found")

    today = date.today()
    report = generate_daily_report(db, user.id, today)

    return _ok({
        "date": today.isoformat(),
        "total_focus_minutes": report.total_focus_minutes,
        "total_distraction_minutes": report.total_distraction_minutes,
        "focus_score": report.focus_score,
        "top_apps": report.top_apps,
        "switch_frequency": report.switch_frequency,
    })


@router.get("/focus/trend")
async def focus_trend(days: int = Query(default=7, ge=1, le=90), db: Session = Depends(get_db)):
    user = _get_default_user(db)
    if not user:
        return _err(40001, "No user found")

    today = date.today()
    trend = []

    for i in range(days):
        d = today - timedelta(days=i)
        report = (
            db.query(DailyReport)
            .filter(
                DailyReport.user_id == user.id,
                DailyReport.date == d,
            )
            .first()
        )
        if report:
            trend.append({
                "date": d.isoformat(),
                "focus_score": report.focus_score,
                "focus_minutes": report.total_focus_minutes,
                "distraction_minutes": report.total_distraction_minutes,
            })
        else:
            trend.append({
                "date": d.isoformat(),
                "focus_score": 0,
                "focus_minutes": 0,
                "distraction_minutes": 0,
            })

    return _ok({
        "days": days,
        "trend": list(reversed(trend)),
    })


class PreferencesUpdate(BaseModel):
    collect_interval_seconds: Optional[int] = None
    focus_threshold_minutes: Optional[int] = None
    idle_threshold_seconds: Optional[int] = None
    intervention_intensity: Optional[str] = None


@router.get("/preferences")
async def get_preferences(db: Session = Depends(get_db)):
    user = _get_default_user(db)
    if not user:
        return _err(40001, "No user found")

    return _ok({
        "username": user.username,
        "preferences": user.preferences or {},
        "created_at": user.created_at.isoformat() if user.created_at else None,
    })


@router.put("/preferences")
async def update_preferences(prefs: PreferencesUpdate, db: Session = Depends(get_db)):
    user = _get_default_user(db)
    if not user:
        return _err(40001, "No user found")

    current_prefs = user.preferences or {}
    update_data = prefs.model_dump(exclude_unset=True)
    current_prefs.update(update_data)
    user.preferences = current_prefs
    db.commit()
    db.refresh(user)

    return _ok({
        "username": user.username,
        "preferences": user.preferences,
    })


@router.get("/analytics/patterns")
async def analytics_patterns(db: Session = Depends(get_db)):
    user = _get_default_user(db)
    if not user:
        return _err(40001, "No user found")

    from mindflow.models.schemas import FocusSession
    today = date.today()
    today_start = datetime.combine(today, datetime.min.time())

    sessions = (
        db.query(FocusSession)
        .filter(
            FocusSession.user_id == user.id,
            FocusSession.start_time >= today_start,
        )
        .all()
    )

    return _ok({
        "date": today.isoformat(),
        "sessions": [
            {
                "start_time": s.start_time.isoformat() if s.start_time else None,
                "end_time": s.end_time.isoformat() if s.end_time else None,
                "focus_score": s.focus_score,
                "session_type": s.session_type,
                "dominant_app": s.dominant_app,
            }
            for s in sessions
        ],
    })


@router.get("/reports/weekly")
async def reports_weekly(db: Session = Depends(get_db)):
    user = _get_default_user(db)
    if not user:
        return _err(40001, "No user found")

    today = date.today()
    reports = (
        db.query(DailyReport)
        .filter(
            DailyReport.user_id == user.id,
            DailyReport.date >= today - timedelta(days=7),
        )
        .order_by(DailyReport.date.asc())
        .all()
    )

    return _ok({
        "reports": [
            {
                "date": r.date.isoformat() if r.date else None,
                "focus_score": r.focus_score,
                "focus_minutes": r.total_focus_minutes,
                "distraction_minutes": r.total_distraction_minutes,
                "switch_frequency": r.switch_frequency,
            }
            for r in reports
        ],
    })
