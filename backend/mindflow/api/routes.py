import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
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
from mindflow.analyzer.data_pipeline import BehaviorFeatureExtractor
from mindflow.analyzer.baseline import BaselineModel
from mindflow.analyzer.deviation import DeviationDetector
from mindflow.logging_config import get_logger

logger = get_logger(__name__)
_MODELS_DIR = Path(__file__).resolve().parents[2] / "data" / "models"


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
    return db.query(User).order_by(User.id).first()


def _activities_to_df(db: Session, user_id: int, days: int = 1) -> pd.DataFrame:
    """Convert recent ActivityLog records into a DataFrame for feature extraction."""
    start_dt = datetime.combine(date.today() - timedelta(days=days - 1), datetime.min.time())
    activities = (
        db.query(ActivityLog)
        .filter(
            ActivityLog.user_id == user_id,
            ActivityLog.timestamp >= start_dt,
        )
        .order_by(ActivityLog.timestamp.asc())
        .all()
    )
    if not activities:
        return pd.DataFrame()
    rows = [{
        "timestamp": a.timestamp,
        "process_name": a.process_name,
        "window_title": a.window_title or "",
        "duration_seconds": a.duration_seconds,
        "is_idle": a.is_idle,
    } for a in activities]
    return pd.DataFrame(rows)


def _extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run feature extraction on activity DataFrame."""
    extractor = BehaviorFeatureExtractor(window_minutes=30)
    return extractor.extract_session_features(df)


def _load_baseline() -> Optional[BaselineModel]:
    path = _MODELS_DIR / "baseline_user1.json"
    if not path.exists():
        return None
    try:
        return BaselineModel.load(path)
    except FileNotFoundError:
        return None
    except Exception:
        logger.warning("Failed to load baseline model", exc_info=True)
        return None


def _load_clustering() -> Optional[object]:
    path = _MODELS_DIR / "clustering.joblib"
    if not path.exists():
        return None
    try:
        from mindflow.analyzer.ml_models import BehaviorClustering
        return BehaviorClustering.load(path)
    except FileNotFoundError:
        return None
    except Exception:
        logger.warning("Failed to load clustering model", exc_info=True)
        return None


def _load_hmm() -> Optional[object]:
    path = _MODELS_DIR / "hmm.joblib"
    if not path.exists():
        return None
    try:
        import joblib
        import numpy as np
        from mindflow.analyzer.ml_models import BehaviorHMM
        data = joblib.load(str(path))
        hmm = BehaviorHMM(n_states=int(data.get("n_states", 5)))
        hmm.transition_matrix = np.array(data.get("transition_matrix"))
        hmm._is_fitted = True
        return hmm
    except FileNotFoundError:
        return None
    except Exception:
        logger.warning("Failed to load HMM model", exc_info=True)
        return None


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
        return _err(40001, "未找到用户")

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
        return _err(40001, "未找到用户")

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
        return _err(40001, "未找到用户")

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
        return _err(40001, "未找到用户")

    return _ok({
        "username": user.username,
        "preferences": user.preferences or {},
        "created_at": user.created_at.isoformat() if user.created_at else None,
    })


@router.put("/preferences")
async def update_preferences(prefs: PreferencesUpdate, db: Session = Depends(get_db)):
    user = _get_default_user(db)
    if not user:
        return _err(40001, "未找到用户")

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
        return _err(40001, "未找到用户")

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
        return _err(40001, "未找到用户")

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


# ── Health ──────────────────────────────────────────────────────────────────


@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    db_ok = True
    try:
        db.query(User.id).limit(1).all()
    except Exception:
        db_ok = False

    models_available = {
        "baseline": (_MODELS_DIR / "baseline_user1.json").exists(),
        "clustering": (_MODELS_DIR / "clustering.joblib").exists(),
        "classifier": (_MODELS_DIR / "classifier.joblib").exists(),
        "hmm": (_MODELS_DIR / "hmm.joblib").exists(),
    }

    return _ok({
        "status": "healthy" if db_ok else "degraded",
        "database": "connected" if db_ok else "error",
        "collector_running": collector.is_running,
        "models_available": models_available,
    })


# ── Data & Privacy ───────────────────────────────────────────────────────────


@router.get("/data/summary")
async def data_summary(db: Session = Depends(get_db)):
    user = _get_default_user(db)
    if not user:
        return _err(40001, "未找到用户")

    total = db.query(ActivityLog).count()
    first = db.query(ActivityLog).order_by(ActivityLog.timestamp.asc()).first()
    last = db.query(ActivityLog).order_by(ActivityLog.timestamp.desc()).first()

    db_path = Path(settings.database_url.replace("sqlite:///", ""))
    if not db_path.is_absolute():
        db_path = Path(__file__).resolve().parents[2] / db_path
    db_size_bytes = db_path.stat().st_size if db_path.exists() else 0

    return _ok({
        "total_activities": total,
        "earliest_record": first.timestamp.isoformat() if first else None,
        "latest_record": last.timestamp.isoformat() if last else None,
        "database_size_mb": round(db_size_bytes / (1024 * 1024), 2),
        "database_path": str(db_path),
        "storage_location": "本地 SQLite，数据不会上传到任何服务器",
    })


# ── ML Analytics ─────────────────────────────────────────────────────────────


@router.get("/analytics/deviation")
async def analytics_deviation(db: Session = Depends(get_db)):
    user = _get_default_user(db)
    if not user:
        return _err(40001, "未找到用户")

    baseline = _load_baseline()
    if baseline is None:
        return _ok({
            "model_available": False,
            "message": "基线模型尚未训练，请先采集数据后运行：python -m mindflow.analyzer.train --from-db",
            "anomalies": [],
            "daily_summary": {},
        })

    df = _activities_to_df(db, user.id, days=1)
    if df.empty:
        return _ok({
            "model_available": True,
            "message": "今天还没有活动数据",
            "anomalies": [],
            "daily_summary": {"total_windows": 0, "anomaly_count": 0},
        })

    features_df = _extract_features(df)
    if features_df.empty:
        return _ok({
            "model_available": True,
            "message": "数据不足，无法提取特征窗口",
            "anomalies": [],
            "daily_summary": {},
        })

    detector = DeviationDetector(baseline)
    anomalies = detector.analyze_dataframe(features_df)
    daily = detector.daily_summary(features_df)

    return _ok({
        "model_available": True,
        "anomalies": anomalies,
        "daily_summary": daily,
    })


@router.get("/analytics/clusters")
async def analytics_clusters(db: Session = Depends(get_db)):
    user = _get_default_user(db)
    if not user:
        return _err(40001, "未找到用户")

    clustering = _load_clustering()
    if clustering is None:
        return _ok({
            "model_available": False,
            "message": "聚类模型尚未训练，请先采集数据后运行：python -m mindflow.analyzer.train --from-db",
            "clusters": [],
        })

    df = _activities_to_df(db, user.id, days=1)
    if df.empty:
        return _ok({
            "model_available": True,
            "message": "今天还没有活动数据",
            "clusters": [],
        })

    features_df = _extract_features(df)
    if features_df.empty:
        return _ok({
            "model_available": True,
            "message": "数据不足，无法提取特征窗口",
            "clusters": [],
        })

    feature_cols = [
        c for c in features_df.columns
        if c not in ("window_start",)
        and features_df[c].dtype in ("float64", "float32", "int64", "int32")
    ]
    X = features_df[feature_cols].to_numpy(dtype="float64")

    try:
        pred_labels = clustering.predict(X)
    except Exception:
        pred_labels = None

    cluster_info = clustering._cluster_info
    clusters = [
        {
            "id": c.cluster_id,
            "label": c.label,
            "sample_count": c.sample_count,
            "avg_focus_score": c.avg_focus_score,
        }
        for c in cluster_info
    ]

    distribution: dict[str, int] = {}
    if pred_labels is not None:
        for label in pred_labels:
            name = clustering.CLUSTER_LABEL_MAP.get(int(label), f"cluster_{label}")
            distribution[name] = distribution.get(name, 0) + 1

    return _ok({
        "model_available": True,
        "clusters": clusters,
        "today_distribution": distribution,
    })


@router.get("/analytics/risk")
async def analytics_risk(db: Session = Depends(get_db)):
    user = _get_default_user(db)
    if not user:
        return _err(40001, "未找到用户")

    hmm = _load_hmm()
    if hmm is None or hmm.transition_matrix is None:
        return _ok({
            "model_available": False,
            "message": "HMM 模型尚未训练，请先采集数据后运行：python -m mindflow.analyzer.train --from-db",
            "current_state": None,
            "risk_level": "unknown",
        })

    df = _activities_to_df(db, user.id, days=1)
    if df.empty:
        return _ok({
            "model_available": True,
            "message": "今天还没有活动数据",
            "current_state": None,
            "risk_level": "unknown",
        })

    features_df = _extract_features(df)
    if features_df.empty or len(features_df) < 2:
        return _ok({
            "model_available": True,
            "message": "特征窗口不足，无法进行状态估计",
            "current_state": None,
            "risk_level": "unknown",
        })

    clustering = _load_clustering()
    if clustering is not None:
        feature_cols = [
            c for c in features_df.columns
            if c not in ("window_start",)
            and features_df[c].dtype in ("float64", "float32", "int64", "int32")
        ]
        X = features_df[feature_cols].to_numpy(dtype="float64")
        try:
            states = clustering.predict(X)
            current_state = int(states[-1]) if len(states) > 0 else 0
        except Exception:
            states = None
            current_state = 0
    else:
        states = None
        current_state = 0

    prediction = hmm.predict_next_state(current_state)
    next_name = prediction["next_state_name"]

    risk_map = {
        "deep_focus": "low",
        "shallow_work": "low",
        "browsing": "medium",
        "procrastination": "high",
        "idle": "medium",
    }
    risk_level = risk_map.get(next_name, "unknown")

    steady = hmm.get_steady_state()
    steady_dist = {
        name: round(float(p), 4)
        for name, p in zip(hmm.state_names, steady)
    }

    return _ok({
        "model_available": True,
        "current_state": hmm.state_names[current_state] if current_state < len(hmm.state_names) else "unknown",
        "current_state_id": current_state,
        "predicted_next_state": next_name,
        "transition_probabilities": {
            name: prob
            for name, prob in zip(hmm.state_names, prediction["probabilities"])
        },
        "risk_level": risk_level,
        "steady_state_distribution": steady_dist,
    })
