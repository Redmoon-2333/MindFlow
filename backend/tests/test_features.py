from datetime import date, datetime

from mindflow.models.database import SessionLocal, init_db
from mindflow.models.schemas import User, ActivityLog
from mindflow.analyzer.features import (
    calculate_focus_score,
    get_top_apps,
    calculate_switch_frequency,
)


def _setup_test_data():
    init_db()
    db = SessionLocal()
    try:
        user = db.query(User).first()
        if user is None:
            user = User(username="test_user", preferences={})
            db.add(user)
            db.commit()
            db.refresh(user)

        existing = db.query(ActivityLog).filter(
            ActivityLog.user_id == user.id,
            ActivityLog.process_name == "vscode.exe",
        ).count()
        if existing > 0:
            return user.id

        today = date.today()
        base_ts = datetime.combine(today, datetime.min.time()).replace(hour=9)
        for i in range(120):
            ts = datetime.fromtimestamp(base_ts.timestamp() + i * 5)
            activity = ActivityLog(
                user_id=user.id,
                timestamp=ts,
                process_name="vscode.exe",
                window_title="test.py - Visual Studio Code",
                window_class="",
                duration_seconds=5,
                is_idle=0,
            )
            db.add(activity)

        for i in range(120, 150):
            ts = datetime.fromtimestamp(base_ts.timestamp() + i * 5)
            activity = ActivityLog(
                user_id=user.id,
                timestamp=ts,
                process_name="chrome.exe",
                window_title="YouTube",
                window_class="",
                duration_seconds=5,
                is_idle=0,
            )
            db.add(activity)

        db.commit()
        return user.id
    finally:
        db.close()


def test_calculate_focus_score():
    user_id = _setup_test_data()
    db = SessionLocal()
    try:
        score = calculate_focus_score(db, user_id, date.today())
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0
    finally:
        db.close()


def test_get_top_apps():
    user_id = _setup_test_data()
    db = SessionLocal()
    try:
        apps = get_top_apps(db, user_id, date.today(), limit=10)
        assert isinstance(apps, list)
        if apps:
            assert "app" in apps[0]
            assert "minutes" in apps[0]
    finally:
        db.close()


def test_calculate_switch_frequency():
    user_id = _setup_test_data()
    db = SessionLocal()
    try:
        freq = calculate_switch_frequency(db, user_id, date.today())
        assert isinstance(freq, float)
    finally:
        db.close()


def test_focus_score_no_data():
    db = SessionLocal()
    try:
        score = calculate_focus_score(db, 99999, date.today())
        assert score == 0.0
    finally:
        db.close()
