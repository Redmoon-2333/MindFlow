from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mindflow.train.__main__ import load_database_events, load_database_v2_data


def test_load_database_events_filters_future_and_invalid_rows(tmp_path: Path) -> None:
    database_path = tmp_path / "mindflow.db"
    connection = sqlite3.connect(database_path)
    connection.execute(
        "CREATE TABLE activity_events ("
        "id TEXT, user_id INTEGER, timestamp TEXT, duration_s REAL, "
        "event_type TEXT, data_json TEXT)"
    )
    now = datetime(2026, 7, 24, 12, tzinfo=UTC)
    payload = json.dumps({
        "app_name": "code.exe",
        "window_title": "main.py",
        "process_name": "code.exe",
        "is_idle": False,
        "timestamp_utc": now.isoformat(),
    })
    rows = [
        ("valid", 1, now.isoformat(), 5.0, "window_snapshot", payload),
        (
            "future",
            1,
            (now + timedelta(hours=1)).isoformat(),
            5.0,
            "window_snapshot",
            payload,
        ),
        (
            "negative",
            1,
            (now - timedelta(hours=1)).isoformat(),
            -1.0,
            "window_snapshot",
            payload,
        ),
    ]
    connection.executemany(
        "INSERT INTO activity_events VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    connection.commit()
    connection.close()

    events = load_database_events(database_path, now_utc=now)

    assert [event.id for event in events] == ["valid"]


def test_load_database_v2_data_joins_feedback_sessions(tmp_path: Path) -> None:
    database_path = tmp_path / "mindflow.db"
    connection = sqlite3.connect(database_path)
    connection.executescript(
        """
        CREATE TABLE behavior_feature_windows (
            id TEXT, user_id INTEGER, window_start_utc TEXT, window_end_utc TEXT,
            feature_schema_version INTEGER, features_json TEXT, label TEXT, created_at TEXT
        );
        CREATE TABLE focus_sessions (
            id TEXT, user_id INTEGER, date TEXT, start_time TEXT, end_time TEXT,
            session_type TEXT, dominant_app TEXT, focus_score REAL, switch_count INTEGER,
            created_at TEXT
        );
        CREATE TABLE focus_session_feedback (
            id TEXT, user_id INTEGER, session_id TEXT, label TEXT, score INTEGER,
            task_type TEXT, created_at TEXT
        );
        """
    )
    start = datetime(2026, 7, 24, 9, tzinfo=UTC)
    connection.execute(
        "INSERT INTO behavior_feature_windows VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "window",
            1,
            start.isoformat(),
            (start + timedelta(minutes=5)).isoformat(),
            2,
            json.dumps({"idle_ratio": 0.1}),
            None,
            start.isoformat(),
        ),
    )
    connection.execute(
        "INSERT INTO focus_sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "session",
            1,
            "2026-07-24",
            start.isoformat(),
            (start + timedelta(minutes=30)).isoformat(),
            "focus",
            "code.exe",
            90.0,
            0,
            start.isoformat(),
        ),
    )
    connection.execute(
        "INSERT INTO focus_session_feedback VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("feedback", 1, "session", "focus", 5, "coding", start.isoformat()),
    )
    connection.commit()
    connection.close()

    windows, feedback = load_database_v2_data(database_path)

    assert windows[0]["features"] == {"idle_ratio": 0.1}
    assert feedback[0]["session_id"] == "session"
    assert feedback[0]["start_time"] == start.isoformat()
