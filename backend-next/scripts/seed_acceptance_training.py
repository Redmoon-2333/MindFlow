"""Seed explicit schema-v4 training feedback only in a disposable acceptance DB."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import insert

from mindflow.infrastructure.database import create_engine, create_session_factory
from mindflow.infrastructure.repositories.focus import focus_sessions
from mindflow.infrastructure.repositories.telemetry import TelemetryRepository
from mindflow.infrastructure.schema import activity_events, focus_session_feedback
from scripts.acceptance_publication_guard import _build_separable_dataset


async def seed(data_dir: Path) -> None:
    windows, feedback = _build_separable_dataset()
    engine = create_engine(f"sqlite+aiosqlite:///{(data_dir / 'mindflow.db').as_posix()}")
    try:
        repository = TelemetryRepository(create_session_factory(engine))
        await repository.upsert_feature_windows([
            {
                "user_id": 1,
                "window_start_utc": datetime.fromisoformat(row["window_start_utc"]),
                "window_end_utc": datetime.fromisoformat(row["window_end_utc"]),
                "feature_schema_version": row["feature_schema_version"],
                "features_json": json.dumps(row["features"]),
                "label": None,
            }
            for row in windows
        ])
        async with engine.begin() as conn:
            for row in feedback:
                session_id = f"release-{row['session_id']}"
                await conn.execute(insert(focus_sessions).values(
                    id=session_id, user_id=1, date=row["start_time"][:10],
                    start_time=row["start_time"], end_time=row["end_time"],
                    session_type="focus", dominant_app="acceptance-code.exe",
                    focus_score=90.0, switch_count=0,
                ))
                await conn.execute(insert(focus_session_feedback).values(
                    id=f"feedback-{session_id}", user_id=1, session_id=session_id,
                    label=row["label"], score=row["score"], task_type=row["task_type"],
                    created_at=row["start_time"],
                ))
            now = datetime.now(UTC)
            for index in range(36):
                timestamp = now - timedelta(seconds=(36 - index) * 40)
                app = "wechat.exe" if index % 2 else "chrome.exe"
                await conn.execute(insert(activity_events).values(
                    id=f"release-recent-{index}", user_id=1,
                    timestamp=timestamp.isoformat(), duration_s=40.0,
                    event_type="window_snapshot",
                    data_json=json.dumps({
                        "app_name": app, "process_name": app,
                        "window_title": "synthetic acceptance fixture",
                        "is_idle": False, "timestamp_utc": timestamp.isoformat(),
                    }),
                ))
        print(f"Seeded {len(windows)} v4 windows, {len(feedback)} feedback sessions, "
              "and 36 synthetic recent events")
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    if (
        Path(tempfile.gettempdir()).resolve() not in data_dir.parents
        or not data_dir.name.startswith("mindflow-full-")
        or not (data_dir / "mindflow.db").is_file()
    ):
        parser.error("Use an initialized mindflow-full-* database below system temp")
    asyncio.run(seed(data_dir))


if __name__ == "__main__":
    main()
