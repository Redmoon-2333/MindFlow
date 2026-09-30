"""Seed a small, deterministic dataset for visual-baseline screenshots.

The reference UI ships static demo data; comparing a populated reference page
against an empty local page would measure the data, not the layout. This
script writes the same *kind* of rows the app produces naturally (focus
sessions + activity events) into an **isolated** data directory so the two
screenshots can be taken under equivalent conditions.

Usage (from backend-next/):

    uv run python scripts/seed_visual_baseline.py --data-dir <dir>
    uv run python scripts/seed_visual_baseline.py --data-dir <dir> --extra-dates 2026-07-29

``--extra-dates`` writes the same row shape on additional absolute dates. It
exists because some existing E2E specs pin a historical date (e.g. the focus
feedback test opens 2026-07-29); seeding that date keeps their original
assertions meaningful instead of loosening them for an empty database.

Safety:
  - Refuses to run without an explicit --data-dir (never touches the default
    user database).
  - Deletes only the rows it is about to re-insert (same user, same dates).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

# Last-7-day focus minutes per day — mirrors the shape of the reference demo
# chart (values in minutes, today partial).
_FOCUS_MINUTES = [182, 287, 165, 175, 194, 204, 221]
_DISTRACT_MINUTES = [46, 71, 52, 40, 61, 55, 49]
_APP_POOL = ["Code.exe", "chrome.exe", "WINWORD.EXE", "python.exe"]
_WINDOW_TITLES = [
    "mindflow-app - dashboard.html",
    "参考前端一比一复刻与接口适配计划 - 文档",
    "Untitled - Visual Studio Code",
    "MindFlow 设计稿 - Figma",
]


async def seed(data_dir: Path, extra_dates: list[date] | None = None) -> dict[str, int]:
    from sqlalchemy import delete, insert
    from sqlalchemy.ext.asyncio import create_async_engine

    from mindflow.infrastructure.repositories.focus import focus_sessions
    from mindflow.infrastructure.schema import activity_events

    db_path = (data_dir / "mindflow.db").resolve()
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path.as_posix()}")

    async with engine.begin() as conn:
        # The API derives its trend window from the *local* business day
        # (``business_today``), so seeding off the UTC date shifted the whole
        # window by one day and left "today" empty. Use the local calendar date.
        today = datetime.now().astimezone().date()
        window = [today - timedelta(days=offset) for offset in range(6, -1, -1)]
        # Trailing-window days keep the reference chart shape; extra dates are
        # chosen by the caller (they pin an E2E fixture date), never derived
        # from "now", and must not shift the window's per-day values.
        window_profile = {
            day: (_FOCUS_MINUTES[i], _DISTRACT_MINUTES[i])
            for i, day in enumerate(window)
        }
        extra = [d for d in (extra_dates or []) if d not in window]
        days = sorted(window + extra)

        # ── Focus sessions ────────────────────────────────────────────
        await conn.execute(
            delete(focus_sessions).where(
                focus_sessions.c.user_id == 1,
                focus_sessions.c.date.in_([d.isoformat() for d in days]),
            )
        )
        session_rows: list[dict[str, object]] = []
        for day in days:
            focus_min, distract_min = window_profile.get(
                day, (_FOCUS_MINUTES[0], _DISTRACT_MINUTES[0])
            )
            # Two focus blocks + one distraction block per day. ``/focus/trend``
            # sums focus rows into focus_min and distraction rows into
            # distraction_min, so these totals land on the reference numbers.
            blocks: list[tuple[str, float]] = [
                ("focus", focus_min * 0.7),
                ("focus", focus_min * 0.3),
                ("distraction", distract_min),
            ]
            cursor = datetime(day.year, day.month, day.day, 9, 0, tzinfo=UTC)
            for index, (session_type, minutes) in enumerate(blocks):
                start = cursor
                end = start + timedelta(minutes=minutes)
                cursor = end + timedelta(minutes=15)
                session_rows.append(
                    {
                        "id": uuid.uuid4().hex,
                        "user_id": 1,
                        "date": day.isoformat(),
                        "start_time": start.isoformat(),
                        "end_time": end.isoformat(),
                        "session_type": session_type,
                        "dominant_app": _APP_POOL[index % len(_APP_POOL)],
                        "focus_score": round(60 + (focus_min % 40) - distract_min / 10, 1),
                        "switch_count": int(distract_min / 8) + index,
                    }
                )
        await conn.execute(insert(focus_sessions), session_rows)

        # ── Activity events ───────────────────────────────────────────
        await conn.execute(
            delete(activity_events).where(
                activity_events.c.user_id == 1,
                activity_events.c.timestamp >= days[0].isoformat(),
            )
        )
        activity_rows: list[dict[str, object]] = []
        for day_index, day in enumerate(days):
            for slot in range(12):
                start = datetime(day.year, day.month, day.day, 8, 0, tzinfo=UTC) + timedelta(
                    hours=slot
                )
                app = _APP_POOL[(day_index + slot) % len(_APP_POOL)]
                title = _WINDOW_TITLES[(day_index + slot) % len(_WINDOW_TITLES)]
                payload = {
                    "app_name": app,
                    "window_title": title,
                    "process_name": app,
                    "is_idle": slot % 5 == 0,
                    # WindowSnapshot.from_dict requires the snapshot timestamp.
                    "timestamp_utc": start.isoformat(),
                }
                activity_rows.append(
                    {
                        "id": uuid.uuid4().hex,
                        "user_id": 1,
                        "timestamp": start.isoformat(),
                        "duration_s": 1800.0,
                        "data_json": json.dumps(payload, ensure_ascii=False),
                        "event_type": "window_snapshot",
                    }
                )
        # One "current" snapshot a few seconds old so /activities/current 200s.
        now = datetime.now(UTC)
        activity_rows.append(
            {
                "id": uuid.uuid4().hex,
                "user_id": 1,
                "timestamp": (now - timedelta(seconds=6)).isoformat(),
                "duration_s": 120.0,
                "data_json": json.dumps(
                    {
                        "app_name": "Code.exe",
                        "window_title": "mindflow-app - dashboard.html",
                        "process_name": "Code.exe",
                        "is_idle": False,
                        "timestamp_utc": (now - timedelta(seconds=6)).isoformat(),
                    },
                    ensure_ascii=False,
                ),
                "event_type": "window_snapshot",
            }
        )
        await conn.execute(insert(activity_events), activity_rows)

        counts = {"focus_sessions": len(session_rows), "activity_events": len(activity_rows)}

    await engine.dispose()
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument(
        "--extra-dates",
        default="",
        help="Comma-separated YYYY-MM-DD dates to seed in addition to the last 7 days",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    if not (data_dir / "mindflow.db").exists():
        print(f"error: no database at {data_dir / 'mindflow.db'}", file=sys.stderr)
        return 2

    extra: list[date] = []
    for raw in [part.strip() for part in args.extra_dates.split(",") if part.strip()]:
        try:
            extra.append(date.fromisoformat(raw))
        except ValueError:
            print(f"error: invalid --extra-dates value {raw!r}", file=sys.stderr)
            return 2

    counts = asyncio.run(seed(data_dir, extra))
    print(f"seeded {counts} into {data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
