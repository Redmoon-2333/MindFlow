"""Persistence for privacy-preserving interaction and browser telemetry."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.domain.ids import new_id

metadata = sa.MetaData()

interaction_buckets = sa.Table(
    "interaction_buckets",
    metadata,
    sa.Column("id", sa.Text(), primary_key=True),
    sa.Column("user_id", sa.Integer(), nullable=False),
    sa.Column("window_start_utc", sa.Text(), nullable=False),
    sa.Column("duration_s", sa.Float(), nullable=False),
    sa.Column("context_key", sa.Text(), nullable=False),
    sa.Column("keypress_count", sa.Integer(), nullable=False),
    sa.Column("mouse_click_count", sa.Integer(), nullable=False),
    sa.Column("scroll_delta", sa.Integer(), nullable=False),
    sa.Column("mouse_distance_px", sa.Float(), nullable=False),
    sa.Column("input_active_s", sa.Float(), nullable=False),
    sa.Column("interaction_burst_count", sa.Integer(), nullable=False),
    sa.Column("created_at", sa.Text(), nullable=False),
)

browser_segments = sa.Table(
    "browser_segments",
    metadata,
    sa.Column("id", sa.Text(), primary_key=True),
    sa.Column("user_id", sa.Integer(), nullable=False),
    sa.Column("timestamp", sa.Text(), nullable=False),
    sa.Column("duration_s", sa.Float(), nullable=False),
    sa.Column("browser_name", sa.Text(), nullable=False),
    sa.Column("domain", sa.Text(), nullable=False),
    sa.Column("audible", sa.Boolean(), nullable=False),
    sa.Column("context_key", sa.Text(), nullable=False),
    sa.Column("created_at", sa.Text(), nullable=False),
)

focus_session_feedback = sa.Table(
    "focus_session_feedback",
    metadata,
    sa.Column("id", sa.Text(), primary_key=True),
    sa.Column("user_id", sa.Integer(), nullable=False),
    sa.Column("session_id", sa.Text(), nullable=False),
    sa.Column("label", sa.Text(), nullable=False),
    sa.Column("score", sa.Integer(), nullable=False),
    sa.Column("task_type", sa.Text(), nullable=True),
    sa.Column("created_at", sa.Text(), nullable=False),
    sa.UniqueConstraint("user_id", "session_id"),
)

browser_tokens = sa.Table(
    "browser_tokens",
    metadata,
    sa.Column("id", sa.Text(), primary_key=True),
    sa.Column("user_id", sa.Integer(), nullable=False),
    sa.Column("token_hash", sa.Text(), nullable=False, unique=True),
    sa.Column("created_at", sa.Text(), nullable=False),
    sa.Column("last_used_at", sa.Text(), nullable=True),
    sa.Column("revoked_at", sa.Text(), nullable=True),
)

behavior_feature_windows = sa.Table(
    "behavior_feature_windows",
    metadata,
    sa.Column("id", sa.Text(), primary_key=True),
    sa.Column("user_id", sa.Integer(), nullable=False),
    sa.Column("window_start_utc", sa.Text(), nullable=False),
    sa.Column("window_end_utc", sa.Text(), nullable=False),
    sa.Column("feature_schema_version", sa.Integer(), nullable=False),
    sa.Column("features_json", sa.Text(), nullable=False),
    sa.Column("label", sa.Text(), nullable=True),
    sa.Column("created_at", sa.Text(), nullable=False),
    sa.UniqueConstraint("user_id", "window_start_utc", "feature_schema_version"),
)


class TelemetryRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def save_interaction_bucket(self, **values: Any) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        row = {
            "id": new_id(),
            "created_at": now,
            **values,
        }
        row["window_start_utc"] = row["window_start_utc"].astimezone(UTC).isoformat()
        async with self._session_factory() as session, session.begin():
            await session.execute(interaction_buckets.insert().values(**row))
        return row

    async def save_browser_heartbeat(self, **values: Any) -> dict[str, Any]:
        timestamp = values["timestamp_utc"].astimezone(UTC)
        async with self._session_factory() as session, session.begin():
            result = await session.execute(
                sa.select(browser_segments)
                .where(browser_segments.c.user_id == values["user_id"])
                .order_by(browser_segments.c.timestamp.desc())
                .limit(1)
            )
            previous = result.fetchone()
            if previous is not None:
                previous_end = datetime.fromisoformat(previous.timestamp) + timedelta(
                    seconds=float(previous.duration_s)
                )
                gap_s = (timestamp - previous_end).total_seconds()
                matches = (
                    previous.browser_name == values["browser_name"]
                    and previous.domain == values["domain"]
                    and bool(previous.audible) == bool(values["audible"])
                    and previous.context_key == values["context_key"]
                )
                if matches and -10.0 <= gap_s <= 10.0:
                    await session.execute(
                        sa.update(browser_segments)
                        .where(browser_segments.c.id == previous.id)
                        .values(
                            duration_s=browser_segments.c.duration_s
                            + float(values["duration_s"])
                        )
                    )
                    return {
                        "id": previous.id,
                        "timestamp": previous.timestamp,
                        "duration_s": float(previous.duration_s)
                        + float(values["duration_s"]),
                    }

            row = {
                "id": new_id(),
                "user_id": values["user_id"],
                "timestamp": timestamp.isoformat(),
                "duration_s": float(values["duration_s"]),
                "browser_name": values["browser_name"],
                "domain": values["domain"],
                "audible": bool(values["audible"]),
                "context_key": values["context_key"],
                "created_at": datetime.now(UTC).isoformat(),
            }
            await session.execute(browser_segments.insert().values(**row))
            return row

    async def last_browser_segment_before(
        self,
        user_id: int,
        timestamp: datetime,
    ) -> dict[str, Any] | None:
        async with self._session_factory() as session:
            result = await session.execute(
                sa.select(browser_segments)
                .where(
                    browser_segments.c.user_id == user_id,
                    browser_segments.c.timestamp < timestamp.astimezone(UTC).isoformat(),
                )
                .order_by(browser_segments.c.timestamp.desc(), browser_segments.c.id.desc())
                .limit(1)
            )
            row = result.fetchone()
            return dict(row._mapping) if row is not None else None

    async def list_browser_segments(
        self,
        user_id: int,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, Any]]:
        async with self._session_factory() as session:
            result = await session.execute(
                sa.select(browser_segments)
                .where(
                    browser_segments.c.user_id == user_id,
                    browser_segments.c.timestamp >= start.astimezone(UTC).isoformat(),
                    browser_segments.c.timestamp < end.astimezone(UTC).isoformat(),
                )
                .order_by(browser_segments.c.timestamp.asc())
            )
            return [dict(row._mapping) for row in result.fetchall()]

    async def save_focus_feedback(
        self,
        user_id: int,
        session_id: str,
        label: str,
        score: int,
        task_type: str | None,
    ) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        async with self._session_factory() as session, session.begin():
            result = await session.execute(
                sa.update(focus_session_feedback)
                .where(
                    focus_session_feedback.c.user_id == user_id,
                    focus_session_feedback.c.session_id == session_id,
                )
                .values(label=label, score=score, task_type=task_type, created_at=now)
            )
            if result.rowcount:
                row_id = await session.scalar(
                    sa.select(focus_session_feedback.c.id).where(
                        focus_session_feedback.c.user_id == user_id,
                        focus_session_feedback.c.session_id == session_id,
                    )
                )
            else:
                row_id = new_id()
                await session.execute(
                    focus_session_feedback.insert().values(
                        id=row_id,
                        user_id=user_id,
                        session_id=session_id,
                        label=label,
                        score=score,
                        task_type=task_type,
                        created_at=now,
                    )
                )
        return {
            "id": row_id,
            "user_id": user_id,
            "session_id": session_id,
            "label": label,
            "score": score,
            "task_type": task_type,
            "created_at": now,
        }

    async def list_focus_feedback(self, user_id: int) -> list[dict[str, Any]]:
        async with self._session_factory() as session:
            result = await session.execute(
                sa.select(focus_session_feedback)
                .where(focus_session_feedback.c.user_id == user_id)
                .order_by(focus_session_feedback.c.created_at.asc())
            )
            return [dict(row._mapping) for row in result.fetchall()]



    async def list_interaction_buckets(
        self,
        user_id: int,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, Any]]:
        async with self._session_factory() as session:
            result = await session.execute(
                sa.select(interaction_buckets)
                .where(
                    interaction_buckets.c.user_id == user_id,
                    interaction_buckets.c.window_start_utc >= start.astimezone(UTC).isoformat(),
                    interaction_buckets.c.window_start_utc < end.astimezone(UTC).isoformat(),
                )
                .order_by(interaction_buckets.c.window_start_utc.asc())
            )
            return [dict(row._mapping) for row in result.fetchall()]

    async def save_feature_window(
        self,
        user_id: int,
        window_start_utc: datetime,
        window_end_utc: datetime,
        feature_schema_version: int,
        features_json: str,
        label: str | None = None,
    ) -> None:
        values = {
            "user_id": user_id,
            "window_start_utc": window_start_utc.astimezone(UTC).isoformat(),
            "window_end_utc": window_end_utc.astimezone(UTC).isoformat(),
            "feature_schema_version": feature_schema_version,
            "features_json": features_json,
            "label": label,
            "created_at": datetime.now(UTC).isoformat(),
        }
        async with self._session_factory() as session, session.begin():
            result = await session.execute(
                sa.update(behavior_feature_windows)
                .where(
                    behavior_feature_windows.c.user_id == user_id,
                    behavior_feature_windows.c.window_start_utc
                    == values["window_start_utc"],
                    behavior_feature_windows.c.feature_schema_version
                    == feature_schema_version,
                )
                .values(**values)
            )
            if not result.rowcount:
                await session.execute(
                    behavior_feature_windows.insert().values(id=new_id(), **values)
                )

    async def list_feature_windows(
        self,
        user_id: int,
        feature_schema_version: int = 2,
    ) -> list[dict[str, Any]]:
        async with self._session_factory() as session:
            result = await session.execute(
                sa.select(behavior_feature_windows)
                .where(
                    behavior_feature_windows.c.user_id == user_id,
                    behavior_feature_windows.c.feature_schema_version
                    == feature_schema_version,
                )
                .order_by(behavior_feature_windows.c.window_start_utc.asc())
            )
            return [dict(row._mapping) for row in result.fetchall()]

    async def cleanup_old_telemetry(
        self,
        interaction_cutoff: datetime,
        activity_cutoff: datetime,
        feature_cutoff: datetime,
    ) -> int:
        total = 0
        async with self._session_factory() as session, session.begin():
            interaction_result = await session.execute(
                sa.delete(interaction_buckets).where(
                    interaction_buckets.c.window_start_utc
                    < interaction_cutoff.astimezone(UTC).isoformat()
                )
            )
            browser_result = await session.execute(
                sa.delete(browser_segments).where(
                    browser_segments.c.timestamp
                    < activity_cutoff.astimezone(UTC).isoformat()
                )
            )
            feature_result = await session.execute(
                sa.delete(behavior_feature_windows).where(
                    behavior_feature_windows.c.window_start_utc
                    < feature_cutoff.astimezone(UTC).isoformat()
                )
            )
            total += int(interaction_result.rowcount or 0)
            total += int(browser_result.rowcount or 0)
            total += int(feature_result.rowcount or 0)
        return total

    async def save_browser_token(self, user_id: int, token_hash: str) -> None:
        async with self._session_factory() as session, session.begin():
            await session.execute(
                browser_tokens.insert().values(
                    id=new_id(),
                    user_id=user_id,
                    token_hash=token_hash,
                    created_at=datetime.now(UTC).isoformat(),
                    last_used_at=None,
                    revoked_at=None,
                )
            )

    async def verify_browser_token(self, token_hash: str) -> bool:
        now = datetime.now(UTC).isoformat()
        async with self._session_factory() as session, session.begin():
            result = await session.execute(
                sa.select(browser_tokens.c.id).where(
                    browser_tokens.c.token_hash == token_hash,
                    browser_tokens.c.revoked_at.is_(None),
                )
            )
            row = result.fetchone()
            if row is None:
                return False
            await session.execute(
                sa.update(browser_tokens)
                .where(browser_tokens.c.id == row.id)
                .values(last_used_at=now)
            )
            return True

    async def revoke_browser_tokens(self, user_id: int) -> int:
        async with self._session_factory() as session, session.begin():
            result = await session.execute(
                sa.update(browser_tokens)
                .where(
                    browser_tokens.c.user_id == user_id,
                    browser_tokens.c.revoked_at.is_(None),
                )
                .values(revoked_at=datetime.now(UTC).isoformat())
            )
            return int(result.rowcount or 0)

    async def get_status(self, user_id: int, target_date: date) -> dict[str, Any]:
        day_prefix = target_date.isoformat()
        async with self._session_factory() as session:
            interaction_count = await session.scalar(
                sa.select(sa.func.count())
                .select_from(interaction_buckets)
                .where(
                    interaction_buckets.c.user_id == user_id,
                    interaction_buckets.c.window_start_utc.startswith(day_prefix),
                )
            )
            browser_count = await session.scalar(
                sa.select(sa.func.count())
                .select_from(browser_segments)
                .where(
                    browser_segments.c.user_id == user_id,
                    browser_segments.c.timestamp.startswith(day_prefix),
                )
            )
            last_interaction = await session.scalar(
                sa.select(sa.func.max(interaction_buckets.c.window_start_utc)).where(
                    interaction_buckets.c.user_id == user_id
                )
            )
            last_browser = await session.scalar(
                sa.select(sa.func.max(browser_segments.c.timestamp)).where(
                    browser_segments.c.user_id == user_id
                )
            )
            active_browser_tokens = await session.scalar(
                sa.select(sa.func.count())
                .select_from(browser_tokens)
                .where(
                    browser_tokens.c.user_id == user_id,
                    browser_tokens.c.revoked_at.is_(None),
                )
            )
        return {
            "interaction_bucket_count": int(interaction_count or 0),
            "browser_segment_count": int(browser_count or 0),
            "last_interaction_at": last_interaction,
            "last_browser_at": last_browser,
            "browser_paired": bool(active_browser_tokens),
        }

    async def delete_scope(
        self,
        user_id: int,
        scope: Literal["interaction", "browser", "feedback", "all"],
    ) -> int:
        tables = {
            "interaction": [interaction_buckets, behavior_feature_windows],
            "browser": [
                browser_segments,
                browser_tokens,
                behavior_feature_windows,
            ],
            "feedback": [focus_session_feedback],
            "all": [
                interaction_buckets,
                browser_segments,
                browser_tokens,
                focus_session_feedback,
                behavior_feature_windows,
            ],
        }[scope]
        total = 0
        async with self._session_factory() as session, session.begin():
            for table in tables:
                result = await session.execute(
                    sa.delete(table).where(table.c.user_id == user_id)
                )
                total += int(result.rowcount or 0)
        return total
