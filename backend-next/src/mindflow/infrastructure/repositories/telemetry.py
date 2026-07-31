"""Persistence for privacy-preserving interaction and browser telemetry."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal

import sqlalchemy as sa
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.domain.ids import new_id
from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION
from mindflow.infrastructure.schema import (
    behavior_feature_windows,
    browser_segments,
    browser_tokens,
    focus_session_feedback,
    interaction_buckets,
)

_FEATURE_UPSERT_BATCH_SIZE = 250


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
        """Save or merge one heartbeat in a single transaction."""
        async with self._session_factory() as session, session.begin():
            return await self._save_browser_heartbeat_in_session(session, values)

    async def save_authenticated_browser_heartbeat(
        self,
        token_hash: str,
        *,
        heartbeat: dict[str, Any] | None,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Validate token, touch last-used time, and optionally save in one transaction."""
        now = datetime.now(UTC).isoformat()
        async with self._session_factory() as session, session.begin():
            token_result = await session.execute(
                sa.update(browser_tokens)
                .where(
                    browser_tokens.c.token_hash == token_hash,
                    browser_tokens.c.revoked_at.is_(None),
                )
                .values(last_used_at=now)
                .returning(browser_tokens.c.id)
            )
            if token_result.fetchone() is None:
                return False, None
            if heartbeat is None:
                return True, None
            segment = await self._save_browser_heartbeat_in_session(session, heartbeat)
            return True, segment

    async def _save_browser_heartbeat_in_session(
        self,
        session: AsyncSession,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        timestamp = values["timestamp_utc"].astimezone(UTC)
        result = await session.execute(
            sa.select(browser_segments)
            .where(browser_segments.c.user_id == values["user_id"])
            .order_by(browser_segments.c.timestamp.desc(), browser_segments.c.id.desc())
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
                duration_s = float(previous.duration_s) + float(values["duration_s"])
                await session.execute(
                    sa.update(browser_segments)
                    .where(browser_segments.c.id == previous.id)
                    .values(duration_s=duration_s)
                )
                return {
                    "id": previous.id,
                    "timestamp": previous.timestamp,
                    "duration_s": duration_s,
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
            row_id = await session.scalar(
                sa.update(focus_session_feedback)
                .where(
                    focus_session_feedback.c.user_id == user_id,
                    focus_session_feedback.c.session_id == session_id,
                )
                .values(label=label, score=score, task_type=task_type, created_at=now)
                .returning(focus_session_feedback.c.id)
            )
            if row_id is None:
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

    async def get_feedback_by_session_ids(
        self, user_id: int, session_ids: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Return feedback keyed by session_id for the given session IDs."""
        if not session_ids:
            return {}
        async with self._session_factory() as session:
            result = await session.execute(
                sa.select(focus_session_feedback)
                .where(
                    focus_session_feedback.c.user_id == user_id,
                    focus_session_feedback.c.session_id.in_(session_ids),
                )
            )
            out: dict[str, dict[str, Any]] = {}
            for row in result.fetchall():
                d = dict(row._mapping)
                out[d["session_id"]] = {
                    "feedback_label": d["label"],
                    "feedback_score": d["score"],
                    "feedback_task_type": d["task_type"],
                }
            return out



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
        await self.upsert_feature_windows([{
            "user_id": user_id,
            "window_start_utc": window_start_utc,
            "window_end_utc": window_end_utc,
            "feature_schema_version": feature_schema_version,
            "features_json": features_json,
            "label": label,
        }])

    async def upsert_feature_windows(
        self,
        rows: list[dict[str, Any]],
        *,
        session: AsyncSession | None = None,
    ) -> list[dict[str, Any]]:
        """Bulk UPSERT feature windows and return the rows that were inserted.

        The ``(user_id, window_start_utc, feature_schema_version)`` unique
        constraint keeps the upsert idempotent. The return value is the
        *truthful* set of rows that did not exist before this call: a caller
        (the telemetry rollup) uses it to fold only genuinely new windows into
        the personal baseline, which prevents Welford double counting when the
        same range is rolled more than once. Detection is a single key-presence
        SELECT inside the same transaction — no N+1 lookups, no ambiguous
        ``rowcount``.

        When *session* is provided, the work runs inside that caller-owned
        transaction (used to persist windows and the baseline atomically);
        otherwise a transaction is opened and committed here.

        Returns:
            The normalized rows that were inserted (never previously stored).
        """
        if not rows:
            return []
        if session is not None:
            return await self._upsert_feature_windows_in_session(session, rows)
        async with self._session_factory() as session, session.begin():
            return await self._upsert_feature_windows_in_session(session, rows)

    async def _upsert_feature_windows_in_session(
        self,
        session: AsyncSession,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        now = datetime.now(UTC).isoformat()
        values = [
            {
                "id": new_id(),
                "user_id": row["user_id"],
                "window_start_utc": row["window_start_utc"].astimezone(UTC).isoformat(),
                "window_end_utc": row["window_end_utc"].astimezone(UTC).isoformat(),
                "feature_schema_version": row["feature_schema_version"],
                "features_json": row["features_json"],
                "label": row.get("label"),
                "created_at": now,
            }
            for row in rows
        ]

        # One batched key-presence probe for the whole payload: which of these
        # keys already exist, before this call's writes are visible.
        existing_stmt = sa.select(
            behavior_feature_windows.c.user_id,
            behavior_feature_windows.c.window_start_utc,
            behavior_feature_windows.c.feature_schema_version,
        ).where(
            sa.or_(
                *[
                    sa.and_(
                        behavior_feature_windows.c.user_id == value["user_id"],
                        behavior_feature_windows.c.window_start_utc == value["window_start_utc"],
                        behavior_feature_windows.c.feature_schema_version
                        == value["feature_schema_version"],
                    )
                    for value in values
                ]
            )
        )
        existing_keys = {
            (row.user_id, row.window_start_utc, row.feature_schema_version)
            for row in (await session.execute(existing_stmt)).fetchall()
        }

        for index in range(0, len(values), _FEATURE_UPSERT_BATCH_SIZE):
            batch = values[index:index + _FEATURE_UPSERT_BATCH_SIZE]
            statement = sqlite_insert(behavior_feature_windows).values(batch)
            statement = statement.on_conflict_do_update(
                index_elements=[
                    behavior_feature_windows.c.user_id,
                    behavior_feature_windows.c.window_start_utc,
                    behavior_feature_windows.c.feature_schema_version,
                ],
                set_={
                    "window_end_utc": statement.excluded.window_end_utc,
                    "features_json": statement.excluded.features_json,
                    "label": statement.excluded.label,
                    "created_at": statement.excluded.created_at,
                },
            )
            await session.execute(statement)

        return [
            value
            for value in values
            if (value["user_id"], value["window_start_utc"], value["feature_schema_version"])
            not in existing_keys
        ]

    async def latest_feature_window(
        self,
        user_id: int,
        feature_schema_version: int = FEATURE_SCHEMA_VERSION,
    ) -> dict[str, Any] | None:
        statement = (
            sa.select(behavior_feature_windows)
            .where(
                behavior_feature_windows.c.user_id == user_id,
                behavior_feature_windows.c.feature_schema_version
                == feature_schema_version,
            )
            .order_by(behavior_feature_windows.c.window_start_utc.desc())
            .limit(1)
        )
        async with self._session_factory() as session:
            row = (await session.execute(statement)).fetchone()
            return dict(row._mapping) if row is not None else None

    async def list_feature_windows(
        self,
        user_id: int,
        feature_schema_version: int = FEATURE_SCHEMA_VERSION,
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

    async def list_feature_windows_in_range(
        self,
        user_id: int,
        start: datetime,
        end: datetime,
        feature_schema_version: int = FEATURE_SCHEMA_VERSION,
    ) -> list[dict[str, Any]]:
        """Bounded feature windows within the half-open [start, end).

        One range query used by the conditional baseline backfill so it never
        loads the user's entire window history (no per-row lookups).
        ``window_start_utc`` is stored as UTC ISO text, so both bounds are
        normalized to UTC before comparison; results come back ascending.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                sa.select(behavior_feature_windows)
                .where(
                    behavior_feature_windows.c.user_id == user_id,
                    behavior_feature_windows.c.feature_schema_version
                    == feature_schema_version,
                    behavior_feature_windows.c.window_start_utc
                    >= start.astimezone(UTC).isoformat(),
                    behavior_feature_windows.c.window_start_utc
                    < end.astimezone(UTC).isoformat(),
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
            interaction_result = await session.scalars(
                sa.delete(interaction_buckets)
                .where(
                    interaction_buckets.c.window_start_utc
                    < interaction_cutoff.astimezone(UTC).isoformat()
                )
                .returning(interaction_buckets.c.id)
            )
            total += len(interaction_result.all())
            browser_result = await session.scalars(
                sa.delete(browser_segments)
                .where(
                    browser_segments.c.timestamp
                    < activity_cutoff.astimezone(UTC).isoformat()
                )
                .returning(browser_segments.c.id)
            )
            total += len(browser_result.all())
            feature_result = await session.scalars(
                sa.delete(behavior_feature_windows)
                .where(
                    behavior_feature_windows.c.window_start_utc
                    < feature_cutoff.astimezone(UTC).isoformat()
                )
                .returning(behavior_feature_windows.c.id)
            )
            total += len(feature_result.all())
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
                sa.update(browser_tokens)
                .where(
                    browser_tokens.c.token_hash == token_hash,
                    browser_tokens.c.revoked_at.is_(None),
                )
                .values(last_used_at=now)
                .returning(browser_tokens.c.id)
            )
            return result.fetchone() is not None

    async def revoke_browser_tokens(self, user_id: int) -> int:
        async with self._session_factory() as session, session.begin():
            result = await session.scalars(
                sa.update(browser_tokens)
                .where(
                    browser_tokens.c.user_id == user_id,
                    browser_tokens.c.revoked_at.is_(None),
                )
                .values(revoked_at=datetime.now(UTC).isoformat())
                .returning(browser_tokens.c.id)
            )
            return len(result.all())

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
                result = await session.scalars(
                    sa.delete(table)
                    .where(table.c.user_id == user_id)
                    .returning(table.c.id)
                )
                total += len(result.all())
        return total
