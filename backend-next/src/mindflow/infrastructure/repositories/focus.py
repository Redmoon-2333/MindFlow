"""SQLAlchemy-backed FocusSession repository.

Stores and retrieves focus session projections derived from the
activity event stream (Wave 5).  Session data is computed by
``services/analysis_service.py`` and written here for persistence.

Table schema matches the Alembic migration (0001_create_core_tables).
All timestamps are stored as ISO8601 text (timezone-aware UTC).
"""

from __future__ import annotations

from datetime import date
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.domain.ids import new_id

# ── Table definition (matches migration 0001_create_core_tables) ─────

metadata = sa.MetaData()

focus_sessions = sa.Table(
    "focus_sessions",
    metadata,
    sa.Column("id", sa.Text(), primary_key=True),
    sa.Column("user_id", sa.Integer(), nullable=False),
    sa.Column("date", sa.Text(), nullable=False),
    sa.Column("start_time", sa.Text(), nullable=False),
    sa.Column("end_time", sa.Text(), nullable=False),
    sa.Column("session_type", sa.Text(), nullable=False),
    sa.Column("dominant_app", sa.Text(), nullable=True),
    sa.Column("focus_score", sa.Float(), nullable=True),
    sa.Column("switch_count", sa.Integer(), nullable=True),
    sa.Column(
        "created_at",
        sa.Text(),
        nullable=False,
        server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%SZ','now'))"),
    ),
)


# ── Repository ────────────────────────────────────────────────────────


class SQLAlchemyFocusSessionRepository:
    """Focus session repository backed by SQLAlchemy Core + async SQLite.

    Args:
        session_factory: Async session maker bound to the application engine.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._session_factory = session_factory

    # ── Public API ────────────────────────────────────────────────────

    async def save_sessions(
        self,
        user_id: int,
        sessions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Batch-insert focus sessions with idempotent replace semantics.

        Each dict in *sessions* must contain:
          ``date``, ``start_time``, ``end_time``, ``session_type``,
          ``dominant_app``, ``focus_score``, ``switch_count``.

        **Idempotency:** Within a single transaction, any existing rows for
        the same user + date(s) are deleted before inserting the new data
        (DELETE + INSERT replace).  This guarantees correct behaviour under
        concurrent callers — the last writer wins and the row count for a
        user+date never doubles.

        A UUIDv7 ``id`` is generated for each session automatically.

        Args:
            user_id: User identifier.
            sessions: Session data dicts (without ``id`` or ``user_id``).

        Returns:
            The inserted row dicts (with generated ``id``).
        """
        # Collect unique dates from session data so we know which rows to
        # replace.  In practice all sessions in one call share the same date,
        # but we handle the general case.
        dates_to_replace = {s["date"] for s in sessions}

        rows = []
        for s in sessions:
            sid = new_id()
            rows.append({
                "id": sid,
                "user_id": user_id,
                "date": s["date"],
                "start_time": s["start_time"],
                "end_time": s["end_time"],
                "session_type": s["session_type"],
                "dominant_app": s.get("dominant_app"),
                "focus_score": s.get("focus_score"),
                "switch_count": s.get("switch_count"),
            })

        async with self._session_factory() as session, session.begin():
            # Delete existing rows for user+date(s) before inserting
            delete_stmt = sa.delete(focus_sessions).where(
                focus_sessions.c.user_id == user_id,
                focus_sessions.c.date.in_(dates_to_replace),
            )
            await session.execute(delete_stmt)
            await session.execute(focus_sessions.insert(), rows)

        # Strip internal fields before returning
        result = []
        for r in rows:
            result.append({k: v for k, v in r.items() if k != "user_id"})
        return result

    async def query_range(
        self,
        user_id: int,
        start_date: date,
        end_date: date,
    ) -> list[dict[str, Any]]:
        """Return focus sessions in [*start_date*, *end_date*], ordered by time.

        Args:
            user_id: User identifier.
            start_date: Inclusive start date.
            end_date: Inclusive end date.

        Returns:
            A list of session dicts sorted by start_time ascending.
        """
        stmt = (
            sa.select(focus_sessions)
            .where(
                focus_sessions.c.user_id == user_id,
                focus_sessions.c.date >= start_date.isoformat(),
                focus_sessions.c.date <= end_date.isoformat(),
            )
            .order_by(focus_sessions.c.start_time.asc())
        )

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            return [_row_to_dict(row) for row in result.fetchall()]

    async def exists_for_date(self, user_id: int, target_date: date) -> bool:
        """Return True if sessions already exist for *user_id* on *target_date*.

        Used for idempotency checks before session identification.
        """
        stmt = (
            sa.select(sa.func.count())
            .select_from(focus_sessions)
            .where(
                focus_sessions.c.user_id == user_id,
                focus_sessions.c.date == target_date.isoformat(),
            )
        )
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            count: int = result.scalar() or 0
            return count > 0

    async def get_by_date(
        self,
        user_id: int,
        target_date: date,
    ) -> list[dict[str, Any]]:
        """Return all focus sessions for *user_id* on *target_date*."""
        return await self.query_range(user_id, target_date, target_date)

    def __repr__(self) -> str:
        return "<SQLAlchemyFocusSessionRepository>"


# ── Serialisation helpers ─────────────────────────────────────────────


def _row_to_dict(row: sa.Row[Any]) -> dict[str, Any]:
    """Convert a database row (``focus_sessions``) to a plain dict."""
    return {
        "id": row.id,
        "user_id": row.user_id,
        "date": row.date,
        "start_time": row.start_time,
        "end_time": row.end_time,
        "session_type": row.session_type,
        "dominant_app": row.dominant_app,
        "focus_score": row.focus_score,
        "switch_count": row.switch_count,
        "created_at": row.created_at,
    }
