"""Read-only repository for the ``baseline_models`` table.

Owns the single source of truth for reading a user's personal behavior
baseline. Previously the table was defined *and* queried inline inside
``services/evidence_service.py`` — infrastructure code leaking into the
services layer, and a second ``sa.Table`` definition alongside Alembic and
``train/``. This repository centralises that access so the service depends
on a repository abstraction instead of raw SQLAlchemy.

The row's ``model_json`` payload is parsed into a ``BaselineModel`` domain
object here; callers never see SQLAlchemy rows.
"""

from __future__ import annotations

import json
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.domain.baseline import BaselineModel

# ── Table definition (matches migration; also referenced by train/) ──────

metadata = sa.MetaData()

baseline_models = sa.Table(
    "baseline_models",
    metadata,
    sa.Column("id", sa.Text(), primary_key=True),
    sa.Column("user_id", sa.Integer(), nullable=False),
    sa.Column("model_json", sa.Text(), nullable=False),
    sa.Column("training_events_count", sa.Integer(), nullable=False),
    sa.Column("created_at", sa.Text(), nullable=False),
    sa.Column("updated_at", sa.Text(), nullable=False),
)


# ── Repository ────────────────────────────────────────────────────────


class BaselineRepository:
    """Read access to a user's persisted ``BaselineModel``.

    Args:
        session_factory: Async session maker bound to the application engine.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._session_factory = session_factory

    async def get_latest(self, user_id: int) -> BaselineModel | None:
        """Return the most recently updated baseline for *user_id*, or None.

        The table has no user_id uniqueness, so a retrained user may own
        several rows — ordering by ``updated_at`` descending ensures the
        freshest model wins (review H1: an unordered fetchone could load a
        stale model). Returns None when no baseline exists or the stored
        JSON is malformed.
        """
        stmt = (
            sa.select(baseline_models.c.model_json)
            .where(baseline_models.c.user_id == user_id)
            .order_by(sa.desc(baseline_models.c.updated_at))
            .limit(1)
        )
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            row = result.fetchone()

        if row is None:
            return None

        try:
            data: dict[str, Any] = json.loads(row.model_json)
        except (json.JSONDecodeError, TypeError):
            return None

        return BaselineModel.from_dict(data)

    def __repr__(self) -> str:
        return "<BaselineRepository>"
