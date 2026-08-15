"""Repository for the ``baseline_models`` table (one row per user).

Owns the single source of truth for reading and writing a user's personal
behavior baseline. Previously the table was defined *and* queried inline
inside ``services/evidence_service.py`` — infrastructure code leaking into
the services layer, and a second ``sa.Table`` definition alongside Alembic
and ``train/``. This repository centralises that access so the service
depends on a repository abstraction instead of raw SQLAlchemy.

The row's ``model_json`` payload is parsed into a ``BaselineModel`` domain
object here; callers never see SQLAlchemy rows.
"""

from __future__ import annotations

import json
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.domain.baseline import BaselineModel
from mindflow.domain.ids import new_id
from mindflow.infrastructure.schema import baseline_models as baseline_models

# ── Repository ────────────────────────────────────────────────────────


class BaselineRepository:
    """Read/write access to a user's persisted ``BaselineModel``.

    ``baseline_models`` enforces one row per user via ``UNIQUE(user_id)``;
    :meth:`upsert` inserts the first row or updates the existing one in
    place, so a user never owns more than one baseline.

    Args:
        session_factory: Async session maker bound to the application engine.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._session_factory = session_factory

    async def upsert(
        self,
        model: BaselineModel,
        *,
        session: AsyncSession | None = None,
    ) -> None:
        """Insert or update the single baseline row for ``model.user_id``.

        The first write creates the row; every later write updates it via
        SQLite ``ON CONFLICT (user_id) DO UPDATE``, which is what keeps the
        table at exactly one row per user. ``created_at`` is set on the
        first insert and deliberately not included in the update set, so a
        re-upsert never rewrites the user's baseline birth time. ``model_json``
        and ``updated_at`` follow the model and move forward on each update.

        When *session* is provided, the write runs inside that caller-owned
        transaction (used to persist windows and the baseline atomically);
        otherwise a transaction is opened and committed here.

        ``model_json`` is serialized inside the transaction: a model that
        cannot be JSON-encoded raises and rolls the whole write back,
        leaving any previously persisted row untouched.
        """
        payload = model.to_dict()
        model_json = json.dumps(payload, ensure_ascii=False)
        stmt = sqlite_upsert(baseline_models).values(
            id=new_id(),
            user_id=model.user_id,
            model_json=model_json,
            training_events_count=model.total_samples(),
            created_at=model.created_at.isoformat(),
            updated_at=model.updated_at.isoformat(),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["user_id"],
            set_={
                "model_json": stmt.excluded.model_json,
                "training_events_count": stmt.excluded.training_events_count,
                "updated_at": stmt.excluded.updated_at,
                # created_at intentionally absent: the first-insert time
                # is the row's birth and must survive re-upserts.
            },
        )
        if session is not None:
            await session.execute(stmt)
            return
        async with self._session_factory() as session, session.begin():
            await session.execute(stmt)

    async def get_latest(
        self,
        user_id: int,
        *,
        session: AsyncSession | None = None,
    ) -> BaselineModel | None:
        """Return the baseline row for *user_id*, or None.

        The table is unique per ``user_id``, so the matching row is the
        user's baseline. Returns None when no baseline exists or the stored
        JSON is malformed. When *session* is provided, the read uses that
        caller-owned session instead of opening its own.
        """
        stmt = (
            sa.select(baseline_models.c.model_json)
            .where(baseline_models.c.user_id == user_id)
            .limit(1)
        )
        if session is not None:
            row = (await session.execute(stmt)).fetchone()
        else:
            async with self._session_factory() as session:
                row = (await session.execute(stmt)).fetchone()

        if row is None:
            return None

        try:
            data: dict[str, Any] = json.loads(row.model_json)
        except (json.JSONDecodeError, TypeError):
            return None

        return BaselineModel.from_dict(data)

    def __repr__(self) -> str:
        return "<BaselineRepository>"
