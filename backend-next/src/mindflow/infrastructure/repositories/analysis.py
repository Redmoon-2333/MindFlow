"""SQLAlchemy-backed ProcrastinationAnalysis repository.

Stores idempotent LLM attribution results (one per user per date via
UNIQUE constraint).  Data is written by ``services/llm_service.py``
and read for cache checks and historical lookup.

Table schema matches the Alembic migration (0001_create_core_tables).
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.domain.ids import new_id

# ── Table definition (matches migration 0001_create_core_tables) ─────

metadata = sa.MetaData()

procrastination_analyses = sa.Table(
    "procrastination_analyses",
    metadata,
    sa.Column("id", sa.Text(), primary_key=True),
    sa.Column("user_id", sa.Integer(), nullable=False),
    sa.Column("date", sa.Text(), nullable=False),
    sa.Column("procrastination_types_json", sa.Text(), nullable=True),
    sa.Column("type_confidence_json", sa.Text(), nullable=True),
    sa.Column("cognitive_distortions_json", sa.Text(), nullable=True),
    sa.Column("cbt_technique", sa.Text(), nullable=True),
    sa.Column("response_text", sa.Text(), nullable=True),
    sa.Column("llm_model", sa.Text(), nullable=True),
    sa.Column("llm_cost_usd", sa.Float(), nullable=True),
    sa.Column(
        "created_at",
        sa.Text(),
        nullable=False,
        server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%SZ','now'))"),
    ),
    sa.PrimaryKeyConstraint("id"),
    sa.UniqueConstraint("user_id", "date"),
)


# ── Repository ───────────────────────────────────────────────────────


class SQLAlchemyProcrastinationAnalysisRepository:
    """Procrastination analysis repository backed by SQLAlchemy Core + async SQLite.

    Uses SQLite UPSERT (ON CONFLICT DO UPDATE) for idempotent writes.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._session_factory = session_factory

    # ── Public API ────────────────────────────────────────────────────

    async def get_by_date(
        self,
        user_id: int,
        target_date: date,
    ) -> dict[str, Any] | None:
        """Return the analysis for *user_id* on *target_date*, or None.

        Returns:
            A dict with the analysis data (types, confidence, etc.)
            or None if no analysis exists for that date.
        """
        stmt = (
            sa.select(procrastination_analyses)
            .where(
                procrastination_analyses.c.user_id == user_id,
                procrastination_analyses.c.date == target_date.isoformat(),
            )
            .limit(1)
        )

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            row = result.fetchone()

        if row is None:
            return None

        return _row_to_analysis(row)

    async def upsert(
        self,
        user_id: int,
        target_date: date,
        *,
        procrastination_types: list[str],
        type_confidence: dict[str, float],
        cognitive_distortions: list[str],
        cbt_technique: str | None,
        response_text: str,
        llm_model: str | None = None,
        llm_cost_usd: float = 0.0,
    ) -> None:
        """Insert or update a procrastination analysis record.

        The UNIQUE(user_id, date) constraint makes this idempotent:
        calling upsert twice with the same (user_id, date) updates
        the existing row rather than creating a duplicate.

        Args:
            user_id: User identifier.
            target_date: Date of the analysis.
            procrastination_types: List of detected type strings.
            type_confidence: Per-type confidence map.
            cognitive_distortions: List of cognitive distortions identified.
            cbt_technique: Recommended CBT technique.
            response_text: User-facing analysis text.
            llm_model: Source identifier (deepseek/ollama/rule_engine).
            llm_cost_usd: Approximate cost of the LLM call.
        """
        stmt = sqlite_upsert(procrastination_analyses).values(
            id=new_id(),
            user_id=user_id,
            date=target_date.isoformat(),
            procrastination_types_json=json.dumps(procrastination_types, ensure_ascii=False),
            type_confidence_json=json.dumps(type_confidence, ensure_ascii=False),
            cognitive_distortions_json=json.dumps(cognitive_distortions, ensure_ascii=False),
            cbt_technique=cbt_technique,
            response_text=response_text,
            llm_model=llm_model,
            llm_cost_usd=llm_cost_usd,
        )

        # On conflict, update the existing row
        stmt = stmt.on_conflict_do_update(
            constraint="procrastination_analyses_user_id_date_key",  # type: ignore[call-arg]
            set_={
                "procrastination_types_json": stmt.excluded.procrastination_types_json,
                "type_confidence_json": stmt.excluded.type_confidence_json,
                "cognitive_distortions_json": stmt.excluded.cognitive_distortions_json,
                "cbt_technique": stmt.excluded.cbt_technique,
                "response_text": stmt.excluded.response_text,
                "llm_model": stmt.excluded.llm_model,
                "llm_cost_usd": stmt.excluded.llm_cost_usd,
            },
        )

        async with self._session_factory() as session, session.begin():
            await session.execute(stmt)

    # ── Exists check ──────────────────────────────────────────────────

    async def exists(self, user_id: int, target_date: date) -> bool:
        """Return True if an analysis exists for *user_id* on *target_date*."""
        stmt = (
            sa.select(sa.literal(1))
            .select_from(procrastination_analyses)
            .where(
                procrastination_analyses.c.user_id == user_id,
                procrastination_analyses.c.date == target_date.isoformat(),
            )
            .limit(1)
        )

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            return result.fetchone() is not None


# ── Serialisation helper ─────────────────────────────────────────────


def _row_to_analysis(row: sa.Row[Any]) -> dict[str, Any]:
    """Convert a database row to a dict matching the API response shape."""
    result: dict[str, Any] = {}

    raw_types = row.procrastination_types_json
    if raw_types:
        result["procrastination_types"] = json.loads(raw_types)
    else:
        result["procrastination_types"] = []

    raw_confidence = row.type_confidence_json
    if raw_confidence:
        result["type_confidence"] = json.loads(raw_confidence)
    else:
        result["type_confidence"] = {}

    raw_distortions = row.cognitive_distortions_json
    if raw_distortions:
        result["cognitive_distortions"] = json.loads(raw_distortions)
    else:
        result["cognitive_distortions"] = []

    if row.cbt_technique:
        result["cbt_technique"] = row.cbt_technique
    if row.response_text:
        result["response_text"] = row.response_text
    if row.llm_model:
        result["source"] = row.llm_model

    return result
