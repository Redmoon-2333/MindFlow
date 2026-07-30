"""SQLAlchemy-backed ProcrastinationAnalysis repository.

Stores idempotent LLM attribution results — one row per
(user_id, date, analysis_kind) via UNIQUE constraint
(migration 0011).  Multiple analysis kinds (daily_panel,
daily_attribution, ml) can coexist for the same day without
overwriting each other.

Data is written by ``services/llm_service.py`` and
``services/panel_service.py``, and read for cache checks
and historical lookup.

Table schema matches Alembic migrations 0001, 0002, and 0011.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.domain.ids import new_id
from mindflow.infrastructure.schema import procrastination_analyses

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
        *,
        analysis_kind: str | None = None,
    ) -> dict[str, Any] | None:
        """Return the analysis for *user_id* on *target_date*, or None.

        When *analysis_kind* is given, filters to that specific kind
        (e.g. ``"daily_panel"``).  When omitted, returns the first
        matching row for the date regardless of kind.

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
        if analysis_kind is not None:
            stmt = stmt.where(
                procrastination_analyses.c.analysis_kind == analysis_kind,
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
        panel_transcript: dict[str, Any] | None = None,
        analysis_kind: str = "daily_attribution",
        source: str | None = None,
    ) -> None:
        """Insert or update a procrastination analysis record.

        The UNIQUE(user_id, date, analysis_kind) constraint makes this
        idempotent per kind: calling upsert twice with the same
        (user_id, date, analysis_kind) updates the existing row rather
        than creating a duplicate.  Different kinds coexist for the
        same day.

        Args:
            user_id: User identifier.
            target_date: Date of the analysis.
            procrastination_types: List of detected type strings.
            type_confidence: Per-type confidence map.
            cognitive_distortions: List of cognitive distortions identified.
            cbt_technique: Recommended CBT technique.
            response_text: User-facing analysis text.
            llm_model: LLM model identifier (deepseek-chat, ollama, etc.).
            llm_cost_usd: Approximate cost of the LLM call.
            panel_transcript: Optional panel transcript and verdict metadata.
            analysis_kind: Workflow kind — ``"daily_panel"``, ``"daily_attribution"``,
                ``"ml"``, or ``"legacy_unknown"``.
            source: Degradation source — ``"panel"``, ``"single_expert"``,
                ``"ollama"``, or ``"rule_engine"``.
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
            panel_transcript_json=(
                json.dumps(panel_transcript, ensure_ascii=False)
                if panel_transcript is not None
                else None
            ),
            analysis_kind=analysis_kind,
            source=source,
        )

        # On conflict, update the existing row. SQLite dialect requires
        # index_elements (constraint= is PostgreSQL-only — E2E-discovered bug:
        # the L3 persistence path crashed with TypeError at runtime).
        stmt = stmt.on_conflict_do_update(
            index_elements=["user_id", "date", "analysis_kind"],
            set_={
                "procrastination_types_json": stmt.excluded.procrastination_types_json,
                "type_confidence_json": stmt.excluded.type_confidence_json,
                "cognitive_distortions_json": stmt.excluded.cognitive_distortions_json,
                "cbt_technique": stmt.excluded.cbt_technique,
                "response_text": stmt.excluded.response_text,
                "llm_model": stmt.excluded.llm_model,
                "llm_cost_usd": stmt.excluded.llm_cost_usd,
                "panel_transcript_json": stmt.excluded.panel_transcript_json,
                "analysis_kind": stmt.excluded.analysis_kind,
                "source": stmt.excluded.source,
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
        result["llm_model"] = row.llm_model
    # New "source" column (degradation tier) takes precedence over
    # the legacy convention of reading source from llm_model.
    source_col = getattr(row, "source", None)
    if source_col:
        result["source"] = source_col
    elif row.llm_model:
        result["source"] = row.llm_model
    if row.panel_transcript_json:
        result["panel_transcript"] = json.loads(row.panel_transcript_json)
    if row.analysis_kind:
        result["analysis_kind"] = row.analysis_kind

    return result
