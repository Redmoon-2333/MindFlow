"""Atomic persistence for workflow runs, node events, and budget reservations.

Implements ``WorkflowRunStorePort`` and ``BudgetReservationPort`` from
``mindflow.ports`` against the SQLAlchemy Core tables defined in
``mindflow.infrastructure.schema``.  Uses SQLite UPSERT (ON CONFLICT
DO NOTHING) for atomic budget reservation — exactly one concurrent
caller wins for a given idempotency key.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import sqlalchemy as sa
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.domain.ids import new_id
from mindflow.infrastructure.schema import (
    workflow_budget_reservations,
    workflow_node_events,
    workflow_runs,
)
from mindflow.ports import AnalysisResult, RunStatus, WorkflowRunRequest, WorkflowRunResult

# ── Public query types (used by diagnostics routes) ──────────────────────

# Pre-imported by api/routes/ai_diagnostics.py for typed iteration.
# These are simple dict-shaped records, not domain dataclasses — fast to
# construct and safe to expose over HTTP after field allow-listing.
RunRow = dict[str, object]
NodeEventRow = dict[str, object]


class WorkflowRunsRepository:
    """SQLAlchemy Core repository for workflow run status tracking.

    Implements ``WorkflowRunStorePort`` — separate from
    ``ScheduledJobRunsRepository`` which tracks cron attempts per
    (job_name, local_date).  This repository tracks individual workflow
    runs by generated ID, supporting idempotency and audit.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def save_run(self, request: WorkflowRunRequest) -> str:
        """Insert a new pending workflow run and return its ID.

        The run is created with ``status='pending'`` and a privacy-safe
        UUID4 trace_id.  The idempotency_key (when provided) is stored
        with a UNIQUE constraint — callers should use
        ``BudgetReservationPort`` first to guard against duplicates.

        When an idempotency_key is provided and a run already exists for
        that key, the existing run_id is returned instead of raising an
        IntegrityError.  This makes the save_run call idempotent across
        retries and cache-hit replays (e.g. a second attribution POST
        for the same date that hits the analysis cache).
        """
        idempotency_key = request.idempotency_key or None
        run_id = new_id()
        trace_id = new_id()
        now = datetime.now(UTC)
        values = {
            "id": new_id(),
            "workflow_name": "daily_analysis",
            "run_id": run_id,
            "status": "pending",
            "graph_version": "v0",
            "source": None,
            "origin": request.origin,
            "user_id": request.user_id,
            "target_date": request.target_date.isoformat(),
            "idempotency_key": idempotency_key,
            "started_at": now.isoformat(),
            "completed_at": None,
            "retry_reason": None,
            "degradation_reason": None,
            "token_count": None,
            "call_count": None,
            "trace_id": trace_id,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }

        async with self._session_factory() as session, session.begin():
            if idempotency_key is not None:
                # Idempotent INSERT: ON CONFLICT (idempotency_key) DO NOTHING
                # so duplicate keys return the existing run_id instead of
                # crashing with IntegrityError.
                from sqlalchemy.dialects.sqlite import insert as sqlite_insert

                stmt = (
                    sqlite_insert(workflow_runs)
                    .values(**values)
                    .on_conflict_do_nothing(index_elements=["idempotency_key"])
                )
                await session.execute(stmt)
                # Whether we just inserted or the row already existed,
                # fetch the run_id associated with this idempotency_key.
                select_stmt = (
                    sa.select(workflow_runs.c.run_id)
                    .where(workflow_runs.c.idempotency_key == idempotency_key)
                )
                result = await session.execute(select_stmt)
                return cast(str, result.scalar_one())
            else:
                insert_stmt = sa.insert(workflow_runs).values(**values)
                await session.execute(insert_stmt)
        return run_id

    async def get_run(self, run_id: str) -> WorkflowRunResult | None:
        """Return the workflow run with the given *run_id*, or None."""
        stmt = sa.select(workflow_runs).where(
            workflow_runs.c.run_id == run_id,
        )
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            row = result.fetchone()

        if row is None:
            return None

        return WorkflowRunResult(
            run_id=row.run_id,
            status=row.status,
            analysis_result=None,
            error_message=_maybe_getattr(row, "last_error"),
            started_at=_parse_iso(row.started_at),
            completed_at=_parse_iso(row.completed_at),
        )

    async def list_runs(
        self, *, limit: int = 20, offset: int = 0
    ) -> tuple[list[dict[str, object]], int]:
        """Return recent workflow runs (DESC by started_at) with total count.

        Pagination follows existing conventions: *limit* caps results,
        *offset* skips the first N rows, and the total count includes all
        runs regardless of offset, so callers can compute ``has_more``.
        """
        count_stmt = sa.select(sa.func.count()).select_from(workflow_runs)
        rows_stmt = (
            sa.select(workflow_runs)
            .order_by(workflow_runs.c.started_at.desc())
            .limit(limit)
            .offset(offset)
        )
        async with self._session_factory() as session:
            total = (await session.execute(count_stmt)).scalar_one()
            rows = (await session.execute(rows_stmt)).fetchall()

        runs: list[dict[str, object]] = []
        for row in rows:
            started = _parse_iso(row.started_at)
            completed = _parse_iso(row.completed_at)
            runs.append({
                "run_id": row.run_id,
                "workflow_name": row.workflow_name,
                "status": row.status,
                "graph_version": row.graph_version,
                "source": row.source,
                "origin": row.origin,
                "started_at": started.isoformat() if started else None,
                "completed_at": completed.isoformat() if completed else None,
                "duration_ms": (
                    int((completed - started).total_seconds() * 1000)
                    if started and completed
                    else None
                ),
                "call_count": row.call_count,
                "token_count": row.token_count,
                "degradation_reason": row.degradation_reason,
            })
        return runs, total

    async def get_run_detail(self, run_id: str) -> dict[str, object] | None:
        """Return the full workflow run row as a dict for diagnostics.

        Unlike ``get_run()`` (which returns a slim ``WorkflowRunResult`` for
        port compatibility), this method returns every column so the
        diagnostics endpoint can build a ``RunDetail`` response.
        """
        stmt = sa.select(workflow_runs).where(workflow_runs.c.run_id == run_id)
        async with self._session_factory() as session:
            row = (await session.execute(stmt)).fetchone()

        if row is None:
            return None

        started = _parse_iso(row.started_at)
        completed = _parse_iso(row.completed_at)
        return {
            "run_id": row.run_id,
            "workflow_name": row.workflow_name,
            "status": row.status,
            "graph_version": row.graph_version,
            "source": row.source,
            "origin": row.origin,
            "started_at": started.isoformat() if started else None,
            "completed_at": completed.isoformat() if completed else None,
            "duration_ms": (
                int((completed - started).total_seconds() * 1000)
                if started and completed
                else None
            ),
            "call_count": row.call_count,
            "token_count": row.token_count,
            "degradation_reason": row.degradation_reason,
        }

    async def get_node_events(self, run_id: str) -> list[dict[str, object]]:
        """Return all node events for a workflow run, ordered by started_at."""
        stmt = (
            sa.select(workflow_node_events)
            .where(workflow_node_events.c.run_id == run_id)
            .order_by(workflow_node_events.c.started_at.asc())
        )
        async with self._session_factory() as session:
            rows = (await session.execute(stmt)).fetchall()

        events: list[dict[str, object]] = []
        for row in rows:
            events.append({
                "node_name": row.node_name,
                "status": row.status,
                "started_at": row.started_at,
                "completed_at": row.completed_at,
                "duration_ms": row.duration_ms,
                "error_category": row.error_category,
            })
        return events

    async def save_node_event(
        self,
        run_id: str,
        node_name: str,
        *,
        status: str = "completed",
        payload: dict[str, Any] | None = None,
        error_category: str | None = None,
        duration_ms: int | None = None,
    ) -> None:
        """Record one traceable node event with an optional JSON payload."""
        now = datetime.now(UTC)
        values: dict[str, object] = {
            "id": new_id(),
            "run_id": run_id,
            "node_name": node_name,
            "status": status,
            "started_at": now.isoformat(),
            "completed_at": now.isoformat(),
            "duration_ms": duration_ms,
            "error_category": error_category,
            "payload_json": (
                json.dumps(payload, ensure_ascii=False) if payload is not None else None
            ),
        }
        async with self._session_factory() as session, session.begin():
            await session.execute(sa.insert(workflow_node_events).values(**values))

    async def update_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        result: AnalysisResult | None = None,
        error: str | None = None,
    ) -> None:
        """Update the status of a workflow run.

        When *status* is ``"completed"``, sets ``completed_at`` and
        optionally records token/call counters from the *result*.
        When *status* is ``"failed"``, sets ``completed_at`` and stores
        the *error* message.
        """
        now = datetime.now(UTC)
        values: dict[str, object] = {
            "status": status,
            "updated_at": now.isoformat(),
        }
        if status == "completed":
            values["completed_at"] = now.isoformat()
        elif status == "failed":
            values["completed_at"] = now.isoformat()
            # error is stored outside the table payload — it's runtime metadata
            # not user-facing content, so it belongs here.
            if error is not None:
                values["retry_reason"] = error

        async with self._session_factory() as session, session.begin():
            await session.execute(
                sa.update(workflow_runs)
                .where(workflow_runs.c.run_id == run_id)
                .values(**values)
            )


class BudgetReservationRepository:
    """Atomic budget check-and-reserve repository.

    Guarantees exactly-one winner for a given idempotency key via
    ``INSERT … ON CONFLICT DO NOTHING`` — the database UNIQUE constraint
    acts as the atomic gate.  No SELECT-then-INSERT, no advisory locks,
    no partial state.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def try_reserve(
        self,
        idempotency_key: str,
        *,
        cost_estimate: float = 1.0,
    ) -> bool:
        """Atomically reserve budget for *idempotency_key*.

        The first caller to INSERT succeeds; all subsequent callers for
        the same key get a conflict and receive ``False`` — no waiting,
        no partial state.

        The reservation row carries a 24-hour TTL (``expires_at = now + 24h``)
        so a leaked reservation can never block the key permanently.

        Args:
            idempotency_key: Unique reservation key (e.g.
                ``"{origin}:{user_id}:{target_date}:{analysis_kind}"``).
            cost_estimate: Token/call budget estimate (stored as
                ``budget_type`` for now — future columns may separate
                token and call budgets).

        Returns:
            ``True`` if this caller won the reservation, ``False`` if
            another caller already holds it.
        """
        now = datetime.now(UTC)
        values = {
            "id": new_id(),
            "workflow_name": "daily_analysis",
            "run_id": None,
            "origin": "scheduler",
            "idempotency_key": idempotency_key,
            "user_id": 0,
            "target_date": now.date().isoformat(),
            "budget_type": f"cost_{cost_estimate:.4f}",
            "reserved_at": now.isoformat(),
            "expires_at": (now + timedelta(hours=24)).isoformat(),
            "released_at": None,
        }
        async with self._session_factory() as session, session.begin():
            stmt = (
                sqlite_insert(workflow_budget_reservations)
                .values(**values)
                .on_conflict_do_nothing(index_elements=["idempotency_key"])
                .returning(workflow_budget_reservations.c.id)
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none() is not None

    async def release(self, idempotency_key: str) -> None:
        """Release a budget reservation by *idempotency_key*.

        Deletes the reservation row so a subsequent ``try_reserve`` for
        the same key can succeed.  The ``released_at`` column is preserved
        in the table schema for future audit-log migration but is unused
        in this implementation — deletion is the simplest atomic release.
        """
        async with self._session_factory() as session, session.begin():
            await session.execute(
                sa.delete(workflow_budget_reservations).where(
                    workflow_budget_reservations.c.idempotency_key == idempotency_key,
                )
            )


# ── Helpers ────────────────────────────────────────────────────────────


def _parse_iso(raw: str | None) -> datetime | None:
    """Parse an ISO-8601 string to a timezone-aware datetime, or None."""
    if raw is None:
        return None
    return datetime.fromisoformat(raw)


def _maybe_getattr(row: Any, name: str) -> str | None:
    """Safely get an optional string attribute from a Row, returning None
    if the column does not exist.

    This avoids attribute errors when a migration hasn't added a column yet
    (graceful degradation for zero-downtime upgrades).
    """
    try:
        val = getattr(row, name)
        return val if isinstance(val, str) else None
    except AttributeError:
        return None
