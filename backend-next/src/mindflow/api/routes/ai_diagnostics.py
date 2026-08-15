"""Read-only diagnostics endpoints for local workflow run inspection.

Endpoints:
  GET /api/v1/ai/runs           — List recent workflow runs (paginated, metadata only)
  GET /api/v1/ai/runs/{run_id}  — Single run detail with sanitised node events

All responses are allowlisted — no prompts, chat content, evidence values,
window titles, or API keys are ever returned.  Each endpoint requires the
same authentication as the rest of the API (auth middleware gates before
rate-limit metering).

Rate limit: global bucket only (no per-endpoint cap — diagnostics are cheap
local reads).  The pagination ceiling mirrors the existing convention:
``limit ≤ 100``, default 20.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Request  # noqa: B008
from loguru import logger

from mindflow.api.deps import get_workflow_runs_repo
from mindflow.api.schemas import DiagnosticsListResponse, NodeEventSummary, RunDetail, RunSummary

router = APIRouter(tags=["ai-diagnostics"])

# ── Constants ───────────────────────────────────────────────────────────

_DEFAULT_LIMIT: int = 20
_MAX_LIMIT: int = 100


# ── Internal helpers ────────────────────────────────────────────────────


def _run_to_summary(raw: dict[str, object]) -> RunSummary:
    """Convert a raw repository row to an allowlisted RunSummary."""
    return RunSummary(
        run_id=_s(raw, "run_id"),
        workflow_name=_s(raw, "workflow_name"),
        status=_s(raw, "status"),
        graph_version=_os(raw, "graph_version"),
        source=_os(raw, "source"),
        origin=_s(raw, "origin"),
        started_at=_os(raw, "started_at"),
        completed_at=_os(raw, "completed_at"),
        duration_ms=_oi(raw, "duration_ms"),
        call_count=_oi(raw, "call_count"),
        degraded=_ob(raw, "degradation_reason"),
    )


def _run_to_detail(raw: dict[str, object], events: list[dict[str, object]]) -> RunDetail:
    """Convert a raw repository row + node events to an allowlisted RunDetail."""
    return RunDetail(
        run_id=_s(raw, "run_id"),
        workflow_name=_s(raw, "workflow_name"),
        status=_s(raw, "status"),
        graph_version=_os(raw, "graph_version"),
        source=_os(raw, "source"),
        origin=_s(raw, "origin"),
        started_at=_os(raw, "started_at"),
        completed_at=_os(raw, "completed_at"),
        duration_ms=_oi(raw, "duration_ms"),
        call_count=_oi(raw, "call_count"),
        token_estimate=_oi(raw, "token_count"),
        degraded=_ob(raw, "degradation_reason"),
        node_events=[
            NodeEventSummary(
                node_name=_s(e, "node_name"),
                status=_s(e, "status"),
                started_at=_os(e, "started_at"),
                completed_at=_os(e, "completed_at"),
                duration_ms=_oi(e, "duration_ms"),
                error_category=_os(e, "error_category"),
            )
            for e in events
        ],
    )


def _s(d: dict[str, object], key: str) -> str:
    return str(d.get(key, ""))


def _os(d: dict[str, object], key: str) -> str | None:
    val = d.get(key)
    return str(val) if val is not None else None


def _oi(d: dict[str, object], key: str) -> int | None:
    val = d.get(key)
    if val is None:
        return None
    if isinstance(val, (int, float, str)):
        try:
            return int(val)
        except (TypeError, ValueError):
            return None
    return None


def _ob(d: dict[str, object], key: str) -> bool:
    """True when *key* has a non-null, non-empty value (degraded flag)."""
    val = d.get(key)
    return val is not None and bool(val)


# ── Routes ──────────────────────────────────────────────────────────────


@router.get("/ai/runs", response_model=DiagnosticsListResponse)
async def list_workflow_runs(
    request: Request,
    limit: int = Query(_DEFAULT_LIMIT, ge=1, le=_MAX_LIMIT, description="Max runs to return"),
    offset: int = Query(0, ge=0, description="Number of runs to skip"),
    repo: Any = Depends(get_workflow_runs_repo),  # noqa: B008
) -> DiagnosticsListResponse:
    """List recent workflow runs (most-recent-first).

    Returns allowlisted metadata only — no prompts, chat content,
    evidence values, window titles, or API keys.
    """
    logger.debug(
        "GET /ai/runs limit={} offset={} client={}",
        limit, offset, request.client.host if request.client else "-",
    )
    try:
        rows, total = await repo.list_runs(limit=limit, offset=offset)
    except Exception:
        logger.exception("Failed to list workflow runs")
        from mindflow.api.errors import _internal_error

        raise _internal_error() from None

    items = [_run_to_summary(r) for r in rows]
    next_off = offset + len(items) if offset + len(items) < total else None

    return DiagnosticsListResponse(
        items=items,
        count=total,
        has_more=next_off is not None,
        next_offset=next_off,
    )


@router.get("/ai/runs/{run_id}", response_model=RunDetail)
async def get_workflow_run(
    run_id: str,
    request: Request,
    repo: Any = Depends(get_workflow_runs_repo),  # noqa: B008
) -> RunDetail:
    """Return a single workflow run with sanitised node events.

    The response is allowlisted — structural metadata only.  Returns
    RFC 9457 ``404 Not Found`` for unknown run IDs.
    """
    logger.debug(
        "GET /ai/runs/{} client={}",
        run_id, request.client.host if request.client else "-",
    )
    try:
        raw = await repo.get_run(run_id)
    except Exception:
        logger.exception("Failed to fetch workflow run {}", run_id)
        from mindflow.api.errors import _internal_error

        raise _internal_error() from None

    if raw is None:
        from mindflow.api.errors import _not_found

        raise _not_found(f"工作流运行 {run_id}")

    # get_run() verified the run exists; now fetch the full row for
    # the detail response (get_run_detail returns a dict with every column).
    full = await repo.get_run_detail(run_id)
    if full is None:
        from mindflow.api.errors import _not_found

        raise _not_found(f"工作流运行 {run_id}")

    try:
        events = await repo.get_node_events(run_id)
    except Exception:
        logger.exception("Failed to fetch node events for {}", run_id)
        from mindflow.api.errors import _internal_error

        raise _internal_error() from None

    return _run_to_detail(full, events)



@router.get("/ai/graph")
async def get_analysis_graph_topology(
    request: Request,
) -> dict[str, Any]:
    """Return the AnalysisGraph node/edge topology for diagnostics.

    Architecture plan G/1.2: lets the user see how the analysis pipeline is
    wired (cache check -> evidence -> crisis gate -> panel -> fallbacks ->
    persistence). Pure structural metadata — no user data involved.
    """
    analysis_workflow = getattr(request.app.state, "workflow_port", None)
    if analysis_workflow is None:
        return {"nodes": [], "edges": [], "available": False}

    try:
        graph_obj = analysis_workflow._get_compiled_graph()
        raw_graph = graph_obj.get_graph()
        # LangGraph Edge objects expose source/target attributes (not tuples).
        nodes = sorted(
            {str(e.source) for e in raw_graph.edges}
            | {str(e.target) for e in raw_graph.edges}
        )
        edges = [
            {"from": str(e.source), "to": str(e.target)}
            for e in raw_graph.edges
        ]
        return {"nodes": nodes, "edges": edges, "available": True}
    except Exception:
        logger.exception("Failed to extract analysis graph topology")
        return {"nodes": [], "edges": [], "available": False}

