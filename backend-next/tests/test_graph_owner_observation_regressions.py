"""Confirmed review regressions: shared-run ownership and observed evidence."""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from datetime import UTC, date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from mindflow.agents.langchain_tools import make_query_evidence
from mindflow.domain.events import make_event
from mindflow.graph import analysis_graph as analysis
from mindflow.graph.chat_graph import ChatGraph, _has_tool_evidence
from mindflow.graph.tools import QueryEvidenceTool, ToolContext
from mindflow.infrastructure.repositories.chat import ChatRepository
from mindflow.infrastructure.repositories.workflow_runs import (
    BudgetReservationRepository,
    WorkflowRunsRepository,
)
from mindflow.infrastructure.security.crisis_detector import CrisisLevel
from mindflow.ports import AnalysisRequest
from mindflow.services.evidence_service import EvidenceBundleBuilder


@pytest.fixture
async def run_repositories(session_factory, create_tables):
    runs = WorkflowRunsRepository(session_factory)
    budget = BudgetReservationRepository(session_factory)
    runs.update_status = AsyncMock(wraps=runs.update_status)
    budget.release = AsyncMock(wraps=budget.release)
    return runs, budget


@pytest.mark.parametrize("outcome", ["cancel", "deadline", "error", "no_activity", "fallback"])
async def test_nonowner_never_mutates_shared_run(
    run_repositories, monkeypatch, outcome,
):
    runs, budget = run_repositories
    repo = SimpleNamespace(get_by_date=AsyncMock(return_value=None), upsert=AsyncMock())
    graph = analysis.AnalysisGraph(repo, runs, budget, AsyncMock(), MagicMock())
    owner_entered, waiter_entered = asyncio.Event(), asyncio.Event()
    owner_release = asyncio.Event()
    invocations = 0

    async def invoke(state, config=None):
        nonlocal invocations
        invocations += 1
        if invocations == 1:
            state.update(await analysis.budget_reserve_node(state))
            assert state["budget_reserved"]
            owner_entered.set()
            await owner_release.wait()
            state["assessment"] = {"types": [], "confidence": {}, "rationale": "owner"}
            state["source"] = "rule_engine"
            state.update(await analysis.terminal_persistence_node(state))
            return state
        waiter_entered.set()
        if outcome == "error":
            raise RuntimeError("synthetic waiter error")
        if outcome == "no_activity":
            raise analysis.NoActivityDataError("synthetic empty waiter")
        state.update(await analysis.budget_reserve_node(state))
        if outcome == "fallback":
            state["assessment"] = {"types": [], "confidence": {}, "rationale": "waiter"}
            state["source"] = "rule_engine"
            state.update(await analysis.terminal_persistence_node(state))
        return state

    graph._compiled = SimpleNamespace(ainvoke=invoke)
    request = AnalysisRequest(
        user_id=1, target_date=date(2026, 9, 19), idempotency_key="shared-owner-probe",
    )
    owner = asyncio.create_task(graph.run_analysis(request))
    waiter = None
    try:
        await asyncio.wait_for(owner_entered.wait(), 2)
        run_rows, count = await runs.list_runs()
        assert count == 1
        run_id = run_rows[0]["run_id"]
        assert (await runs.get_run(run_id)).status == "running"
        writes_before = runs.update_status.await_count
        if outcome == "deadline":
            monkeypatch.setattr(analysis, "_ANALYSIS_WORKFLOW_TIMEOUT_S", 0.03)
        if outcome == "fallback":
            monkeypatch.setattr(analysis, "_COMPETING_ANALYSIS_WAIT_TIMEOUT_S", 0.01)
        waiter = asyncio.create_task(graph.run_analysis(request))
        # The total deadline may expire during SQLite lookup, before graph entry.
        if outcome != "deadline":
            await asyncio.wait_for(waiter_entered.wait(), 2)
        if outcome == "cancel":
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
        elif outcome == "deadline":
            with pytest.raises(TimeoutError):
                await waiter
        elif outcome == "no_activity":
            with pytest.raises(analysis.NoActivityDataError):
                await waiter
        else:
            await asyncio.wait_for(waiter, 2)

        assert not owner.done()
        assert (await runs.get_run(run_id)).status == "running"
        assert runs.update_status.await_count == writes_before
        budget.release.assert_not_awaited()
        repo.upsert.assert_not_awaited()
        assert not await budget.try_reserve(request.idempotency_key)

        owner_release.set()
        await asyncio.wait_for(owner, 2)
        assert (await runs.get_run(run_id)).status == "completed"
        repo.upsert.assert_awaited_once()
        budget.release.assert_awaited_once_with(request.idempotency_key)
        assert await budget.try_reserve(request.idempotency_key)
        await budget.release(request.idempotency_key)
    finally:
        owner_release.set()
        for task in (owner, waiter):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            owner, *([waiter] if waiter is not None else []), return_exceptions=True,
        )


async def test_owner_cancellation_still_fails_run_and_releases_budget(run_repositories):
    runs, budget = run_repositories
    graph = analysis.AnalysisGraph(
        AsyncMock(), runs, budget, AsyncMock(), MagicMock(),
    )
    entered = asyncio.Event()

    async def invoke(state, config=None):
        state.update(await analysis.budget_reserve_node(state))
        entered.set()
        await asyncio.Event().wait()

    graph._compiled = SimpleNamespace(ainvoke=invoke)
    request = AnalysisRequest(
        user_id=1, target_date=date(2026, 9, 19), idempotency_key="cancel-owner",
    )
    task = asyncio.create_task(graph.run_analysis(request))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        rows, _ = await runs.list_runs()
        assert rows[0]["status"] == "failed"
        budget.release.assert_awaited_once_with(request.idempotency_key)
        assert await budget.try_reserve(request.idempotency_key)
        await budget.release(request.idempotency_key)
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.parametrize("node", [
    analysis.terminal_persistence_node, analysis.handle_persistence_failure_node,
])
async def test_nonowner_persistence_nodes_have_no_side_effects(node):
    repo, runs, budget = AsyncMock(), AsyncMock(), AsyncMock()
    runtime = analysis.AnalysisRunContext(
        analysis_repo=repo, workflow_run_repo=runs, budget_repo=budget,
    )
    await node({
        "runtime": runtime, "user_id": 1, "target_date": date(2026, 9, 19),
        "run_id": "owner-run", "idempotency_key": "owner-key",
        "budget_reserved": False,
        "assessment": {"types": [], "confidence": {}, "rationale": "cached"},
    })
    repo.upsert.assert_not_awaited()
    runs.update_status.assert_not_awaited()
    budget.release.assert_not_awaited()


async def test_cache_replay_cannot_overwrite_active_owner(run_repositories):
    runs, budget = run_repositories
    cached = {"types": [], "confidence": {}, "rationale": "cached", "source": "panel"}
    repo = SimpleNamespace(get_by_date=AsyncMock(return_value=cached), upsert=AsyncMock())
    request = AnalysisRequest(
        user_id=1, target_date=date(2026, 9, 19), idempotency_key="cached-owner",
    )
    from mindflow.ports import WorkflowRunRequest

    run_id = await runs.save_run(WorkflowRunRequest(
        user_id=1, target_date=request.target_date, idempotency_key=request.idempotency_key,
    ))
    await budget.try_reserve(request.idempotency_key)
    await runs.update_status(run_id, "running")
    runs.update_status.reset_mock()
    graph = analysis.AnalysisGraph(repo, runs, budget, AsyncMock(), MagicMock())
    result = await graph.run_analysis(request)
    assert result.verdict.cached
    assert (await runs.get_run(run_id)).status == "running"
    runs.update_status.assert_not_awaited()
    repo.upsert.assert_not_awaited()
    budget.release.assert_not_awaited()
    await budget.release(request.idempotency_key)


@pytest.mark.parametrize("has_activity", [False, True])
async def test_real_builder_tool_and_chat_require_observations(
    session_factory, create_tables, has_activity,
):
    events = [
        make_event(
            user_id=1, timestamp_utc=datetime.now(UTC), duration_s=120,
            process_name="Code.exe",
        ),
    ] if has_activity else []
    activity_repo = SimpleNamespace(query_overlapping_range=AsyncMock(return_value=events))
    builder = EvidenceBundleBuilder(
        activity_repo, AsyncMock(), session_factory, baseline_repo=AsyncMock(),
    )
    builder._load_baseline = AsyncMock(return_value=None)
    builder._build_intervention_history = AsyncMock(return_value=[])
    adapter = QueryEvidenceTool(builder)
    tool = make_query_evidence(adapter)
    token = adapter.bind_context(ToolContext(user_id=1, session_id="observation"))
    try:
        payload = json.loads(await tool.ainvoke({}))
    finally:
        adapter.reset_context(token)
    assert payload["evidence"], "Empty windows really do contain placeholder metrics"
    assert (payload["behavior_summary"]["duration_min"] > 0) is has_activity

    model = MagicMock()
    model.bind_tools.return_value = model
    model.ainvoke = AsyncMock(side_effect=[
        AIMessage(content="", tool_calls=[{
            "id": "call-observation", "name": "query_evidence", "args": {},
        }]),
        AIMessage(content="Observed evidence checked."),
    ])
    detector = MagicMock()
    detector.scan.return_value = (CrisisLevel.NONE, None)
    graph = ChatGraph(
        ChatRepository(session_factory), detector, model=model,
        tools=[tool], tool_adapters=[adapter], recursion_limit=30,
    )
    result = await graph.ask(1, "observation", "Check my activity")
    assert result.evidence_cited is has_activity
    assert result.tools_used == ("query_evidence",)


@pytest.mark.parametrize(("payload", "expected"), [
    ({"evidence": [{"metric": "focus_score"}]}, False),
    ({"evidence": [{"metric": "focus_score", "value": None}]}, False),
    ({"evidence": [{"metric": "focus_score", "value": {}}]}, False),
    ({"evidence": [{"metric": "focus_score", "value": False}]}, False),
    ({"evidence": [{"metric": "focus_score", "value": 0}]}, True),
    ({"evidence": [{"metric": "focus_score", "value": 80}]}, True),
    ({"evidence": [{"metric": "top_apps", "value": "Code"}]}, True),
    ({
        "evidence": [{"metric": "focus_score", "severity": "info"}],
        "behavior_summary": {"duration_min": 2},
    }, True),
    ({
        "evidence": [{"metric": "focus_score", "severity": "info"}],
        "behavior_summary": {"duration_min": 0},
    }, False),
])
def test_evidence_requires_an_observation_not_just_a_metric(payload, expected):
    assert _has_tool_evidence("query_evidence", json.dumps(payload)) is expected
