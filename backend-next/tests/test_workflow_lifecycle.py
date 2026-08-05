"""RED→GREEN tests for the analysis workflow lifecycle fixes.

Covers:
  - Cached fallback rows retain real degradation metadata on BOTH cache
    paths (``cache_idempotency_check_node`` and the budget re-check in
    ``budget_reserve_node``).
  - Budget reservation rows carry an exact 24h TTL (``expires_at``).
  - Per-run reservation ownership: an owner failure releases its own
    reservation; a failing non-owner concurrent run never deletes the
    winner's reservation.
  - Persisted runs transition ``pending → running → completed/failed``
    so stale-run recovery can observe ``running``.
  - The initial graph state carries a single effective ``force`` entry
    that merges ``request.force`` and ``request.retry_if_degraded``.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
import sqlalchemy as sa

from mindflow.domain.procrastination import RuleEngine
from mindflow.graph.analysis_graph import (
    AnalysisGraph,
    AnalysisRunContext,
    budget_reserve_node,
    cache_idempotency_check_node,
)
from mindflow.infrastructure.repositories.workflow_runs import (
    BudgetReservationRepository,
    WorkflowRunsRepository,
)
from mindflow.infrastructure.schema import (
    workflow_budget_reservations,
    workflow_runs,
)
from mindflow.ports import AnalysisRequest

# Key derived by run_analysis for the fixture request below.
_IDEMPOTENCY_KEY = "api:1:2026-07-29:daily_attribution"


@pytest.fixture
async def workflow_tables(engine) -> None:
    """Create the workflow orchestration tables in the temp SQLite DB."""
    async with engine.begin() as conn:
        await conn.run_sync(workflow_runs.metadata.create_all)


@pytest.fixture
async def budget_repo(session_factory, workflow_tables) -> BudgetReservationRepository:
    """Real BudgetReservationRepository over SQLite (shared by run contexts)."""
    return BudgetReservationRepository(session_factory=session_factory)


@pytest.fixture
async def workflow_run_repo(session_factory, workflow_tables) -> WorkflowRunsRepository:
    """Real WorkflowRunsRepository over SQLite."""
    return WorkflowRunsRepository(session_factory=session_factory)


def _mock_analysis_repo(cached=None) -> AsyncMock:
    repo = AsyncMock()
    repo.get_by_date.return_value = cached
    repo.upsert.return_value = None
    return repo


def _failing_evidence_builder() -> AsyncMock:
    builder = AsyncMock()
    builder.build.side_effect = RuntimeError("evidence builder boom")
    return builder


def _success_evidence_builder() -> AsyncMock:
    builder = AsyncMock()
    bundle = AsyncMock()
    bundle.user_id = 1
    bundle.window = (
        datetime(2026, 7, 29, 0, 0, tzinfo=UTC),
        datetime(2026, 7, 30, 0, 0, tzinfo=UTC),
    )
    bundle.items = ()
    bundle.behavior_summary = AsyncMock(
        duration_min=120.0,
        actual_focus_min=80.0,
        context_switches_per_hour=8.0,
        longest_focus_block_s=1200.0,
        social_media_ratio=0.1,
        start_delay_min=5.0,
        baseline_deviation=0.5,
    )
    bundle.intervention_history = ()
    bundle.novelty_flags = ()
    bundle.events = ()
    builder.build.return_value = bundle
    return builder


def _request(*, force: bool = False, retry_if_degraded: bool = False) -> AnalysisRequest:
    return AnalysisRequest(
        user_id=1,
        target_date=date(2026, 7, 29),
        force=force,
        retry_if_degraded=retry_if_degraded,
        origin="api",
    )


def _graph(**overrides) -> AnalysisGraph:
    """Build a fully-wired AnalysisGraph; override any collaborator by kwarg."""
    defaults: dict = {
        "analysis_repo": _mock_analysis_repo(),
        "workflow_run_repo": AsyncMock(spec=WorkflowRunsRepository),
        "budget_repo": AsyncMock(spec=BudgetReservationRepository),
        "evidence_builder": _success_evidence_builder(),
        "crisis_detector": AsyncMock(),
        "panel_graph": None,
        "deepseek_client": None,
        "ollama_base_url": None,
        "ollama_model": "qwen3:8b",
        "rule_engine": RuleEngine(),
        "timezone": "local",
    }
    defaults.update(overrides)
    return AnalysisGraph(**defaults)


# ═══════════════════════════════════════════════════════════════════════════════
# Cached fallback metadata on both cache paths
# ═══════════════════════════════════════════════════════════════════════════════


class TestCacheDegradationMetadata:
    async def test_cache_check_retains_degradation_metadata(self) -> None:
        """A cached fallback must keep its degraded/source/degradation_path."""
        cached = {
            "source": "ollama",
            "degraded": True,
            "degradation_path": ["deepseek", "ollama"],
            "procrastination_types": ["impulsivity"],
        }
        runtime = AnalysisRunContext(analysis_repo=_mock_analysis_repo(cached))
        state = {
            "user_id": 1,
            "target_date": date(2026, 7, 29),
            "force": False,
            "runtime": runtime,
        }

        result = await cache_idempotency_check_node(state)  # type: ignore[arg-type]

        assert result["cache_hit"] is True
        assert result["source"] == "ollama"
        assert result["degraded"] is True
        assert result["degradation_path"] == ["deepseek", "ollama"]

    async def test_budget_recheck_retains_degradation_metadata(self) -> None:
        """Budget-loser re-check of the cache must keep degradation metadata."""
        cached = {
            "source": "rule_engine",
            "degraded": True,
            "degradation_path": ["deepseek", "ollama", "rule_engine"],
            "procrastination_types": ["impulsivity"],
        }
        budget_repo = AsyncMock()
        budget_repo.try_reserve.return_value = False
        runtime = AnalysisRunContext(
            analysis_repo=_mock_analysis_repo(cached),
            budget_repo=budget_repo,
        )
        state = {
            "user_id": 1,
            "target_date": date(2026, 7, 29),
            "idempotency_key": _IDEMPOTENCY_KEY,
            "runtime": runtime,
        }

        result = await budget_reserve_node(state)  # type: ignore[arg-type]

        assert result["budget_reserved"] is False
        assert result["cache_hit"] is True
        assert result["source"] == "rule_engine"
        assert result["degraded"] is True
        assert result["degradation_path"] == ["deepseek", "ollama", "rule_engine"]


# ═══════════════════════════════════════════════════════════════════════════════
# Reservation TTL over a real SQLite repository
# ═══════════════════════════════════════════════════════════════════════════════


class TestReservationTTL:
    async def test_try_reserve_writes_exact_24h_expiry(
        self, budget_repo: BudgetReservationRepository
    ) -> None:
        """try_reserve must persist expires_at = reserved_at + 24h."""
        wall_clock_before = datetime.now(UTC)

        assert await budget_repo.try_reserve(_IDEMPOTENCY_KEY) is True

        async with budget_repo._session_factory() as session:  # noqa: SLF001
            row = (
                await session.execute(
                    sa.select(workflow_budget_reservations).where(
                        workflow_budget_reservations.c.idempotency_key
                        == _IDEMPOTENCY_KEY
                    )
                )
            ).fetchone()

        assert row is not None
        reserved_at = datetime.fromisoformat(row.reserved_at)
        expires_at = datetime.fromisoformat(row.expires_at)

        # Exact 24h from the repository's own reserved_at timestamp.
        assert expires_at - reserved_at == timedelta(hours=24)

        # Bounded from the wall clock (source-derived tolerance).
        assert timedelta(hours=23, minutes=59) <= expires_at - wall_clock_before
        assert expires_at - wall_clock_before <= timedelta(hours=24, minutes=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-run reservation ownership
# ═══════════════════════════════════════════════════════════════════════════════


class TestReservationOwnership:
    async def test_owner_failure_releases_its_own_reservation(
        self,
        session_factory,
        workflow_run_repo: WorkflowRunsRepository,
        budget_repo: BudgetReservationRepository,
    ) -> None:
        """A run that won the reservation must release it on failure."""
        graph = _graph(
            analysis_repo=_mock_analysis_repo(),
            workflow_run_repo=workflow_run_repo,
            budget_repo=budget_repo,
            evidence_builder=_failing_evidence_builder(),
        )

        # Graph invocation fails after budget_reserve_node succeeded — the
        # run owns the reservation it created.
        result = await graph.run_analysis(_request())

        assert result.verdict is not None  # failure handled gracefully

        # The owner's reservation must have been released.
        async with budget_repo._session_factory() as session:  # noqa: SLF001
            row = (
                await session.execute(
                    sa.select(workflow_budget_reservations).where(
                        workflow_budget_reservations.c.idempotency_key
                        == _IDEMPOTENCY_KEY
                    )
                )
            ).fetchone()
        assert row is None, "owner failure must release its own reservation"

        # The key is immediately retryable.
        assert await budget_repo.try_reserve(_IDEMPOTENCY_KEY) is True

    async def test_failing_non_owner_keeps_winners_reservation(
        self,
        session_factory,
        workflow_run_repo: WorkflowRunsRepository,
        budget_repo: BudgetReservationRepository,
    ) -> None:
        """A failing concurrent run that never won must not delete the winner's row."""

        class FailingReserveBudgetRepo(BudgetReservationRepository):
            """Shares the real release() but its reserve attempt raises."""

            def __init__(self, sf) -> None:
                super().__init__(sf)
                self.release_calls = 0

            async def try_reserve(self, idempotency_key, *, cost_estimate: float = 1.0) -> bool:
                raise RuntimeError("reserve call boom")

            async def release(self, idempotency_key: str) -> None:
                self.release_calls += 1
                await super().release(idempotency_key)

        # Winner (run context 1) holds the reservation via the real repo.
        assert await budget_repo.try_reserve(_IDEMPOTENCY_KEY) is True

        # Loser (run context 2) shares the real reservation repository but
        # its reserve attempt raises → run_analysis failure path.
        loser_repo = FailingReserveBudgetRepo(session_factory)
        graph = _graph(
            analysis_repo=_mock_analysis_repo(),
            workflow_run_repo=workflow_run_repo,
            budget_repo=loser_repo,
            evidence_builder=_failing_evidence_builder(),
        )

        result = await graph.run_analysis(_request())

        assert result.verdict is not None  # failure handled gracefully
        assert loser_repo.release_calls == 0, (
            "non-owner failure must not release the shared reservation"
        )

        # The winner's reservation is still present.
        async with budget_repo._session_factory() as session:  # noqa: SLF001
            row = (
                await session.execute(
                    sa.select(workflow_budget_reservations).where(
                        workflow_budget_reservations.c.idempotency_key
                        == _IDEMPOTENCY_KEY
                    )
                )
            ).fetchone()
        assert row is not None, "winner's reservation must survive the loser's failure"


# ═══════════════════════════════════════════════════════════════════════════════
# Run status lifecycle over a real SQLite repository
# ═══════════════════════════════════════════════════════════════════════════════


class RecordingWorkflowRepo(WorkflowRunsRepository):
    """Delegates to the real repo and records every status update."""

    def __init__(self, sf) -> None:
        super().__init__(sf)
        self.statuses: list[str] = []

    async def update_status(self, run_id, status, *, result=None, error=None) -> None:
        self.statuses.append(status)
        await super().update_status(run_id, status, result=result, error=error)


async def _db_status(session_factory, run_id: str) -> str | None:
    """Fetch the persisted workflow run status directly from SQLite."""
    async with session_factory() as session:
        row = (
            await session.execute(
                sa.select(workflow_runs).where(workflow_runs.c.run_id == run_id)
            )
        ).fetchone()
    return row.status if row is not None else None


class TestRunStatusLifecycle:
    async def test_success_transitions_pending_running_completed(
        self,
        session_factory,
        workflow_tables,
        budget_repo: BudgetReservationRepository,
    ) -> None:
        """A successful run records running → completed; row ends completed."""
        recording_repo = RecordingWorkflowRepo(session_factory)
        graph = _graph(
            analysis_repo=_mock_analysis_repo(),
            workflow_run_repo=recording_repo,
            budget_repo=budget_repo,
            evidence_builder=_success_evidence_builder(),
        )

        result = await graph.run_analysis(_request())

        assert recording_repo.statuses == ["running", "completed"]
        assert await _db_status(session_factory, result.run_id) == "completed"

    async def test_failure_transitions_pending_running_failed(
        self,
        session_factory,
        workflow_tables,
        budget_repo: BudgetReservationRepository,
    ) -> None:
        """A failing run records running → failed; row ends failed."""
        recording_repo = RecordingWorkflowRepo(session_factory)
        graph = _graph(
            analysis_repo=_mock_analysis_repo(),
            workflow_run_repo=recording_repo,
            budget_repo=budget_repo,
            evidence_builder=_failing_evidence_builder(),
        )

        result = await graph.run_analysis(_request())

        assert recording_repo.statuses == ["running", "failed"]
        assert await _db_status(session_factory, result.run_id) == "failed"

    async def test_run_is_running_while_graph_executes(
        self,
        session_factory,
        workflow_tables,
        budget_repo: BudgetReservationRepository,
    ) -> None:
        """Stale-run recovery can observe 'running' mid-execution.

        A hard crash (no failure handler) after the graph starts leaves the
        run visible as ``running`` rather than stuck in ``pending`` forever.
        """
        workflow_repo = WorkflowRunsRepository(session_factory=session_factory)

        observed: dict[str, str | None] = {"status": None}
        evidence = _success_evidence_builder()

        async def hard_crash_build(*args, **kwargs):
            # Simulate a crash while the graph is mid-flight: the run must
            # already be persisted as "running".
            runs, _ = await workflow_repo.list_runs(limit=1)
            observed["status"] = str(runs[0]["status"]) if runs else None
            raise RuntimeError("hard crash mid-graph")

        evidence.build.side_effect = hard_crash_build
        graph = _graph(
            analysis_repo=_mock_analysis_repo(),
            workflow_run_repo=workflow_repo,
            budget_repo=budget_repo,
            evidence_builder=evidence,
        )

        await graph.run_analysis(_request())

        assert observed["status"] == "running"


# ═══════════════════════════════════════════════════════════════════════════════
# Initial force entry
# ═══════════════════════════════════════════════════════════════════════════════


class TestInitialForceEntry:
    async def test_retry_if_degraded_acts_as_force(self) -> None:
        """The single initial 'force' entry merges retry_if_degraded.

        Guards against the duplicate-key regression: if the surviving entry
        were only ``request.force``, a retry_if_degraded request would hit the
        cache instead of re-running.
        """
        cached = {
            "source": "panel",
            "degraded": False,
            "degradation_path": [],
            "procrastination_types": ["impulsivity"],
        }
        budget_repo = AsyncMock(spec=BudgetReservationRepository)
        budget_repo.try_reserve.return_value = True
        analysis_repo = _mock_analysis_repo(cached)

        graph = _graph(
            analysis_repo=analysis_repo,
            budget_repo=budget_repo,
        )

        await graph.run_analysis(_request(retry_if_degraded=True))

        # Cache was bypassed (force effective True) → reservation attempted.
        budget_repo.try_reserve.assert_awaited_once()
        # A fresh analysis was persisted.
        analysis_repo.upsert.assert_awaited_once()
