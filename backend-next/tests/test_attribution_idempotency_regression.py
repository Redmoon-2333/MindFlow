"""Regression test: second run_analysis for same date must return 200, not 500.

Root cause: save_run() in WorkflowRunsRepository uses bare INSERT without
ON CONFLICT handling. When the same idempotency_key is reused, the UNIQUE
constraint on workflow_runs.idempotency_key is violated, raising
IntegrityError → propagates through run_analysis() → caught by route
handler's generic Exception → HTTP 500.

Fix: save_run() must use ON CONFLICT (idempotency_key) DO NOTHING and
return the existing run_id when the key already exists.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import async_sessionmaker

from mindflow.domain.events import make_event
from mindflow.domain.evidence import EvidenceBundle
from mindflow.domain.procrastination import RuleEngine
from mindflow.graph.analysis_graph import AnalysisGraph
from mindflow.infrastructure.database import create_engine, create_session_factory
from mindflow.infrastructure.repositories.analysis import (
    SQLAlchemyProcrastinationAnalysisRepository,
)
from mindflow.infrastructure.repositories.workflow_runs import (
    WorkflowRunsRepository,
)
from mindflow.infrastructure.schema import procrastination_analyses
from mindflow.ports import AnalysisRequest


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_regression.db"


@pytest.fixture
def db_url(tmp_db_path: Path) -> str:
    return f"sqlite+aiosqlite:///{tmp_db_path}"


@pytest.fixture
async def engine(db_url: str):
    eng = create_engine(db_url)
    # Create ALL tables that the graph depends on
    async with eng.begin() as conn:
        await conn.run_sync(procrastination_analyses.metadata.create_all)
    try:
        yield eng
    finally:
        await eng.dispose()


@pytest.fixture
async def session_factory(engine) -> async_sessionmaker:
    return create_session_factory(engine)


@pytest.fixture
async def analysis_repo(session_factory) -> SQLAlchemyProcrastinationAnalysisRepository:
    return SQLAlchemyProcrastinationAnalysisRepository(session_factory=session_factory)


@pytest.fixture
async def workflow_run_repo(session_factory) -> WorkflowRunsRepository:
    """Real WorkflowRunsRepository — UNIQUE(idempotency_key) enforced."""
    return WorkflowRunsRepository(session_factory=session_factory)


@pytest.fixture
def mock_budget_repo() -> AsyncMock:
    repo = AsyncMock()
    repo.try_reserve.return_value = True
    repo.release.return_value = None
    return repo


@pytest.fixture
def mock_evidence_builder() -> AsyncMock:
    builder = AsyncMock()
    bundle = EvidenceBundle(
        user_id=1,
        window=(datetime(2026, 7, 29, 0, 0), datetime(2026, 7, 30, 0, 0)),
        items=(),
        behavior_summary=MagicMock(
            duration_min=120.0,
            actual_focus_min=80.0,
            context_switches_per_hour=8.0,
            longest_focus_block_s=1200.0,
            social_media_ratio=0.1,
            start_delay_min=5.0,
            baseline_deviation=0.5,
        ),
        intervention_history=(),
        novelty_flags=(),
        events=(
            make_event(user_id=1, timestamp_utc=datetime(2026, 7, 29, 12, 0, tzinfo=UTC)),
        ),
    )
    builder.build.return_value = bundle
    return builder


@pytest.fixture
def mock_crisis_detector() -> MagicMock:
    from mindflow.infrastructure.security.crisis_detector import CrisisLevel

    detector = MagicMock()
    detector.scan_texts.return_value = (CrisisLevel.NONE, None)
    return detector


@pytest.fixture
def analysis_graph(
    analysis_repo,
    workflow_run_repo,
    mock_budget_repo,
    mock_evidence_builder,
    mock_crisis_detector,
) -> AnalysisGraph:
    """AnalysisGraph with real analysis_repo AND real workflow_run_repo."""
    return AnalysisGraph(
        analysis_repo=analysis_repo,
        workflow_run_repo=workflow_run_repo,
        budget_repo=mock_budget_repo,
        evidence_builder=mock_evidence_builder,
        crisis_detector=mock_crisis_detector,
        panel_graph=None,
        deepseek_client=None,
        ollama_base_url=None,
        ollama_model="qwen3:8b",
        rule_engine=RuleEngine(),
        timezone="local",
    )


@pytest.mark.anyio
async def test_second_request_returns_200_not_500(
    analysis_graph: AnalysisGraph,
    analysis_repo: SQLAlchemyProcrastinationAnalysisRepository,
    workflow_run_repo: WorkflowRunsRepository,
) -> None:
    """Two identical run_analysis calls → both return AnalysisResult, no exception.

    Before fix: second call raises IntegrityError (UNIQUE constraint on
    idempotency_key in workflow_runs) which skips the graph's try/except
    and propagates as HTTP 500.
    """
    request = AnalysisRequest(
        user_id=1,
        target_date=date(2026, 7, 29),
        force=False,
        origin="api",
    )

    # ── First request: runs rule_engine, persists ──
    result1 = await analysis_graph.run_analysis(request)
    assert result1.verdict is not None, "First request must produce a verdict"
    assert result1.verdict.source == "rule_engine"
    assert len(result1.verdict.types) > 0

    # Verify persistence happened
    stored = await analysis_repo.get_by_date(
        1, date(2026, 7, 29), analysis_kind="daily_attribution"
    )
    assert stored is not None, "First request must persist analysis"
    assert stored.get("source") == "rule_engine"

    # Verify exactly one workflow run exists
    from mindflow.infrastructure.schema import workflow_runs

    async with workflow_run_repo._session_factory() as session:
        count_stmt = (
            sa.select(sa.func.count())
            .select_from(workflow_runs)
            .where(workflow_runs.c.idempotency_key == "api:1:2026-07-29:daily_attribution")
        )
        run_count = (await session.execute(count_stmt)).scalar()
        assert run_count == 1, f"Expected 1 workflow run, got {run_count}"

    # ── Second request: same idempotency_key → must NOT raise ──
    result2 = await analysis_graph.run_analysis(request)
    assert result2.verdict is not None, "Second request must produce a verdict (no IntegrityError)"
    assert result2.verdict.source == "rule_engine"

    # Verdict content must match the first run (cache replay)
    assert len(result2.verdict.types) > 0

    # Verify only ONE analysis row exists (no duplicate)
    async with analysis_repo._session_factory() as session:
        count_stmt = (
            sa.select(sa.func.count())
            .select_from(procrastination_analyses)
            .where(
                procrastination_analyses.c.user_id == 1,
                procrastination_analyses.c.date == "2026-07-29",
                procrastination_analyses.c.analysis_kind == "daily_attribution",
            )
        )
        row_count = (await session.execute(count_stmt)).scalar()
        assert row_count == 1, f"Expected 1 analysis row, got {row_count}"

    # Verify no extra workflow runs were created
    async with workflow_run_repo._session_factory() as session:
        count_stmt = (
            sa.select(sa.func.count())
            .select_from(workflow_runs)
            .where(workflow_runs.c.idempotency_key == "api:1:2026-07-29:daily_attribution")
        )
        run_count = (await session.execute(count_stmt)).scalar()
        assert run_count == 1, f"Expected 1 workflow run after second request, got {run_count}"
