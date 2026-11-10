"""Regression: the contract §3 ``tmt`` payload must survive cache replay.

Root cause (2026-11-10, offline demo): ``terminal_persistence_node`` only built
the ``panel_transcript`` metadata block when the verdict carried a non-empty
deliberation transcript.  The L3 rule-engine verdict never has one, so the
block — and with it the deterministic ``deviation``/``tmt`` payloads — was
dropped from the persisted row.  The first request returned ``tmt`` from graph
state; every later request read ``panel_transcript_json`` back as NULL and
reported ``tmt: null`` ("暂无动机分解数据").

These tests run the real graph, the real repositories and a real RuleEngine
with no LLM configured, which is exactly the offline demo path.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.domain.events import make_event
from mindflow.domain.procrastination import RuleEngine
from mindflow.domain.tmt import TMT_NOTE
from mindflow.graph.analysis_graph import AnalysisGraph
from mindflow.infrastructure.database import create_engine, create_session_factory
from mindflow.infrastructure.repositories.activity import SQLAlchemyActivityRepository
from mindflow.infrastructure.repositories.analysis import (
    SQLAlchemyProcrastinationAnalysisRepository,
)
from mindflow.infrastructure.repositories.intervention import InterventionLogRepository
from mindflow.infrastructure.repositories.tasks import SQLAlchemyTaskRepository
from mindflow.infrastructure.repositories.workflow_runs import (
    BudgetReservationRepository,
    WorkflowRunsRepository,
)
from mindflow.infrastructure.schema import metadata as schema_metadata
from mindflow.ports import AnalysisRequest
from mindflow.services.evidence_service import EvidenceBundleBuilder

_TARGET_DATE = date(2026, 11, 10)
_TMT_COMPONENTS = ("expectancy", "value", "impulsiveness", "delay")


@pytest.fixture
async def session_factory(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_engine(f"sqlite+aiosqlite:///{tmp_path / 'tmt.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(schema_metadata.create_all)
    try:
        yield create_session_factory(engine)
    finally:
        await engine.dispose()


@pytest.fixture
async def offline_graph(session_factory: async_sessionmaker[AsyncSession]) -> AnalysisGraph:
    """AnalysisGraph wired to real repositories, with every LLM tier absent."""
    activity_repo = SQLAlchemyActivityRepository(
        session_factory=session_factory, pulsetime_s=10,
    )
    await _seed_events(activity_repo)

    builder = EvidenceBundleBuilder(
        activity_repo=activity_repo,
        intervention_repo=InterventionLogRepository(session_factory=session_factory),
        session_factory=session_factory,
        task_repo=SQLAlchemyTaskRepository(session_factory=session_factory),
    )
    return AnalysisGraph(
        analysis_repo=SQLAlchemyProcrastinationAnalysisRepository(
            session_factory=session_factory,
        ),
        workflow_run_repo=WorkflowRunsRepository(session_factory),
        budget_repo=BudgetReservationRepository(session_factory),
        evidence_builder=builder,
        crisis_detector=_NoCrisisDetector(),
        panel_graph=None,
        deepseek_client=None,
        ollama_base_url=None,
        ollama_model="qwen3:8b",
        rule_engine=RuleEngine(),
        timezone="UTC",
    )


class _NoCrisisDetector:
    """Crisis detector stub: the seeded text is benign."""

    def scan_texts(self, texts: list[str]) -> tuple[Any, None]:
        from mindflow.infrastructure.security.crisis_detector import CrisisLevel

        return CrisisLevel.NONE, None


async def _seed_events(repo: SQLAlchemyActivityRepository) -> None:
    base = datetime(2026, 11, 10, 8, 0, tzinfo=UTC)
    for index in range(60):
        await repo.append_event(
            make_event(
                user_id=1,
                timestamp_utc=base + timedelta(seconds=index * 60),
                duration_s=60.0,
                process_name="code.exe",
                app_name="VS Code",
                window_title="main.py",
                is_idle=False,
            )
        )


async def test_offline_attribution_keeps_tmt_on_cache_hit(
    offline_graph: AnalysisGraph,
) -> None:
    """First run and the cache-hit replay must both carry a full TMT breakdown."""
    request = AnalysisRequest(
        user_id=1, target_date=_TARGET_DATE, force=False, origin="api",
    )

    first = await offline_graph.run_analysis(request)
    assert first.verdict.source == "rule_engine"
    assert first.verdict.tmt is not None, "first run lost the TMT payload"
    _assert_full_breakdown(first.verdict.tmt)

    second = await offline_graph.run_analysis(request)
    assert second.verdict.cached is True
    assert second.verdict.tmt is not None, "cache hit lost the TMT payload"
    _assert_full_breakdown(second.verdict.tmt)
    assert dict(second.verdict.tmt) == dict(first.verdict.tmt)


def _assert_full_breakdown(tmt: Any) -> None:
    breakdown = dict(tmt)
    assert set(_TMT_COMPONENTS) <= set(breakdown["components"])
    assert 0.0 <= breakdown["motivation"] <= 1.0
    assert breakdown["motivation_level"] in {"low", "medium", "high"}
    assert breakdown["dominant_barrier"] in set(_TMT_COMPONENTS)
    assert breakdown["note"] == TMT_NOTE
