"""Procrastination analysis repository persistence tests.

Covers:
  - panel_metadata round-trips through upsert → get_by_date
  - analysis_kind parameter behavior (filter by kind, None default)
  - multiple analysis kinds coexisting for same day
"""

from __future__ import annotations

from datetime import date

from mindflow.infrastructure.repositories.analysis import (
    SQLAlchemyProcrastinationAnalysisRepository,
    procrastination_analyses,
)


async def test_panel_metadata_round_trips(engine, session_factory) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(procrastination_analyses.metadata.create_all)
    repository = SQLAlchemyProcrastinationAnalysisRepository(session_factory)
    target_date = date(2026, 7, 25)
    panel_metadata = {
        "transcript": [
            {"role": "数据分析师", "content": "模式分析完成", "round": 0},
        ],
        "dissent": ["少数意见"],
        "escalated": True,
        "call_count": 9,
    }

    await repository.upsert(
        user_id=1,
        target_date=target_date,
        procrastination_types=["impulsivity"],
        type_confidence={"impulsivity": 0.85},
        cognitive_distortions=[],
        cbt_technique="stimulus_control",
        response_text="测试会诊结果",
        llm_model="panel",
        panel_transcript=panel_metadata,
    )

    stored = await repository.get_by_date(1, target_date)

    assert stored is not None
    assert stored["source"] == "panel"
    assert stored["panel_transcript"] == panel_metadata


# ═══════════════════════════════════════════════════════════════════════════════
# analysis_kind parameter behavior — filter by kind, None returns first match
# ═══════════════════════════════════════════════════════════════════════════════


class TestAnalysisKindFilter:
    """analysis_kind parameter on get_by_date() behaves correctly."""

    async def test_filter_by_kind_matches_exact(self, engine, session_factory) -> None:
        """When analysis_kind is given, only rows with that kind are returned."""
        async with engine.begin() as connection:
            await connection.run_sync(procrastination_analyses.metadata.create_all)

        repository = SQLAlchemyProcrastinationAnalysisRepository(session_factory)
        target_date = date(2026, 7, 29)

        # Insert daily_panel
        await repository.upsert(
            user_id=1,
            target_date=target_date,
            procrastination_types=["impulsivity"],
            type_confidence={"impulsivity": 0.85},
            cognitive_distortions=[],
            cbt_technique="stimulus_control",
            response_text="面板结果",
            llm_model="panel",
            analysis_kind="daily_panel",
            source="panel",
        )

        # Insert daily_attribution (same day, different kind)
        await repository.upsert(
            user_id=1,
            target_date=target_date,
            procrastination_types=["task_aversion"],
            type_confidence={"task_aversion": 0.65},
            cognitive_distortions=[],
            cbt_technique="graded_exposure",
            response_text="归因结果",
            llm_model="deepseek",
            analysis_kind="daily_attribution",
            source="single_expert",
        )

        # Query with analysis_kind="daily_panel"
        panel_result = await repository.get_by_date(
            1, target_date, analysis_kind="daily_panel"
        )
        assert panel_result is not None
        assert "面板结果" in panel_result["response_text"]
        assert panel_result["analysis_kind"] == "daily_panel"

        # Query with analysis_kind="daily_attribution"
        attr_result = await repository.get_by_date(
            1, target_date, analysis_kind="daily_attribution"
        )
        assert attr_result is not None
        assert "归因结果" in attr_result["response_text"]
        assert attr_result["analysis_kind"] == "daily_attribution"

    async def test_none_kind_returns_first_match(self, engine, session_factory) -> None:
        """When analysis_kind is None, returns first row for that date."""
        async with engine.begin() as connection:
            await connection.run_sync(procrastination_analyses.metadata.create_all)

        repository = SQLAlchemyProcrastinationAnalysisRepository(session_factory)
        target_date = date(2026, 7, 29)

        await repository.upsert(
            user_id=1,
            target_date=target_date,
            procrastination_types=["impulsivity"],
            type_confidence={"impulsivity": 0.85},
            cognitive_distortions=[],
            cbt_technique="stimulus_control",
            response_text="面板结果",
            llm_model="panel",
            analysis_kind="daily_panel",
            source="panel",
        )

        # None returns the default (first match, no kind filter)
        result = await repository.get_by_date(1, target_date)
        assert result is not None
        assert "面板结果" in result["response_text"]

    async def test_kind_not_found_returns_none(self, engine, session_factory) -> None:
        """When the specific kind doesn't exist for that date, returns None."""
        async with engine.begin() as connection:
            await connection.run_sync(procrastination_analyses.metadata.create_all)

        repository = SQLAlchemyProcrastinationAnalysisRepository(session_factory)
        target_date = date(2026, 7, 29)

        await repository.upsert(
            user_id=1,
            target_date=target_date,
            procrastination_types=["impulsivity"],
            type_confidence={"impulsivity": 0.85},
            cognitive_distortions=[],
            cbt_technique="stimulus_control",
            response_text="面板结果",
            llm_model="panel",
            analysis_kind="daily_panel",
            source="panel",
        )

        # Query for a non-existent kind
        result = await repository.get_by_date(
            1, target_date, analysis_kind="daily_attribution"
        )
        assert result is None

    async def test_upsert_idempotent_per_kind(self, engine, session_factory) -> None:
        """Repeated upsert for same kind updates, doesn't create duplicates."""
        async with engine.begin() as connection:
            await connection.run_sync(procrastination_analyses.metadata.create_all)

        repository = SQLAlchemyProcrastinationAnalysisRepository(session_factory)
        target_date = date(2026, 7, 29)

        # First upsert
        await repository.upsert(
            user_id=1,
            target_date=target_date,
            procrastination_types=["impulsivity"],
            type_confidence={"impulsivity": 0.85},
            cognitive_distortions=[],
            cbt_technique="stimulus_control",
            response_text="第一次写入",
            llm_model="panel",
            analysis_kind="daily_panel",
            source="panel",
        )

        # Second upsert — same kind, should update
        await repository.upsert(
            user_id=1,
            target_date=target_date,
            procrastination_types=["task_aversion"],
            type_confidence={"task_aversion": 0.60},
            cognitive_distortions=[],
            cbt_technique="graded_exposure",
            response_text="第二次覆盖",
            llm_model="panel",
            analysis_kind="daily_panel",
            source="panel",
        )

        result = await repository.get_by_date(
            1, target_date, analysis_kind="daily_panel"
        )
        assert result is not None
        # Should have the updated content
        assert "第二次覆盖" in result["response_text"]
        assert result["procrastination_types"] == ["task_aversion"]

    async def test_source_field_persisted_and_read_back(
        self, engine, session_factory
    ) -> None:
        """The source column is correctly stored and returned."""
        async with engine.begin() as connection:
            await connection.run_sync(procrastination_analyses.metadata.create_all)

        repository = SQLAlchemyProcrastinationAnalysisRepository(session_factory)

        for source_val in ("panel", "single_expert", "ollama", "rule_engine"):
            target_date = date(2026, 7, 29)
            await repository.upsert(
                user_id=1,
                target_date=target_date,
                procrastination_types=["impulsivity"],
                type_confidence={"impulsivity": 0.80},
                cognitive_distortions=[],
                cbt_technique="stimulus_control",
                response_text=f"source={source_val}",
                llm_model="test",
                analysis_kind=f"test_{source_val}",
                source=source_val,
            )

            stored = await repository.get_by_date(
                1, target_date, analysis_kind=f"test_{source_val}"
            )
            assert stored is not None
            assert stored["source"] == source_val
