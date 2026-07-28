"""Procrastination analysis repository persistence tests."""

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
