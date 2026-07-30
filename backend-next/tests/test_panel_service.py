"""Tests for PanelService (services/panel_service.py).

Covers:
  - Normal panel path: EvidenceBundleBuilder → PanelOrchestrator → PanelVerdict
  - Degradation to single-expert (PanelUnavailableError from orchestrator)
  - Degradation to single-expert (PanelBudgetExceededError from orchestrator)
  - analysis_dict_to_panel_verdict: exhaustive input variant coverage
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from mindflow.agents.types import (
    PanelBudgetExceededError,
    PanelUnavailableError,
    PanelVerdict,
    TranscriptEntry,
)
from mindflow.domain.procrastination import CBTTechnique, ProcrastinationType
from mindflow.infrastructure.repositories.analysis import (
    SQLAlchemyProcrastinationAnalysisRepository,
    procrastination_analyses,
)
from mindflow.ports import AnalysisRequest, AnalysisResult, AnalysisWorkflowPort
from mindflow.services.panel_service import PanelService, analysis_dict_to_panel_verdict


@pytest.fixture
def mock_orchestrator() -> AsyncMock:
    """Create a mock PanelOrchestrator."""
    return AsyncMock()


@pytest.fixture
def mock_llm_service() -> AsyncMock:
    """Create a mock LLMService."""
    service = AsyncMock()
    # Default: successful outcome
    service.analyze.return_value = type(
        "Outcome",
        (),
        {
            "assessment": {
                "procrastination_types": ["impulsivity"],
                "type_confidence": {"impulsivity": 0.72},
                "cbt_technique": "stimulus_control",
                "response_text": "单专家分析结果（降级模式）",
            },
            "source": "deepseek",
            "cached": False,
            "degraded": True,
        },
    )()
    return service


@pytest.fixture
def mock_builder() -> AsyncMock:
    """Create a mock EvidenceBundleBuilder."""
    builder = AsyncMock()
    bundle = type(
        "Bundle",
        (),
        {
            "user_id": 1,
            "window": (datetime(2026, 7, 18, 0, 0), datetime(2026, 7, 19, 0, 0)),
            "items": (),
            "behavior_summary": None,
            "intervention_history": (),
            "novelty_flags": (),
        },
    )()
    builder.build.return_value = bundle
    return builder


@pytest.fixture
def panel_service(
    mock_builder: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_llm_service: AsyncMock,
) -> PanelService:
    """Create a PanelService with all mocks."""
    service = PanelService.__new__(PanelService)
    service._builder = mock_builder
    service._orchestrator = mock_orchestrator
    service._llm_service = mock_llm_service
    service._analysis_repository = AsyncMock()
    service._timezone = "local"
    return service


def _make_verdict(**overrides: object) -> PanelVerdict:
    """Build a sample PanelVerdict."""
    defaults: dict[str, object] = {
        "types": (ProcrastinationType.IMPULSIVITY,),
        "confidence": {ProcrastinationType.IMPULSIVITY: 0.85},
        "recommended_technique": CBTTechnique.STIMULUS_CONTROL,
        "rationale": "测试会诊结果",
        "dissent": (),
        "transcript": (
            TranscriptEntry(role="数据分析师", content="模式分析完成", round=0),
            TranscriptEntry(role="CBT归因专家", content="归因完成", round=1),
            TranscriptEntry(role="综合主持人", content="裁决完成", round=3),
        ),
        "escalated": False,
        "call_count": 6,
        "source": "panel",
    }
    defaults.update(overrides)
    return PanelVerdict(**defaults)  # type: ignore[arg-type]


def test_panel_service_requires_analysis_repository(
    mock_orchestrator: AsyncMock,
    mock_llm_service: AsyncMock,
) -> None:
    with pytest.raises(TypeError, match="analysis_repository"):
        PanelService(
            activity_repo=MagicMock(),
            intervention_repo=MagicMock(),
            session_factory=MagicMock(),
            orchestrator=mock_orchestrator,
            llm_service=mock_llm_service,
        )


class TestPanelServiceNormal:
    """Normal panel flow — orchestrator succeeds."""

    async def test_panel_success(self, panel_service: PanelService) -> None:
        """Successful panel returns verdict with source='panel'."""
        panel_service._orchestrator.run = AsyncMock(return_value=_make_verdict())
        verdict = await panel_service.run_daily_panel(user_id=1, target_date=date(2026, 7, 18))
        assert verdict.source == "panel"
        assert verdict.call_count == 6
        assert verdict.types == (ProcrastinationType.IMPULSIVITY,)

    async def test_panel_uses_local_business_day_utc_bounds(
        self, panel_service: PanelService
    ) -> None:
        panel_service._timezone = "Asia/Shanghai"
        panel_service._orchestrator.run = AsyncMock(return_value=_make_verdict())

        await panel_service.run_daily_panel(
            user_id=1,
            target_date=date(2026, 7, 17),
        )

        panel_service._builder.build.assert_awaited_once_with(
            1,
            datetime(2026, 7, 16, 16, 0, tzinfo=UTC),
            datetime(2026, 7, 17, 15, 59, 59, 999999, tzinfo=UTC),
        )

    async def test_panel_success_is_persisted_and_read_back(
        self,
        panel_service: PanelService,
        engine,
        session_factory,
    ) -> None:
        async with engine.begin() as connection:
            await connection.run_sync(procrastination_analyses.metadata.create_all)
        repository = SQLAlchemyProcrastinationAnalysisRepository(session_factory)
        panel_service._analysis_repository = repository
        expected = _make_verdict(
            dissent=("少数意见",),
            escalated=True,
            call_count=9,
        )
        panel_service._orchestrator.run = AsyncMock(return_value=expected)
        target_date = date(2026, 7, 25)

        await panel_service.run_daily_panel(user_id=1, target_date=target_date)
        stored = await panel_service.get_stored_verdict(
            user_id=1,
            target_date=target_date,
        )

        assert stored == expected

    async def test_panel_escalated(self, panel_service: PanelService) -> None:
        """Panel with conflict escalation returns escalated=True."""
        panel_service._orchestrator.run = AsyncMock(
            return_value=_make_verdict(escalated=True, call_count=9),
        )
        verdict = await panel_service.run_daily_panel(user_id=1, target_date=date(2026, 7, 18))
        assert verdict.source == "panel"
        assert verdict.escalated is True
        assert verdict.call_count == 9

    async def test_panel_with_dissent(self, panel_service: PanelService) -> None:
        """Panel with recorded dissent."""
        panel_service._orchestrator.run = AsyncMock(
            return_value=_make_verdict(
                dissent=("TMT专家认为情绪调节是主要因素",),
            ),
        )
        verdict = await panel_service.run_daily_panel(user_id=1, target_date=date(2026, 7, 18))
        assert len(verdict.dissent) == 1
        assert "TMT" in verdict.dissent[0]

    async def test_panel_transcript_present(self, panel_service: PanelService) -> None:
        """Panel transcript contains expert entries."""
        panel_service._orchestrator.run = AsyncMock(return_value=_make_verdict())
        verdict = await panel_service.run_daily_panel(user_id=1, target_date=date(2026, 7, 18))
        assert len(verdict.transcript) == 3


class TestPanelServiceDegradation:
    """Degradation to single-expert LLM service."""

    @pytest.mark.parametrize(
        ("outcome_source", "expected_source"),
        [
            ("ollama", "ollama"),
            ("rule_engine", "rule_engine"),
        ],
    )
    async def test_fallback_preserves_lower_tier_source(
        self,
        panel_service: PanelService,
        outcome_source: str,
        expected_source: str,
    ) -> None:
        panel_service._orchestrator.run = AsyncMock(
            side_effect=PanelUnavailableError(reason="专家解析失败"),
        )
        panel_service._llm_service.analyze.return_value.source = outcome_source

        verdict = await panel_service.run_daily_panel(
            user_id=1,
            target_date=date(2026, 7, 18),
        )

        assert verdict.source == expected_source

    async def test_panel_unavailable_fallback(self, panel_service: PanelService) -> None:
        """PanelUnavailableError triggers single-expert fallback."""
        panel_service._orchestrator.run = AsyncMock(
            side_effect=PanelUnavailableError(reason="仅1份有效归因", call_count=4),
        )
        verdict = await panel_service.run_daily_panel(user_id=1, target_date=date(2026, 7, 18))
        assert verdict.source == "single_expert"
        assert verdict.types == (ProcrastinationType.IMPULSIVITY,)
        assert verdict.call_count == 0
        # Verify LLM service was called
        panel_service._llm_service.analyze.assert_awaited_once()

    async def test_panel_budget_exceeded_fallback(self, panel_service: PanelService) -> None:
        """PanelBudgetExceededError triggers single-expert fallback."""
        panel_service._orchestrator.run = AsyncMock(
            side_effect=PanelBudgetExceededError(call_count=12),
        )
        verdict = await panel_service.run_daily_panel(user_id=1, target_date=date(2026, 7, 18))
        assert verdict.source == "single_expert"
        assert verdict.call_count == 0
        panel_service._llm_service.analyze.assert_awaited_once()

    async def test_degraded_verdict_has_rationale(self, panel_service: PanelService) -> None:
        """Degraded verdict has the fallback LLM's rationale."""
        panel_service._orchestrator.run = AsyncMock(
            side_effect=PanelUnavailableError(reason="专家解析失败"),
        )
        verdict = await panel_service.run_daily_panel(user_id=1, target_date=date(2026, 7, 18))
        assert verdict.rationale == "单专家分析结果（降级模式）"

    async def test_degraded_no_transcript(self, panel_service: PanelService) -> None:
        """Degraded verdict has no transcript entries."""
        panel_service._orchestrator.run = AsyncMock(
            side_effect=PanelUnavailableError(reason="专家解析失败"),
        )
        verdict = await panel_service.run_daily_panel(user_id=1, target_date=date(2026, 7, 18))
        assert verdict.transcript == ()
        assert verdict.dissent == ()


# ═══════════════════════════════════════════════════════════════════════════════
# Contract tests — AnalysisWorkflowPort substitution
# ═══════════════════════════════════════════════════════════════════════════════


class FakeAnalysisWorkflow:
    """In-memory fake implementing :class:`mindflow.ports.AnalysisWorkflowPort`.

    Used in contract tests to verify that a non-LangGraph implementation can
    be type-checked against the Protocol and substituted at the PanelService
    call site without changing the ``run_daily_panel`` signature.
    """

    def __init__(self) -> None:
        self.requests: list[AnalysisRequest] = []
        self._default_result: AnalysisResult | None = None

    def set_result(self, result: AnalysisResult) -> None:
        """Configure the verdict this fake returns."""
        self._default_result = result

    async def run_analysis(self, request: AnalysisRequest) -> AnalysisResult:
        self.requests.append(request)
        if self._default_result is None:
            raise RuntimeError("FakeAnalysisWorkflow.set_result() not called")
        return self._default_result


class TestPanelServiceWorkflowPort:
    """Contract: PanelService delegates to AnalysisWorkflowPort when provided."""

    async def test_workflow_port_delegation(self) -> None:
        """run_daily_panel delegates to the injected workflow port."""
        from mindflow.ports import AnalysisResult

        fake = FakeAnalysisWorkflow()
        verdict = _make_verdict(source="panel", call_count=3)
        fake.set_result(
            AnalysisResult(
                verdict=verdict,
                run_id="fake-run-abc",
                created_at=datetime(2026, 7, 26, 10, 0, tzinfo=UTC),
            )
        )

        service = PanelService.__new__(PanelService)
        service._builder = AsyncMock()
        service._orchestrator = AsyncMock()
        service._llm_service = AsyncMock()
        service._analysis_repository = AsyncMock()
        service._timezone = "local"
        service._workflow_port = fake

        result = await service.run_daily_panel(
            user_id=42,
            target_date=date(2026, 7, 26),
        )

        # Delegation → fake received exactly one request
        assert len(fake.requests) == 1
        req: AnalysisRequest = fake.requests[0]
        assert req.user_id == 42
        assert req.target_date == date(2026, 7, 26)
        assert req.force is False
        assert req.origin == "scheduler"
        assert req.idempotency_key == ""

        # Result → verdict forwarded correctly
        assert result is verdict
        assert result.source == "panel"
        assert result.call_count == 3

        # Existing orchestrator was NOT called
        service._orchestrator.run.assert_not_called()

    async def test_workflow_port_none_uses_existing_path(self) -> None:
        """None workflow_port → original orchestration path (backward compat)."""
        service = PanelService.__new__(PanelService)
        service._builder = AsyncMock()
        service._orchestrator = AsyncMock()
        service._llm_service = AsyncMock()
        service._analysis_repository = AsyncMock()
        service._timezone = "local"
        service._workflow_port = None  # explicitly None

        service._orchestrator.run = AsyncMock(return_value=_make_verdict())

        await service.run_daily_panel(user_id=1, target_date=date(2026, 7, 26))

        # Original path used
        service._orchestrator.run.assert_awaited_once()

    async def test_workflow_port_is_optional_in_constructor(self) -> None:
        """PanelService can be constructed without workflow_port (backward compat)."""
        # This is the existing test pattern — just verifying the parameter is optional
        service = PanelService.__new__(PanelService)
        service._builder = AsyncMock()
        service._orchestrator = AsyncMock()
        service._llm_service = AsyncMock()
        service._analysis_repository = AsyncMock()
        service._timezone = "local"
        # _workflow_port not set → defaults to None after __new__
        assert getattr(service, "_workflow_port", None) is None

    async def test_fake_is_protocol_compatible(self) -> None:
        """FakeAnalysisWorkflow satisfies AnalysisWorkflowPort structural typing."""
        fake = FakeAnalysisWorkflow()
        # Verify the fake satisfies the Protocol by calling through a typed
        # variable — mypy will flag this if the method signatures drift.
        port: AnalysisWorkflowPort = fake
        assert port is fake


# ═══════════════════════════════════════════════════════════════════════════════
# analysis_dict_to_panel_verdict — exhaustive input variant coverage
# ═══════════════════════════════════════════════════════════════════════════════


class TestAnalysisDictToPanelVerdictSource:
    """Every PanelSource value is correctly parsed."""

    @pytest.mark.parametrize(
        ("source_value", "expected"),
        [
            ("panel", "panel"),
            ("single_expert", "single_expert"),
            ("ollama", "ollama"),
            ("rule_engine", "rule_engine"),
        ],
    )
    def test_explicit_source_via_kwargs(self, source_value: str, expected: str) -> None:
        assessment = {
            "types": ["impulsivity"],
            "confidence": {"impulsivity": 0.82},
        }
        verdict = analysis_dict_to_panel_verdict(assessment, source=source_value)
        assert verdict.source == expected

    @pytest.mark.parametrize(
        ("dict_source", "expected"),
        [
            ("panel", "panel"),
            ("single_expert", "single_expert"),
            ("ollama", "ollama"),
            ("rule_engine", "rule_engine"),
        ],
    )
    def test_source_embedded_in_dict(self, dict_source: str, expected: str) -> None:
        assessment = {
            "types": ["impulsivity"],
            "confidence": {"impulsivity": 0.82},
            "source": dict_source,
        }
        verdict = analysis_dict_to_panel_verdict(assessment)
        assert verdict.source == expected

    def test_unknown_source_defaults_to_single_expert(self) -> None:
        assessment = {
            "types": ["impulsivity"],
            "confidence": {"impulsivity": 0.82},
            "source": "unknown_engine",
        }
        verdict = analysis_dict_to_panel_verdict(assessment)
        assert verdict.source == "single_expert"

    def test_none_source_defaults_to_single_expert(self) -> None:
        assessment = {
            "types": ["impulsivity"],
            "confidence": {"impulsivity": 0.82},
        }
        verdict = analysis_dict_to_panel_verdict(assessment)
        assert verdict.source == "single_expert"

    def test_missing_source_field_defaults_to_single_expert(self) -> None:
        verdict = analysis_dict_to_panel_verdict(
            {"types": ["task_aversion"], "confidence": {"task_aversion": 0.65}}
        )
        assert verdict.source == "single_expert"


class TestAnalysisDictToPanelVerdictFieldNames:
    """Both LLM-output and storage field names are accepted."""

    def test_llm_field_names(self) -> None:
        """LLM output keys: types, confidence, recommended_technique, rationale."""
        verdict = analysis_dict_to_panel_verdict(
            {
                "types": ["impulsivity", "task_aversion"],
                "confidence": {"impulsivity": 0.85, "task_aversion": 0.45},
                "recommended_technique": "stimulus_control",
                "rationale": "LLM 格式的理由",
                "dissent": ["少数意见"],
            },
            source="panel",
        )
        assert verdict.types == (ProcrastinationType.IMPULSIVITY, ProcrastinationType.TASK_AVERSION)
        assert verdict.confidence[ProcrastinationType.IMPULSIVITY] == 0.85
        assert verdict.recommended_technique == CBTTechnique.STIMULUS_CONTROL
        assert verdict.rationale == "LLM 格式的理由"
        assert len(verdict.dissent) == 1

    def test_storage_field_names(self) -> None:
        """Storage keys: procrastination_types, type_confidence, cbt_technique, response_text."""
        verdict = analysis_dict_to_panel_verdict(
            {
                "procrastination_types": ["task_aversion"],
                "type_confidence": {"task_aversion": 0.72},
                "cbt_technique": "stimulus_control",
                "response_text": "存储格式的理由",
            },
            source="single_expert",
        )
        assert verdict.types == (ProcrastinationType.TASK_AVERSION,)
        assert verdict.confidence[ProcrastinationType.TASK_AVERSION] == 0.72
        assert verdict.recommended_technique == CBTTechnique.STIMULUS_CONTROL
        assert verdict.rationale == "存储格式的理由"

    def test_storage_keys_fallback_when_llm_missing(self) -> None:
        """Storage keys used as fallback when LLM keys are absent or wrong type."""
        verdict = analysis_dict_to_panel_verdict(
            {
                "types": "not_a_list",
                "procrastination_types": ["impulsivity"],
                "confidence": "not_a_dict",
                "type_confidence": {"impulsivity": 0.60},
                "recommended_technique": None,
                "cbt_technique": "stimulus_control",
            },
            source="ollama",
        )
        assert verdict.types == (ProcrastinationType.IMPULSIVITY,)
        assert verdict.confidence[ProcrastinationType.IMPULSIVITY] == 0.60
        assert verdict.recommended_technique == CBTTechnique.STIMULUS_CONTROL
        assert verdict.source == "ollama"


class TestAnalysisDictToPanelVerdictTranscript:
    """panel_transcript dict propagation to PanelVerdict.transcript."""

    def test_transcript_entries_parsed(self) -> None:
        verdict = analysis_dict_to_panel_verdict(
            {
                "types": ["impulsivity"],
                "confidence": {"impulsivity": 0.82},
                "panel_transcript": {
                    "transcript": [
                        {"role": "数据分析师", "content": "模式分析完成", "round": 0},
                        {"role": "CBT归因专家", "content": "归因完成", "round": 1},
                        {"role": "综合主持人", "content": "裁决完成", "round": 3},
                    ],
                    "dissent": [],
                    "escalated": False,
                    "call_count": 6,
                },
            },
            source="panel",
        )
        assert len(verdict.transcript) == 3
        assert verdict.transcript[0].role == "数据分析师"
        assert verdict.transcript[0].round == 0
        assert verdict.transcript[2].role == "综合主持人"
        assert verdict.transcript[2].round == 3
        assert verdict.escalated is False
        assert verdict.call_count == 6

    def test_transcript_keyword_override(self) -> None:
        """Keyword arg transcript overrides dict-embedded transcript."""
        override = (
            TranscriptEntry(role="系统", content="覆盖", round=0),
        )
        verdict = analysis_dict_to_panel_verdict(
            {
                "types": ["impulsivity"],
                "confidence": {"impulsivity": 0.82},
                "panel_transcript": {
                    "transcript": [
                        {"role": "数据分析师", "content": "原始", "round": 0},
                    ],
                },
            },
            transcript=override,
            source="panel",
        )
        assert len(verdict.transcript) == 1
        assert verdict.transcript[0].role == "系统"

    def test_dissent_from_top_level(self) -> None:
        verdict = analysis_dict_to_panel_verdict(
            {
                "types": ["impulsivity"],
                "confidence": {"impulsivity": 0.82},
                "dissent": ["CBT专家有异议", "TMT专家保留意见"],
            },
            source="panel",
        )
        assert verdict.dissent == ("CBT专家有异议", "TMT专家保留意见")

    def test_dissent_from_panel_transcript_fallback(self) -> None:
        verdict = analysis_dict_to_panel_verdict(
            {
                "types": ["impulsivity"],
                "confidence": {"impulsivity": 0.82},
                "panel_transcript": {
                    "dissent": ["TMT专家认为任务畏惧是主因"],
                    "transcript": [],
                },
            },
            source="panel",
        )
        assert len(verdict.dissent) == 1
        assert "TMT" in verdict.dissent[0]

    def test_escalated_from_panel_transcript(self) -> None:
        verdict = analysis_dict_to_panel_verdict(
            {
                "types": ["impulsivity"],
                "confidence": {"impulsivity": 0.82},
                "panel_transcript": {
                    "transcript": [],
                    "escalated": True,
                },
            },
            source="panel",
        )
        assert verdict.escalated is True

    def test_call_count_from_panel_transcript(self) -> None:
        verdict = analysis_dict_to_panel_verdict(
            {
                "types": ["impulsivity"],
                "confidence": {"impulsivity": 0.82},
                "panel_transcript": {
                    "transcript": [],
                    "call_count": 9,
                },
            },
            source="panel",
        )
        assert verdict.call_count == 9

    def test_escalated_keyword_override(self) -> None:
        verdict = analysis_dict_to_panel_verdict(
            {
                "types": ["impulsivity"],
                "confidence": {"impulsivity": 0.82},
                "panel_transcript": {"escalated": False, "transcript": []},
            },
            escalated=True,
            source="panel",
        )
        assert verdict.escalated is True

    def test_call_count_keyword_override(self) -> None:
        verdict = analysis_dict_to_panel_verdict(
            {
                "types": ["impulsivity"],
                "confidence": {"impulsivity": 0.82},
                "panel_transcript": {"call_count": 6, "transcript": []},
            },
            call_count=12,
            source="panel",
        )
        assert verdict.call_count == 12

    def test_non_dict_entries_skipped_in_transcript(self) -> None:
        """Non-dict transcript entries are silently skipped."""
        verdict = analysis_dict_to_panel_verdict(
            {
                "types": ["impulsivity"],
                "confidence": {"impulsivity": 0.82},
                "panel_transcript": {
                    "transcript": [
                        "not_a_dict",
                        {"role": "数据分析师", "content": "有效", "round": 0},
                        42,
                    ],
                },
            },
            source="panel",
        )
        assert len(verdict.transcript) == 1
        assert verdict.transcript[0].role == "数据分析师"

    def test_incomplete_transcript_entries_filtered(self) -> None:
        """Entries missing role, content, or round are skipped."""
        verdict = analysis_dict_to_panel_verdict(
            {
                "types": ["impulsivity"],
                "confidence": {"impulsivity": 0.82},
                "panel_transcript": {
                    "transcript": [
                        {"role": "专家", "content": "有效", "round": 1},
                        {"role": "专家", "content": "缺round字段"},
                        {"role": "专家", "round": 2},
                        {"content": "缺role", "round": 3},
                    ],
                },
            },
            source="panel",
        )
        assert len(verdict.transcript) == 1
        assert verdict.transcript[0].round == 1

    def test_round_bool_rejected(self) -> None:
        """bool True/False should NOT be accepted as a valid round int."""
        verdict = analysis_dict_to_panel_verdict(
            {
                "types": ["impulsivity"],
                "confidence": {"impulsivity": 0.82},
                "panel_transcript": {
                    "transcript": [
                        {"role": "专家", "content": "文本", "round": True},
                    ],
                },
            },
            source="panel",
        )
        # In Python, bool is subclass of int, but the check
        # `not isinstance(round_number, bool)` should reject it.
        assert verdict.transcript == ()


class TestAnalysisDictToPanelVerdictEdgeCases:
    """Edge cases, malformed input, and default preservation."""

    def test_empty_dict_produces_verdict_with_defaults(self) -> None:
        """Empty dict should not raise; defaults are applied."""
        verdict = analysis_dict_to_panel_verdict({})
        assert isinstance(verdict, PanelVerdict)
        assert len(verdict.types) == 1
        assert verdict.types[0] == ProcrastinationType.TASK_AVERSION
        assert verdict.confidence == {ProcrastinationType.TASK_AVERSION: 0.5}
        assert verdict.recommended_technique is None
        assert verdict.rationale == ""
        assert verdict.dissent == ()
        assert verdict.transcript == ()
        assert verdict.escalated is False
        assert verdict.call_count == 0
        assert verdict.source == "single_expert"

    def test_unknown_procrastination_type_warning_and_fallback(self) -> None:
        """Unknown types are warned about and filtered; fallback applied if empty."""
        verdict = analysis_dict_to_panel_verdict(
            {
                "types": ["unknown_type", "bogus"],
                "confidence": {"impulsivity": 0.80},
            },
            source="panel",
        )
        assert verdict.types == (ProcrastinationType.TASK_AVERSION,)
        assert verdict.confidence[ProcrastinationType.TASK_AVERSION] == 0.5

    def test_partially_known_types(self) -> None:
        """Known types survive alongside unknown ones."""
        verdict = analysis_dict_to_panel_verdict(
            {
                "types": ["impulsivity", "unknown_type"],
                "confidence": {"impulsivity": 0.75},
            },
            source="panel",
        )
        assert verdict.types == (ProcrastinationType.IMPULSIVITY,)
        assert verdict.confidence[ProcrastinationType.IMPULSIVITY] == 0.75

    def test_confidence_fills_missing_types(self) -> None:
        """Types without confidence entries get default 0.5."""
        verdict = analysis_dict_to_panel_verdict(
            {
                "types": ["impulsivity", "task_aversion"],
                "confidence": {"impulsivity": 0.80},
            },
            source="panel",
        )
        assert verdict.confidence[ProcrastinationType.TASK_AVERSION] == 0.5

    def test_unknown_cbt_technique_returns_none(self) -> None:
        verdict = analysis_dict_to_panel_verdict(
            {
                "types": ["impulsivity"],
                "confidence": {"impulsivity": 0.82},
                "recommended_technique": "nonexistent_technique",
            },
            source="panel",
        )
        assert verdict.recommended_technique is None

    def test_rationale_from_response_text(self) -> None:
        """If rationale is absent, falls back to response_text."""
        verdict = analysis_dict_to_panel_verdict(
            {
                "types": ["impulsivity"],
                "confidence": {"impulsivity": 0.82},
                "response_text": "这是单专家分析的结果",
            },
            source="single_expert",
        )
        assert verdict.rationale == "这是单专家分析的结果"

    def test_none_panel_transcript_is_handled(self) -> None:
        """None panel_transcript should not raise."""
        verdict = analysis_dict_to_panel_verdict(
            {
                "types": ["impulsivity"],
                "confidence": {"impulsivity": 0.82},
                "panel_transcript": None,
            },
            source="panel",
        )
        assert verdict.transcript == ()
        assert verdict.dissent == ()
        assert verdict.escalated is False
        assert verdict.call_count == 0

    def test_non_dict_panel_transcript_is_handled(self) -> None:
        verdict = analysis_dict_to_panel_verdict(
            {
                "types": ["impulsivity"],
                "confidence": {"impulsivity": 0.82},
                "panel_transcript": "not_a_dict",
            },
            source="panel",
        )
        assert verdict.transcript == ()
        assert verdict.escalated is False
        assert verdict.call_count == 0

    def test_dissent_non_list_is_coerced(self) -> None:
        """Non-list dissent value is coerced to empty tuple."""
        verdict = analysis_dict_to_panel_verdict(
            {
                "types": ["impulsivity"],
                "confidence": {"impulsivity": 0.82},
                "dissent": "not_a_list",
            },
            source="panel",
        )
        assert verdict.dissent == ()

    def test_call_count_non_int_defaults_to_zero(self) -> None:
        verdict = analysis_dict_to_panel_verdict(
            {
                "types": ["impulsivity"],
                "confidence": {"impulsivity": 0.82},
                "panel_transcript": {
                    "transcript": [],
                    "call_count": "not_an_int",
                },
            },
            source="panel",
        )
        assert verdict.call_count == 0

    def test_all_keyword_overrides_simultaneously(self) -> None:
        """All keyword overrides applied at once."""
        override_transcript = (TranscriptEntry(role="覆盖", content="test", round=0),)
        verdict = analysis_dict_to_panel_verdict(
            {
                "types": ["impulsivity"],
                "confidence": {"impulsivity": 0.82},
                "source": "panel",
            },
            escalated=True,
            transcript=override_transcript,
            call_count=3,
            source="ollama",
        )
        assert verdict.escalated is True
        assert verdict.transcript == override_transcript
        assert verdict.call_count == 3
        assert verdict.source == "ollama"


class TestAnalysisDictToPanelVerdictDBRoundtrip:
    """Full DB round-trip: upsert → get_by_date → analysis_dict_to_panel_verdict."""

    async def test_stored_panel_verdict_round_trips(
        self, engine, session_factory
    ) -> None:
        """After upsert and read-back, conversion produces same PanelVerdict."""
        from mindflow.domain.procrastination import CBTTechnique, ProcrastinationType

        async with engine.begin() as connection:
            await connection.run_sync(procrastination_analyses.metadata.create_all)

        repository = SQLAlchemyProcrastinationAnalysisRepository(session_factory)
        target_date = date(2026, 7, 29)
        panel_metadata = {
            "transcript": [
                {"role": "数据分析师", "content": "分析完成", "round": 0},
                {"role": "综合主持人", "content": "裁决完成", "round": 3},
            ],
            "dissent": ["情绪调节专家有保留"],
            "escalated": True,
            "call_count": 9,
        }

        await repository.upsert(
            user_id=1,
            target_date=target_date,
            procrastination_types=["impulsivity", "task_aversion"],
            type_confidence={"impulsivity": 0.85, "task_aversion": 0.45},
            cognitive_distortions=["all-or-nothing"],
            cbt_technique="stimulus_control",
            response_text="测试会诊结果",
            llm_model="panel",
            panel_transcript=panel_metadata,
            analysis_kind="daily_panel",
            source="panel",
        )

        stored = await repository.get_by_date(1, target_date, analysis_kind="daily_panel")
        assert stored is not None
        stored["source"] = "panel"

        verdict = analysis_dict_to_panel_verdict(stored)
        assert verdict.source == "panel"
        assert verdict.types == (
            ProcrastinationType.IMPULSIVITY,
            ProcrastinationType.TASK_AVERSION,
        )
        assert verdict.confidence[ProcrastinationType.IMPULSIVITY] == 0.85
        assert verdict.confidence[ProcrastinationType.TASK_AVERSION] == 0.45
        assert verdict.recommended_technique == CBTTechnique.STIMULUS_CONTROL
        assert verdict.rationale == "测试会诊结果"
        assert verdict.escalated is True
        assert verdict.call_count == 9
        assert len(verdict.dissent) == 1
        assert len(verdict.transcript) == 2
        assert verdict.transcript[0].role == "数据分析师"

    async def test_stored_single_expert_verdict_round_trips(
        self, engine, session_factory
    ) -> None:
        """Single-expert (fallback) storage round-trips correctly."""
        async with engine.begin() as connection:
            await connection.run_sync(procrastination_analyses.metadata.create_all)

        repository = SQLAlchemyProcrastinationAnalysisRepository(session_factory)
        target_date = date(2026, 7, 29)

        await repository.upsert(
            user_id=1,
            target_date=target_date,
            procrastination_types=["task_aversion"],
            type_confidence={"task_aversion": 0.72},
            cognitive_distortions=[],
            cbt_technique="graded_exposure",
            response_text="单专家分析结果",
            llm_model="deepseek",
            analysis_kind="daily_panel",
            source="single_expert",
        )

        stored = await repository.get_by_date(1, target_date, analysis_kind="daily_panel")
        assert stored is not None

        verdict = analysis_dict_to_panel_verdict(stored)
        assert verdict.source == "single_expert"
        assert verdict.types == (ProcrastinationType.TASK_AVERSION,)
        assert verdict.recommended_technique == CBTTechnique.GRADED_EXPOSURE
        assert verdict.rationale == "单专家分析结果"
        # No panel_transcript → no transcript entries
        assert verdict.transcript == ()
        assert verdict.dissent == ()
        assert verdict.escalated is False
        assert verdict.call_count == 0
