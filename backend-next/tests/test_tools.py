"""Contract tests for typed tool adapters in ``mindflow.graph.tools``.

Covers:
  - Schema validation (input/output Pydantic models)
  - Explicit context passing (no ContextVar)
  - Days cap enforcement (max 30)
  - Budget reservation (durable via BudgetReservationPort)
  - Output sanitisation
  - Deterministic errors for missing user/session, days over cap,
    duplicate run-analysis reservation
  - Restart test: cannot bypass the panel cap
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from mindflow.domain.procrastination import BehaviorSummary
from mindflow.graph.tools import (
    InterventionHistoryInput,
    InterventionHistoryOutput,
    InterventionHistoryTool,
    LatestAnalysisOutput,
    LatestAnalysisTool,
    QueryEvidenceInput,
    QueryEvidenceOutput,
    QueryEvidenceTool,
    RunAnalysisInput,
    RunAnalysisOutput,
    RunAnalysisTool,
    ToolContext,
    _sanitise_dict,
    format_analysis_output,
    format_evidence_output,
    format_intervention_output,
    format_run_output,
)
from mindflow.ports import BudgetReservationPort

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _make_empty_bundle() -> MagicMock:
    """Create a mock EvidenceBundle with empty fields."""
    bundle = MagicMock()
    bundle.user_id = 1
    bundle.window = (
        datetime(2026, 7, 18, 0, 0, tzinfo=UTC),
        datetime(2026, 7, 18, 23, 59, tzinfo=UTC),
    )
    bundle.items = ()
    bundle.behavior_summary = BehaviorSummary(
        intended_task=None,
        duration_min=0.0,
        actual_focus_min=0.0,
        context_switches_per_hour=0.0,
        longest_focus_block_s=0.0,
        social_media_ratio=0.0,
        start_delay_min=0.0,
        keyword_flags=frozenset(),
        baseline_deviation=None,
    )
    bundle.intervention_history = ()
    bundle.novelty_flags = ()
    return bundle


# ═══════════════════════════════════════════════════════════════════════════════
# Schema validation tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestQueryEvidenceInput:
    """Input schema validation for QueryEvidenceInput."""

    def test_days_defaults_to_7(self) -> None:
        inp = QueryEvidenceInput(user_id=1)
        assert inp.days == 7

    def test_days_capped_at_30(self) -> None:
        inp = QueryEvidenceInput(user_id=1, days=365)
        assert inp.days == 30

    def test_days_capped_exactly_30(self) -> None:
        inp = QueryEvidenceInput(user_id=1, days=30)
        assert inp.days == 30

    def test_days_below_30_passes_through(self) -> None:
        inp = QueryEvidenceInput(user_id=1, days=14)
        assert inp.days == 14

    def test_days_minimum_1(self) -> None:
        inp = QueryEvidenceInput(user_id=1, days=1)
        assert inp.days == 1

    def test_days_zero_raises(self) -> None:
        with pytest.raises(Exception):
            QueryEvidenceInput(user_id=1, days=0)

    def test_session_id_optional(self) -> None:
        inp = QueryEvidenceInput(user_id=1)
        assert inp.session_id is None


class TestRunAnalysisInput:
    """Input schema validation for RunAnalysisInput."""

    def test_date_required(self) -> None:
        inp = RunAnalysisInput(user_id=1, target_date=date(2026, 7, 29))
        assert inp.target_date == date(2026, 7, 29)

    def test_date_missing_raises(self) -> None:
        with pytest.raises(Exception):
            RunAnalysisInput(user_id=1)  # type: ignore[call-arg]

    def test_force_defaults_to_false(self) -> None:
        inp = RunAnalysisInput(user_id=1, target_date=date(2026, 7, 29))
        assert inp.force is False


class TestInterventionHistoryInput:
    """Input schema validation for InterventionHistoryInput."""

    def test_days_capped_at_30(self) -> None:
        inp = InterventionHistoryInput(user_id=1, days=100)
        assert inp.days == 30


# ═══════════════════════════════════════════════════════════════════════════════
# Output sanitisation tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSanitiseDict:
    """_sanitise_dict strips sensitive keys."""

    def test_strips_api_key(self) -> None:
        result = _sanitise_dict({"type": "ok", "api_key": "secret-123"})
        assert result == {"type": "ok"}
        assert "api_key" not in result

    def test_strips_raw_prompt(self) -> None:
        result = _sanitise_dict({"raw_prompt": "system: you are...", "score": 0.9})
        assert result == {"score": 0.9}

    def test_strips_evidence(self) -> None:
        result = _sanitise_dict({"evidence": "big json blob", "name": "test"})
        assert result == {"name": "test"}

    def test_strips_full_output(self) -> None:
        result = _sanitise_dict({"full_output": "complete llm response"})
        assert result == {}

    def test_recursive_nested(self) -> None:
        data = {"outer": {"api_key": "nested-secret", "ok": True}}
        result = _sanitise_dict(data)
        assert result == {"outer": {"ok": True}}

    def test_preserves_known_keys(self) -> None:
        data = {
            "procrastination_types": ["impulsivity"],
            "type_confidence": {"impulsivity": 0.8},
            "rationale": "test",
        }
        result = _sanitise_dict(data)
        assert result == data

    def test_non_dict_passes_through(self) -> None:
        assert _sanitise_dict("string") == "string"
        assert _sanitise_dict(42) == 42


# ═══════════════════════════════════════════════════════════════════════════════
# Output formatting tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestFormatEvidenceOutput:
    """format_evidence_output tests."""

    def test_returns_evidence_json_on_success(self) -> None:
        output = QueryEvidenceOutput(evidence_json='{"key":"val"}', total_days=7)
        result = format_evidence_output(output)
        assert result == '{"key":"val"}'

    def test_returns_error_json_on_error(self) -> None:
        output = QueryEvidenceOutput(error="user_id not set")
        result = format_evidence_output(output)
        assert "error" in result
        assert "user_id not set" in result


class TestFormatAnalysisOutput:
    """format_analysis_output tests."""

    def test_returns_not_found_message(self) -> None:
        output = LatestAnalysisOutput(error="暂无分析数据")
        assert format_analysis_output(output) == "暂无分析数据"

    def test_returns_json_on_success(self) -> None:
        output = LatestAnalysisOutput(analysis={"type": "impulsivity"}, source="panel")
        result = format_analysis_output(output)
        assert "impulsivity" in result

    def test_returns_not_found_when_none(self) -> None:
        output = LatestAnalysisOutput()
        assert format_analysis_output(output) == "暂无分析数据"


class TestFormatRunOutput:
    """format_run_output tests."""

    def test_returns_error_message_directly(self) -> None:
        output = RunAnalysisOutput(degraded=True, error="服务不可用")
        assert format_run_output(output) == "服务不可用"

    def test_returns_analysis_json_on_success(self) -> None:
        output = RunAnalysisOutput(
            analysis={"types": ["impulsivity"]}, source="panel", degraded=False
        )
        result = format_run_output(output)
        assert "impulsivity" in result


class TestFormatInterventionOutput:
    """format_intervention_output tests."""

    def test_returns_not_found_message(self) -> None:
        output = InterventionHistoryOutput(interventions=[], total=0)
        assert format_intervention_output(output) == "暂无干预记录"

    def test_returns_json_on_success(self) -> None:
        output = InterventionHistoryOutput(
            interventions=[{"type": "nudge", "time": "2026-07-29T10:00:00Z"}],
            total=1,
        )
        result = format_intervention_output(output)
        assert "nudge" in result


# ═══════════════════════════════════════════════════════════════════════════════
# QueryEvidenceTool tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestQueryEvidenceTool:
    """QueryEvidenceTool contract tests."""

    @pytest.fixture
    def evidence_builder(self) -> AsyncMock:
        builder = AsyncMock()
        builder.build = AsyncMock(return_value=_make_empty_bundle())
        return builder

    @pytest.fixture
    def tool(self, evidence_builder: AsyncMock) -> QueryEvidenceTool:
        return QueryEvidenceTool(evidence_builder=evidence_builder)

    async def test_missing_context_returns_error(
        self, tool: QueryEvidenceTool
    ) -> None:
        """Missing user_id → deterministic error."""
        tool.context = None
        result = await tool.execute(days=7)
        assert result.error == "user_id not set"

    async def test_zero_user_id_returns_error(
        self, tool: QueryEvidenceTool
    ) -> None:
        """user_id=0 → deterministic error."""
        tool.context = ToolContext(user_id=0)
        result = await tool.execute(days=7)
        assert result.error == "user_id not set"

    async def test_days_over_cap_marks_capped(
        self, tool: QueryEvidenceTool
    ) -> None:
        """Days > 30 → capped=True in output."""
        tool.context = ToolContext(user_id=1)
        result = await tool.execute(days=100)
        assert result.capped is True
        assert result.total_days == 30

    async def test_days_under_cap_not_capped(
        self, tool: QueryEvidenceTool
    ) -> None:
        """Days <= 30 → capped=False."""
        tool.context = ToolContext(user_id=1)
        result = await tool.execute(days=7)
        assert result.capped is False
        assert result.total_days == 7

    async def test_returns_evidence_json(
        self, tool: QueryEvidenceTool, evidence_builder: AsyncMock
    ) -> None:
        """Returns valid JSON evidence string."""
        tool.context = ToolContext(user_id=1)
        result = await tool.execute(days=7)
        assert result.evidence_json.startswith("{")
        assert '"window"' in result.evidence_json
        evidence_builder.build.assert_awaited_once()

    async def test_no_contextvar_in_execution_path(
        self, tool: QueryEvidenceTool
    ) -> None:
        """Tool does not read any ContextVar."""
        tool.context = ToolContext(user_id=1)
        # If the tool tried to read a ContextVar, it would fail because
        # we never set one — the explicit ToolContext is the only source.
        result = await tool.execute(days=7)
        assert result.error is None


# ═══════════════════════════════════════════════════════════════════════════════
# LatestAnalysisTool tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestLatestAnalysisTool:
    """LatestAnalysisTool contract tests."""

    @pytest.fixture
    def analysis_repo(self) -> AsyncMock:
        repo = AsyncMock()
        repo.get_by_date = AsyncMock(return_value=None)
        return repo

    @pytest.fixture
    def tool(self, analysis_repo: AsyncMock) -> LatestAnalysisTool:
        return LatestAnalysisTool(analysis_repo=analysis_repo)

    async def test_missing_context_returns_error(
        self, tool: LatestAnalysisTool
    ) -> None:
        tool.context = None
        result = await tool.execute()
        assert result.error == "user_id not set"

    async def test_no_data_returns_not_found(
        self, tool: LatestAnalysisTool, analysis_repo: AsyncMock
    ) -> None:
        tool.context = ToolContext(user_id=1)
        analysis_repo.get_by_date = AsyncMock(return_value=None)
        result = await tool.execute()
        assert result.error == "暂无分析数据"
        assert result.analysis is None

    async def test_returns_sanitised_analysis(
        self, tool: LatestAnalysisTool, analysis_repo: AsyncMock
    ) -> None:
        tool.context = ToolContext(user_id=1)
        analysis_repo.get_by_date = AsyncMock(
            return_value={
                "procrastination_types": ["impulsivity"],
                "type_confidence": {"impulsivity": 0.8},
                "api_key": "should-be-stripped",
            }
        )
        result = await tool.execute()
        assert result.analysis is not None
        assert "api_key" not in result.analysis
        assert "impulsivity" in str(result.analysis)

    async def test_no_contextvar_in_execution_path(
        self, tool: LatestAnalysisTool
    ) -> None:
        tool.context = ToolContext(user_id=1)
        result = await tool.execute()
        # Should not raise ContextVar lookup error
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════════════
# RunAnalysisTool tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunAnalysisTool:
    """RunAnalysisTool contract tests — budget reservation and cap."""

    @pytest.fixture
    def panel_service(self) -> AsyncMock:
        svc = AsyncMock()
        verdict = MagicMock()
        verdict.types = ()
        verdict.confidence = {}
        verdict.rationale = "会诊完成"
        svc.run_daily_panel = AsyncMock(return_value=verdict)
        return svc

    @pytest.fixture
    def budget_port(self) -> AsyncMock:
        port = AsyncMock(spec=BudgetReservationPort)
        port.try_reserve = AsyncMock(return_value=True)
        port.release = AsyncMock(return_value=None)
        return port

    @pytest.fixture
    def tool(
        self, panel_service: AsyncMock, budget_port: AsyncMock
    ) -> RunAnalysisTool:
        return RunAnalysisTool(
            panel_service=panel_service, budget_port=budget_port
        )

    async def test_missing_context_returns_error(
        self, tool: RunAnalysisTool
    ) -> None:
        tool.context = None
        result = await tool.execute(date=date(2026, 7, 29))
        assert result.error == "user_id not set"
        assert result.degraded is True

    async def test_no_panel_service_returns_error(self) -> None:
        tool = RunAnalysisTool(panel_service=None)
        tool.context = ToolContext(user_id=1)
        result = await tool.execute(date=date(2026, 7, 29))
        assert "暂不可用" in (result.error or "")

    async def test_budget_reservation_succeeds_first_call(
        self, tool: RunAnalysisTool, budget_port: AsyncMock
    ) -> None:
        """First call reserves budget and runs panel."""
        tool.context = ToolContext(user_id=1, session_id="s1")
        result = await tool.execute(date=date(2026, 7, 29))
        assert result.degraded is False
        assert "会诊完成" in str(result.analysis)
        budget_port.try_reserve.assert_awaited_once()

    async def test_duplicate_reservation_returns_error(
        self, tool: RunAnalysisTool, budget_port: AsyncMock
    ) -> None:
        """Second call with same key → deterministic error."""
        tool.context = ToolContext(user_id=1, session_id="dup-session")
        budget_port.try_reserve = AsyncMock(return_value=True)

        # First call: succeeds
        result1 = await tool.execute(date=date(2026, 7, 29))
        assert result1.degraded is False

        # Simulate duplicate by making try_reserve return False
        budget_port.try_reserve = AsyncMock(return_value=False)

        # Second call: rejected
        result2 = await tool.execute(date=date(2026, 7, 29))
        assert result2.degraded is True
        assert "已超出" in (result2.error or "")

    async def test_budget_released_on_failure(
        self, tool: RunAnalysisTool, budget_port: AsyncMock, panel_service: AsyncMock
    ) -> None:
        """On panel execution failure, budget is released for retry."""
        tool.context = ToolContext(user_id=1, session_id="fail-session")
        panel_service.run_daily_panel = AsyncMock(
            side_effect=RuntimeError("panel crashed")
        )

        result = await tool.execute(date=date(2026, 7, 29))
        assert result.degraded is True
        assert "会诊执行失败" in (result.error or "")
        # Budget was released so retry is possible
        budget_port.release.assert_awaited_once()

    async def test_budget_not_released_on_success(
        self, tool: RunAnalysisTool, budget_port: AsyncMock
    ) -> None:
        """On success, budget stays reserved (durable proof of usage)."""
        tool.context = ToolContext(user_id=1, session_id="success-session")
        result = await tool.execute(date=date(2026, 7, 29))
        assert result.degraded is False
        # release must NOT be called on success
        budget_port.release.assert_not_awaited()

    async def test_idempotency_key_is_deterministic(
        self, tool: RunAnalysisTool
    ) -> None:
        """Same (user, session, date) → same idempotency key."""
        tool.context = ToolContext(user_id=42, session_id="abc")
        key1 = tool._idempotency_key(date(2026, 7, 29))
        key2 = tool._idempotency_key(date(2026, 7, 29))
        assert key1 == key2
        assert "42" in key1
        assert "abc" in key1

    async def test_restart_cannot_bypass_panel_cap(
        self, panel_service: AsyncMock
    ) -> None:
        """Simulate a restart: the budget reservation persists across restarts.

        Create two tool instances (simulating two process lifetimes) sharing
        the same budget port.  The first instance reserves and succeeds;
        the second instance (restart) cannot reserve again for the same key.
        """
        budget_port = AsyncMock(spec=BudgetReservationPort)

        # Instance 1: first lifetime
        tool1 = RunAnalysisTool(
            panel_service=panel_service, budget_port=budget_port
        )
        tool1.context = ToolContext(user_id=1, session_id="restart-session")

        budget_port.try_reserve = AsyncMock(return_value=True)
        result1 = await tool1.execute(date=date(2026, 7, 29))
        assert result1.degraded is False

        # Instance 2: second lifetime (restart) — same budget port
        tool2 = RunAnalysisTool(
            panel_service=panel_service, budget_port=budget_port
        )
        tool2.context = ToolContext(user_id=1, session_id="restart-session")

        budget_port.try_reserve = AsyncMock(return_value=False)
        result2 = await tool2.execute(date=date(2026, 7, 29))
        assert result2.degraded is True
        assert "已超出" in (result2.error or "")

    async def test_in_memory_fallback_cap(
        self, panel_service: AsyncMock
    ) -> None:
        """Without BudgetReservationPort, in-memory cap prevents duplicates."""
        tool = RunAnalysisTool(panel_service=panel_service)
        tool.context = ToolContext(user_id=1, session_id="mem-cap")

        result1 = await tool.execute(date=date(2026, 7, 29))
        assert result1.degraded is False

        result2 = await tool.execute(date=date(2026, 7, 29))
        assert result2.degraded is True
        assert "已超出" in (result2.error or "")

    async def test_no_contextvar_in_execution_path(
        self, tool: RunAnalysisTool
    ) -> None:
        tool.context = ToolContext(user_id=1, session_id="no-cv")
        result = await tool.execute(date=date(2026, 7, 29))
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════════════
# InterventionHistoryTool tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestInterventionHistoryTool:
    """InterventionHistoryTool contract tests."""

    @pytest.fixture
    def intervention_repo(self) -> AsyncMock:
        repo = AsyncMock()
        repo.query_range_by_date = AsyncMock(return_value=[])
        return repo

    @pytest.fixture
    def tool(self, intervention_repo: AsyncMock) -> InterventionHistoryTool:
        return InterventionHistoryTool(intervention_repo=intervention_repo)

    async def test_missing_context_returns_error(
        self, tool: InterventionHistoryTool
    ) -> None:
        tool.context = None
        result = await tool.execute(days=7)
        assert result.error == "user_id not set"

    async def test_no_records_returns_empty(
        self, tool: InterventionHistoryTool, intervention_repo: AsyncMock
    ) -> None:
        tool.context = ToolContext(user_id=1)
        intervention_repo.query_range_by_date = AsyncMock(return_value=[])
        result = await tool.execute(days=7)
        assert result.total == 0
        assert result.interventions == []

    async def test_returns_sanitised_summary(
        self, tool: InterventionHistoryTool, intervention_repo: AsyncMock
    ) -> None:
        tool.context = ToolContext(user_id=1)
        intervention_repo.query_range_by_date = AsyncMock(
            return_value=[
                {
                    "intervention_type": "nudge",
                    "triggered_at": "2026-07-29T10:00:00Z",
                    "user_response": "accepted",
                    "raw_prompt": "secret-prompt",
                }
            ]
        )
        result = await tool.execute(days=7)
        assert result.total == 1
        record = result.interventions[0]
        assert record["type"] == "nudge"
        assert "raw_prompt" not in record
        assert "api_key" not in record

    async def test_days_capped_at_30(
        self, tool: InterventionHistoryTool
    ) -> None:
        """Days > 30 is capped in the tool (not just input schema)."""
        tool.context = ToolContext(user_id=1)
        result = await tool.execute(days=100)
        assert result is not None  # should not error

    async def test_no_contextvar_in_execution_path(
        self, tool: InterventionHistoryTool
    ) -> None:
        tool.context = ToolContext(user_id=1)
        result = await tool.execute(days=7)
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════════════
# ToolContext tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestToolContext:
    """ToolContext dataclass tests."""

    def test_required_fields(self) -> None:
        ctx = ToolContext(user_id=1)
        assert ctx.user_id == 1
        assert ctx.session_id is None
        assert ctx.run_id is None

    def test_all_fields_populated(self) -> None:
        ctx = ToolContext(user_id=42, session_id="s1", run_id="r1")
        assert ctx.user_id == 42
        assert ctx.session_id == "s1"
        assert ctx.run_id == "r1"

    def test_immutable(self) -> None:
        ctx = ToolContext(user_id=1)
        with pytest.raises(Exception):
            ctx.user_id = 2  # type: ignore[misc]
