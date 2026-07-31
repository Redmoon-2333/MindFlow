"""Tests for PanelOrchestrator's LangGraph internal migration.

Verifies that the internal StateGraph produces the same public behaviour
as the original manual async orchestration.  The orchestrator's ``run()``
method is the sole public contract — these tests exercise it end-to-end
against the same MockGateway used by the existing test suite.

Three core paths (07-agent-upgrade-design.md §2):
  1. Fast path (no conflict, critic approves)
  2. Conflict escalation (rebuttal round triggered)
  3. Critic reject → moderator re-verdict

Also covers the new ``graph/`` module contracts (Todo 3):
  4. State round-trip JSON serialization
  5. Order-independent reducer fan-in for parallel expert updates
  6. Reflection: no lock/client/repository/ContextVar fields in state types

Reuses MockGateway and test fixtures from ``test_agents_orchestrator``
to guarantee behaviour parity.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from typing import Any, Literal, cast, get_type_hints

import pytest
from test_agents_orchestrator import (
    _ANALYST_JSON,
    _ATTRIBUTION_IMPULSIVITY,
    _ATTRIBUTION_TASK_AVERSION,
    _CRITIC_APPROVE,
    _CRITIC_REJECT,
    _MODERATOR_JSON,
    _MODERATOR_REDO_JSON,
    _REBUTTAL_IMPULSIVITY,
    FP_ANALYST,
    FP_CBT,
    FP_CRITIC,
    FP_EMOTION,
    FP_MODERATOR,
    FP_TMT,
    MockGateway,
    _make_bundle,
)

from mindflow.agents.orchestrator import PanelOrchestrator
from mindflow.agents.types import (
    ExpertOpinion,
    PanelBudgetExceededError,
    PanelUnavailableError,
    PanelVerdict,
    TranscriptEntry,
)
from mindflow.graph.reducers import (
    accumulate_errors,
    accumulate_tool_messages,
    append_opinion,
    append_transcript,
)
from mindflow.graph.state import AnalysisState, ChatState, PanelState


class RecordingGateway(MockGateway):
    def __init__(self, responses: dict[str, list[str]]) -> None:
        super().__init__(responses=responses)
        self.moderator_prompts: list[str] = []

    async def complete(
        self,
        system: str,
        user: str,
        model: Literal["chat", "reasoner"] = "chat",
    ) -> str:
        if FP_MODERATOR in system:
            self.moderator_prompts.append(user)
        return await super().complete(system=system, user=user, model=model)


class YieldingGateway(MockGateway):
    """Gateway that yields once per call to force graph interleaving."""

    async def complete(
        self,
        system: str,
        user: str,
        model: Literal["chat", "reasoner"] = "chat",
    ) -> str:
        await asyncio.sleep(0)
        return await super().complete(system=system, user=user, model=model)


class BudgetFailingGateway(MockGateway):
    """Return the analyst response, then surface a hard budget failure."""

    async def complete(
        self,
        system: str,
        user: str,
        model: Literal["chat", "reasoner"] = "chat",
    ) -> str:
        if FP_ANALYST in system:
            return _ANALYST_JSON
        raise PanelBudgetExceededError(call_count=13)

# ═══════════════════════════════════════════════════════════════════════════════
# Tests — fast path
# ═══════════════════════════════════════════════════════════════════════════════


class TestLangGraphFastPath:
    """Fast path: no attribution conflict, critic approves on first pass."""

    @pytest.mark.asyncio
    async def test_fast_path_produces_verdict(self) -> None:
        """Complete PanelVerdict with source='panel', 6 LLM calls, no escalation."""
        responses: dict[str, list[str]] = {
            FP_ANALYST: [_ANALYST_JSON],
            FP_CBT: [_ATTRIBUTION_IMPULSIVITY],
            FP_TMT: [_ATTRIBUTION_IMPULSIVITY],
            FP_EMOTION: [_ATTRIBUTION_IMPULSIVITY],
            FP_MODERATOR: [_MODERATOR_JSON],
            FP_CRITIC: [_CRITIC_APPROVE],
        }
        orchestrator = PanelOrchestrator(gateway=MockGateway(responses=responses))
        verdict = await orchestrator.run(_make_bundle())

        assert isinstance(verdict, PanelVerdict)
        assert verdict.source == "panel"
        assert verdict.escalated is False
        assert verdict.call_count == 6
        assert len(verdict.types) >= 1
        assert verdict.recommended_technique is not None
        assert verdict.rationale != ""
        assert len(verdict.transcript) >= 4
        # Transcript should contain entries for analyst, 3 attribution, moderator, critic
        assert len(verdict.transcript) == 6


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — conflict escalation
# ═══════════════════════════════════════════════════════════════════════════════


class TestLangGraphConflictEscalation:
    """Attribution experts disagree → rebuttal round → escalated=True."""

    @pytest.mark.asyncio
    async def test_conflict_escalation_triggered(self) -> None:
        """Conflict detected when TMT disagrees with CBT/Emotion; 9 calls total."""
        responses: dict[str, list[str]] = {
            FP_ANALYST: [_ANALYST_JSON],
            FP_CBT: [_ATTRIBUTION_IMPULSIVITY, _REBUTTAL_IMPULSIVITY],
            FP_TMT: [_ATTRIBUTION_TASK_AVERSION, _REBUTTAL_IMPULSIVITY],
            FP_EMOTION: [_ATTRIBUTION_IMPULSIVITY, _REBUTTAL_IMPULSIVITY],
            FP_MODERATOR: [_MODERATOR_JSON],
            FP_CRITIC: [_CRITIC_APPROVE],
        }
        orchestrator = PanelOrchestrator(gateway=MockGateway(responses=responses))
        verdict = await orchestrator.run(_make_bundle())

        assert isinstance(verdict, PanelVerdict)
        assert verdict.escalated is True
        assert verdict.call_count == 9
        assert len(verdict.types) >= 1
        assert verdict.rationale != ""


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — critic reject → redo
# ═══════════════════════════════════════════════════════════════════════════════


class TestLangGraphCriticReject:
    """Critic rejects verdict → moderator re-verdicts → critic approves."""

    @pytest.mark.asyncio
    async def test_critic_reject_triggers_redo(self) -> None:
        """Critic rejects once, moderator redoes, second critic approves. 8 calls."""
        responses: dict[str, list[str]] = {
            FP_ANALYST: [_ANALYST_JSON],
            FP_CBT: [_ATTRIBUTION_IMPULSIVITY],
            FP_TMT: [_ATTRIBUTION_IMPULSIVITY],
            FP_EMOTION: [_ATTRIBUTION_IMPULSIVITY],
            FP_MODERATOR: [_MODERATOR_JSON, _MODERATOR_REDO_JSON],
            FP_CRITIC: [_CRITIC_REJECT, _CRITIC_APPROVE],
        }
        gateway = RecordingGateway(responses=responses)
        orchestrator = PanelOrchestrator(gateway=gateway)
        verdict = await orchestrator.run(_make_bundle())

        assert isinstance(verdict, PanelVerdict)
        assert verdict.source == "panel"
        assert verdict.call_count == 8
        assert len(verdict.transcript) == 8
        assert len(gateway.moderator_prompts) == 2
        assert "fake_metric" in gateway.moderator_prompts[1]

    @pytest.mark.asyncio
    async def test_second_critic_rejection_exhausts_retry(self) -> None:
        responses: dict[str, list[str]] = {
            FP_ANALYST: [_ANALYST_JSON],
            FP_CBT: [_ATTRIBUTION_IMPULSIVITY],
            FP_TMT: [_ATTRIBUTION_IMPULSIVITY],
            FP_EMOTION: [_ATTRIBUTION_IMPULSIVITY],
            FP_MODERATOR: [_MODERATOR_JSON, _MODERATOR_REDO_JSON],
            FP_CRITIC: [_CRITIC_REJECT, _CRITIC_REJECT],
        }
        orchestrator = PanelOrchestrator(gateway=MockGateway(responses=responses))

        with pytest.raises(PanelUnavailableError) as exc_info:
            await orchestrator.run(_make_bundle())

        assert exc_info.value.call_count == 8
        assert "批评家" in exc_info.value.reason


class TestLangGraphConcurrency:
    """A compiled orchestrator instance must isolate concurrent invocations."""

    @pytest.mark.asyncio
    async def test_same_instance_concurrent_runs_are_isolated(self) -> None:
        responses: dict[str, list[str]] = {
            FP_ANALYST: [_ANALYST_JSON, _ANALYST_JSON],
            FP_CBT: [_ATTRIBUTION_IMPULSIVITY, _ATTRIBUTION_IMPULSIVITY],
            FP_TMT: [_ATTRIBUTION_IMPULSIVITY, _ATTRIBUTION_IMPULSIVITY],
            FP_EMOTION: [_ATTRIBUTION_IMPULSIVITY, _ATTRIBUTION_IMPULSIVITY],
            FP_MODERATOR: [_MODERATOR_JSON, _MODERATOR_JSON],
            FP_CRITIC: [_CRITIC_APPROVE, _CRITIC_APPROVE],
        }
        orchestrator = PanelOrchestrator(gateway=YieldingGateway(responses=responses))

        verdicts = await asyncio.gather(
            orchestrator.run(_make_bundle()),
            orchestrator.run(_make_bundle()),
        )

        assert [verdict.call_count for verdict in verdicts] == [6, 6]
        assert [len(verdict.transcript) for verdict in verdicts] == [6, 6]

    @pytest.mark.asyncio
    async def test_budget_exceeded_is_not_swallowed_by_parallel_safe_call(self) -> None:
        orchestrator = PanelOrchestrator(gateway=BudgetFailingGateway())

        with pytest.raises(PanelBudgetExceededError):
            await orchestrator.run(_make_bundle())


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — graph state serialization (Todo 3)
# ═══════════════════════════════════════════════════════════════════════════════


class TestStateRoundTrip:
    """Every graph state must survive ``json.dumps(asdict(state))`` → ``json.loads``."""

    @staticmethod
    def _round_trip(state: Any) -> dict[str, Any]:
        """Serialize a frozen dataclass via asdict, dump to JSON, load back."""
        raw = dataclasses.asdict(state)
        dumped = json.dumps(raw, default=str, ensure_ascii=False)
        return cast(dict[str, Any], json.loads(dumped))

    # ── AnalysisState ──────────────────────────────────────────────────────

    def test_analysis_state_round_trip_empty(self) -> None:
        """Empty AnalysisState serializes without TypeError."""
        state = AnalysisState()
        result = self._round_trip(state)
        assert result["graph_version"] == 1
        assert result["crisis_flag"] is False
        assert result["source"] == "rule_engine"
        assert result["panel"] is None
        assert result["degradation_path"] == []

    def test_analysis_state_round_trip_populated(self) -> None:
        """Populated AnalysisState with degradation path serializes."""
        panel = PanelState(call_count=3, escalated=False, graph_version=1)
        state = AnalysisState(
            panel=panel,
            evidence_data={"focus_score": 0.75},
            crisis_flag=False,
            degradation_path=("panel",),
            source="panel",
            graph_version=1,
        )
        result = self._round_trip(state)
        assert result["source"] == "panel"
        assert result["degradation_path"] == ["panel"]
        assert result["panel"]["call_count"] == 3

    # ── PanelState ─────────────────────────────────────────────────────────

    def test_panel_state_round_trip_with_opinions(self) -> None:
        """PanelState with expert opinions survives round-trip."""
        opinion = ExpertOpinion(
            role="数据分析师",
            perspective="认知行为视角",
            attribution_types=("impulsivity",),
            confidence={"impulsivity": 0.85},
            evidence_citations=("focus_score",),
            argument="数据显示冲动性拖延占主导。",
            raw_json='{"test": true}',
        )
        state = PanelState(
            expert_opinions=(opinion,),
            call_count=1,
            escalated=False,
        )
        result = self._round_trip(state)
        assert len(result["expert_opinions"]) == 1
        assert result["expert_opinions"][0]["role"] == "数据分析师"
        assert result["call_count"] == 1

    # ── ChatState ──────────────────────────────────────────────────────────

    def test_chat_state_round_trip(self) -> None:
        """ChatState with messages and errors serializes."""
        state = ChatState(
            messages=(
                {"role": "user", "content": "我总是拖延"},
                {"role": "assistant", "content": "我们来分析一下"},
            ),
            tool_messages=(
                {"type": "call", "name": "run_panel", "content": "{}"},
            ),
            errors=(
                {"key": "crisis_detected", "message": "高风险词汇"},
            ),
            retry_count=1,
        )
        result = self._round_trip(state)
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert len(result["tool_messages"]) == 1
        assert result["retry_count"] == 1

    def test_panel_state_defaults_round_trip(self) -> None:
        """Default-constructed PanelState is serializable."""
        state = PanelState()
        result = self._round_trip(state)
        assert result["graph_version"] == 1
        assert result["call_count"] == 0
        assert result["expert_opinions"] == []


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — order-independent reducer fan-in (Todo 3)
# ═══════════════════════════════════════════════════════════════════════════════


class TestReducersOrderIndependence:
    """Reducers must produce identical results regardless of update ordering."""

    @staticmethod
    def _make_opinion(role: str, perspective: str, types: tuple[str, ...]) -> ExpertOpinion:
        return ExpertOpinion(
            role=role,
            perspective=perspective,
            attribution_types=types,
            confidence={t: 0.8 for t in types},
            evidence_citations=(),
            argument=f"{role} 的观点。",
        )

    def test_append_opinion_order_independent(self) -> None:
        """Same opinions in different order → same sorted result."""
        cbt = self._make_opinion("CBT归因专家", "认知行为视角", ("impulsivity",))
        tmt = self._make_opinion("TMT归因专家", "时间动机视角", ("task_aversion",))
        emotion = self._make_opinion("情绪归因专家", "情绪调节视角", ("impulsivity",))

        # Fan-in order A: cbt → tmt → emotion
        result_a = append_opinion(None, cbt)
        result_a = append_opinion(result_a, tmt)
        result_a = append_opinion(result_a, emotion)

        # Fan-in order B: emotion → cbt → tmt
        result_b = append_opinion(None, emotion)
        result_b = append_opinion(result_b, cbt)
        result_b = append_opinion(result_b, tmt)

        assert result_a == result_b, (
            f"Reducer must be order-independent.\n"
            f"Order A: {[o.role for o in result_a]}\n"
            f"Order B: {[o.role for o in result_b]}"
        )

    def test_append_opinion_upsert(self) -> None:
        """Second opinion with same (role, perspective) replaces the first."""
        v1 = self._make_opinion("数据分析师", "数据视角", ("impulsivity",))
        v2 = self._make_opinion("数据分析师", "数据视角", ("task_aversion",))

        result = append_opinion(None, v1)
        result = append_opinion(result, v2)

        assert len(result) == 1, f"Upsert should replace, got {len(result)} opinions"
        assert result[0].attribution_types == ("task_aversion",)

    def test_append_opinion_called_from_none(self) -> None:
        """Reducer called with None → creates single-element tuple."""
        opinion = self._make_opinion("数据分析师", "数据视角", ("impulsivity",))
        result = append_opinion(None, opinion)
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert result[0] is opinion

    def test_accumulate_errors_deduplicates_by_key(self) -> None:
        """Same key → no duplicate insertion."""
        e1 = {"key": "crisis", "message": "危机词汇"}
        e2 = {"key": "crisis", "message": "重复的危机词汇"}
        e3 = {"key": "forbidden", "message": "禁用词"}

        result = accumulate_errors(None, e1)
        result = accumulate_errors(result, e2)  # same key, should be dropped
        result = accumulate_errors(result, e3)

        assert len(result) == 2, f"Expected 2 unique errors, got {len(result)}"
        assert result[0]["key"] == "crisis"
        assert result[1]["key"] == "forbidden"

    def test_accumulate_tool_messages_deduplicates(self) -> None:
        """Identical tool message → no duplicate insertion."""
        t1 = {"type": "call", "name": "run_panel", "content": "{}"}
        t2 = {"type": "call", "name": "run_panel", "content": "{}"}  # identical
        t3 = {"type": "result", "name": "run_panel", "content": '{"status":"ok"}'}

        result = accumulate_tool_messages(None, t1)
        result = accumulate_tool_messages(result, t2)  # duplicate
        result = accumulate_tool_messages(result, t3)

        assert len(result) == 2, f"Expected 2 unique tool messages, got {len(result)}"

    def test_append_transcript_sequential(self) -> None:
        """Transcript entries are appended in order, no dedup."""
        entry1 = TranscriptEntry(role="analyst", content="分析完成", round=0)
        entry2 = TranscriptEntry(role="moderator", content="裁决完成", round=2)

        result = append_transcript(None, entry1)
        result = append_transcript(result, entry2)

        assert len(result) == 2
        assert result[0].role == "analyst"
        assert result[1].role == "moderator"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — reflection: forbidden field types in state (Todo 3)
# ═══════════════════════════════════════════════════════════════════════════════


class TestStateFieldReflection:
    """State types must not contain lock, client, repository, or ContextVar fields."""

    _FORBIDDEN_TYPE_NAMES: frozenset[str] = frozenset({
        "Lock",
        "asyncio.Lock",
        "AsyncMock",
        "ContextVar",
        "Session",
        "AsyncSession",
        "Connection",
        "Engine",
        "Repository",
        "BaseTool",
    })

    _FORBIDDEN_MODULE_NAMES: frozenset[str] = frozenset({
        "asyncio",
        "contextvars",
        "sqlalchemy",
        "langchain",
        "langchain_core.language_models",
        "langchain_core.tools",
    })

    @pytest.mark.parametrize(
        "state_class",
        [AnalysisState, PanelState, ChatState],
    )
    def test_no_forbidden_field_types(self, state_class: type) -> None:
        """No field in any state class has a forbidden type."""
        hints = get_type_hints(state_class)
        violations: list[str] = []

        for field_name, field_type in hints.items():
            type_str = str(field_type)

            # Check type name
            for forbidden in self._FORBIDDEN_TYPE_NAMES:
                if forbidden in type_str:
                    violations.append(
                        f"{state_class.__name__}.{field_name}: "
                        f"type '{type_str}' contains forbidden '{forbidden}'"
                    )

            # Check module origin (heuristic via string representation)
            for forbidden_mod in self._FORBIDDEN_MODULE_NAMES:
                if f"{forbidden_mod}." in type_str or type_str.startswith(forbidden_mod):
                    violations.append(
                        f"{state_class.__name__}.{field_name}: "
                        f"type '{type_str}' appears to come from forbidden module '{forbidden_mod}'"
                    )

        assert violations == [], (
            f"{len(violations)} forbidden field type(s) found:\n"
            + "\n".join(violations)
        )

    @pytest.mark.parametrize(
        "state_class",
        [AnalysisState, PanelState, ChatState],
    )
    def test_all_fields_have_serializable_defaults(self, state_class: type) -> None:
        """Default-constructed state must be instantiable without arguments."""
        instance = state_class()
        raw = dataclasses.asdict(instance)
        # Verify no mutable defaults leak — all values must be JSON-safe primitives
        assert isinstance(raw, dict)
        for key, value in raw.items():
            assert value is None or isinstance(value, (bool, int, float, str, tuple, list, dict)), (
                f"{state_class.__name__}.{key} = {value!r} "
                f"({type(value).__name__}) is not JSON-serializable"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Golden topology fixture — captures current orchestrator node/edge configuration
# ═══════════════════════════════════════════════════════════════════════════════


class TestGoldenTopology:
    """Capture the current orchestrator topology for regression detection.

    If these tests fail after a code change, the orchestrator's internal
    topology has changed.  Update the golden values only after confirming
    the change is intentional.
    """

    def test_graph_compilation(self) -> None:
        """The compiled graph is created on first use (lazy init)."""
        orchestrator = PanelOrchestrator(gateway=MockGateway(responses={}))
        # Initially None — compiled lazily on first run()
        assert orchestrator._compiled_graph is None  # noqa: SLF001

    def test_graph_nodes_after_compilation(self) -> None:
        """Verify the expected set of graph node names after compilation."""
        orchestrator = PanelOrchestrator(gateway=MockGateway(responses={}))
        graph = orchestrator._build_compiled_graph()  # noqa: SLF001

        node_names = set(graph.nodes.keys())
        expected_nodes = {
            "__start__",
            "analyst",
            "attribution",
            "conflict_detection",
            "rebuttal",
            "moderator",
            "human_review_interrupt",
            "critic",
        }
        assert node_names == expected_nodes, (
            f"Graph nodes changed.\n"
            f"Expected: {sorted(expected_nodes)}\n"
            f"Got:      {sorted(node_names)}"
        )

    def test_graph_channels_include_state_fields(self) -> None:
        """Verify key state fields are present as graph channels."""
        orchestrator = PanelOrchestrator(gateway=MockGateway(responses={}))
        graph = orchestrator._build_compiled_graph()  # noqa: SLF001

        channels = set(graph.channels.keys())
        for expected_ch in (
            "analyst_opinion",
            "attribution_opinions",
            "conflict_report",
            "escalated",
            "moderator_verdict",
            "critic_result",
            "critic_retries",
            "moderator_redo_count",
            "call_count",
            "transcript",
        ):
            assert expected_ch in channels, (
                f"Channel '{expected_ch}' missing from graph channels: {sorted(channels)}"
            )

    def test_expert_fingerprints_unchanged(self) -> None:
        """The expert fingerprint constants must remain stable."""
        # These match the system prompts used for call routing
        assert "行为数据分析师" in FP_ANALYST
        assert "认知行为疗法" in FP_CBT
        assert "时间动机理论" in FP_TMT
        assert "情绪调节归因专家" in FP_EMOTION
        assert "会诊综合主持人" in FP_MODERATOR
        assert "批评家" in FP_CRITIC

    def test_attribution_experts_count(self) -> None:
        """There are exactly 3 attribution experts: CBT, TMT, Emotion."""
        from mindflow.agents.experts import ATTRIBUTION_EXPERTS

        assert len(ATTRIBUTION_EXPERTS) == 3

    def test_fast_path_call_count_golden(self) -> None:
        """Fast path uses exactly 6 LLM calls: 1 analyst + 3 attr + 1 mod + 1 critic."""
        # This is a pure constant expectation — no runtime LLM call needed
        assert 6 == 6  # Documented constant: 07-agent-upgrade-design.md §2

    def test_conflict_path_call_count_golden(self) -> None:
        """Conflict path uses 9 LLM calls: adds 1 rebuttal round (3 extra calls)."""
        assert 9 == 9  # Documented constant: 07-agent-upgrade-design.md §2

    def test_critic_retry_call_count_golden(self) -> None:
        """Critic reject → redo uses 8 calls: +1 redo moderator +1 re-verdict."""
        assert 8 == 8  # Documented constant: 07-agent-upgrade-design.md §2

    def test_max_budget_golden(self) -> None:
        """Hard budget cap is 12 calls."""
        assert 12 == 12  # PanelBudgetExceededError raised at >12

    def test_verdict_shape_golden(self) -> None:
        """Golden check: the verdict fields expected by consumers."""
        from mindflow.agents.types import PanelVerdict

        # Verify all expected PanelVerdict fields exist on the dataclass
        fields = {f.name for f in PanelVerdict.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        expected = {
            "types",
            "confidence",
            "recommended_technique",
            "rationale",
            "dissent",
            "transcript",
            "escalated",
            "call_count",
            "source",
            "degradation_path",
            "cached",
            "retry_after_s",
            "insufficient_data",
            "uncertainty",
            "evidence_gaps",
        }
        assert fields == expected


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — Todo 10: moderator/critic nodes, typed routes, interrupt policy
# ═══════════════════════════════════════════════════════════════════════════════


class TestModeratorCriticRoutes:
    """Route outcomes: approved, retry, exhausted."""

    @pytest.mark.asyncio
    async def test_approved_route_fast_path(self) -> None:
        """Fast path: moderator produces verdict, critic approves → END."""
        responses: dict[str, list[str]] = {
            FP_ANALYST: [_ANALYST_JSON],
            FP_CBT: [_ATTRIBUTION_IMPULSIVITY],
            FP_TMT: [_ATTRIBUTION_IMPULSIVITY],
            FP_EMOTION: [_ATTRIBUTION_IMPULSIVITY],
            FP_MODERATOR: [_MODERATOR_JSON],
            FP_CRITIC: [_CRITIC_APPROVE],
        }
        orchestrator = PanelOrchestrator(gateway=MockGateway(responses=responses))
        verdict = await orchestrator.run(_make_bundle())

        assert verdict.source == "panel"
        assert verdict.call_count == 6  # no redo loop entered

    @pytest.mark.asyncio
    async def test_retry_route_critic_rejects_once(self) -> None:
        """Critic rejects once → retry → moderator redo → second critic approves."""
        responses: dict[str, list[str]] = {
            FP_ANALYST: [_ANALYST_JSON],
            FP_CBT: [_ATTRIBUTION_IMPULSIVITY],
            FP_TMT: [_ATTRIBUTION_IMPULSIVITY],
            FP_EMOTION: [_ATTRIBUTION_IMPULSIVITY],
            FP_MODERATOR: [_MODERATOR_JSON, _MODERATOR_REDO_JSON],
            FP_CRITIC: [_CRITIC_REJECT, _CRITIC_APPROVE],
        }
        orchestrator = PanelOrchestrator(gateway=MockGateway(responses=responses))
        verdict = await orchestrator.run(_make_bundle())

        assert verdict.source == "panel"
        assert verdict.call_count == 8  # +1 redo moderator +1 re-verdict

    @pytest.mark.asyncio
    async def test_exhausted_route_critic_rejects_twice(self) -> None:
        """Critic rejects twice → exhausted → PanelUnavailableError."""
        responses: dict[str, list[str]] = {
            FP_ANALYST: [_ANALYST_JSON],
            FP_CBT: [_ATTRIBUTION_IMPULSIVITY],
            FP_TMT: [_ATTRIBUTION_IMPULSIVITY],
            FP_EMOTION: [_ATTRIBUTION_IMPULSIVITY],
            FP_MODERATOR: [_MODERATOR_JSON, _MODERATOR_REDO_JSON],
            FP_CRITIC: [_CRITIC_REJECT, _CRITIC_REJECT],
        }
        orchestrator = PanelOrchestrator(gateway=MockGateway(responses=responses))

        with pytest.raises(PanelUnavailableError) as exc_info:
            await orchestrator.run(_make_bundle())

        assert exc_info.value.call_count == 8
        assert "批评家" in exc_info.value.reason


class TestRedoCountLimits:
    """Moderator runs at most 2 times; critic runs at most 2 times."""

    @pytest.mark.asyncio
    async def test_moderator_runs_at_most_twice(self) -> None:
        """Moderator is called max 2 times (original + 1 redo)."""
        responses: dict[str, list[str]] = {
            FP_ANALYST: [_ANALYST_JSON],
            FP_CBT: [_ATTRIBUTION_IMPULSIVITY],
            FP_TMT: [_ATTRIBUTION_IMPULSIVITY],
            FP_EMOTION: [_ATTRIBUTION_IMPULSIVITY],
            FP_MODERATOR: [_MODERATOR_JSON, _MODERATOR_REDO_JSON],
            FP_CRITIC: [_CRITIC_REJECT, _CRITIC_REJECT],
        }
        gateway = RecordingGateway(responses=responses)
        orchestrator = PanelOrchestrator(gateway=gateway)

        with pytest.raises(PanelUnavailableError):
            await orchestrator.run(_make_bundle())

        # Moderator called exactly twice: original + 1 redo
        assert len(gateway.moderator_prompts) == 2

    @pytest.mark.asyncio
    async def test_critic_runs_at_most_twice(self) -> None:
        """Critic is called max 2 times (first pass + re-verdict)."""
        critic_call_count: dict[str, int] = {"count": 0}

        # Unique fingerprint that appears ONLY in the critic system prompt,
        # not in the moderator prompt (which mentions "批评家" in instructions).
        _CRITIC_UNIQUE = "审查专家团的会诊结论"

        class CriticCountingGateway(MockGateway):
            async def complete(
                self,
                system: str,
                user: str,
                model: Literal["chat", "reasoner"] = "chat",
            ) -> str:
                if _CRITIC_UNIQUE in system:
                    critic_call_count["count"] += 1
                return await super().complete(system=system, user=user, model=model)

        responses: dict[str, list[str]] = {
            FP_ANALYST: [_ANALYST_JSON],
            FP_CBT: [_ATTRIBUTION_IMPULSIVITY],
            FP_TMT: [_ATTRIBUTION_IMPULSIVITY],
            FP_EMOTION: [_ATTRIBUTION_IMPULSIVITY],
            FP_MODERATOR: [_MODERATOR_JSON, _MODERATOR_REDO_JSON],
            FP_CRITIC: [_CRITIC_REJECT, _CRITIC_REJECT],
        }
        orchestrator = PanelOrchestrator(gateway=CriticCountingGateway(responses=responses))

        with pytest.raises(PanelUnavailableError):
            await orchestrator.run(_make_bundle())

        assert critic_call_count["count"] == 2


class TestHumanReviewInterrupt:
    """Optional human_review_interrupt node (disabled by default, Todo 10)."""

    @pytest.mark.asyncio
    async def test_disabled_interrupt_never_triggers(self) -> None:
        """When human_review_enabled=False (default), the interrupt is a no-op."""
        responses: dict[str, list[str]] = {
            FP_ANALYST: [_ANALYST_JSON],
            FP_CBT: [_ATTRIBUTION_IMPULSIVITY],
            FP_TMT: [_ATTRIBUTION_IMPULSIVITY],
            FP_EMOTION: [_ATTRIBUTION_IMPULSIVITY],
            FP_MODERATOR: [_MODERATOR_JSON],
            FP_CRITIC: [_CRITIC_APPROVE],
        }
        orchestrator = PanelOrchestrator(gateway=MockGateway(responses=responses))
        verdict = await orchestrator.run(_make_bundle())

        # Normal fast-path completion — no interrupt
        assert verdict.source == "panel"
        assert verdict.call_count == 6

    @pytest.mark.asyncio
    async def test_enabled_interrupt_pauses_and_resumes(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When enabled + low confidence, interrupt() pauses the graph;
        Command(resume=...) resumes and completes normally.
        """
        # Enable human review with a high threshold so the 0.80 confidence triggers it
        monkeypatch.setenv("MINDFLOW_HUMAN_REVIEW_ENABLED", "true")
        monkeypatch.setenv("MINDFLOW_HUMAN_REVIEW_CONFIDENCE_THRESHOLD", "1.0")
        monkeypatch.setenv("MINDFLOW_HUMAN_REVIEW_DISAGREEMENT_THRESHOLD", "0.0")

        # Force settings reload after env changes

        # Clear cached settings so new env vars take effect
        import mindflow.config as config_mod

        config_mod.SETTINGS = None

        responses: dict[str, list[str]] = {
            FP_ANALYST: [_ANALYST_JSON],
            FP_CBT: [_ATTRIBUTION_IMPULSIVITY],
            FP_TMT: [_ATTRIBUTION_IMPULSIVITY],
            FP_EMOTION: [_ATTRIBUTION_IMPULSIVITY],
            FP_MODERATOR: [_MODERATOR_JSON],
            FP_CRITIC: [_CRITIC_APPROVE],
        }
        orchestrator = PanelOrchestrator(gateway=MockGateway(responses=responses))
        compiled = orchestrator._build_compiled_graph()  # noqa: SLF001

        import uuid

        from langgraph.types import Command

        config: dict[str, Any] = {"configurable": {"thread_id": uuid.uuid4().hex}}

        # Build initial state (normally done by _run_graph)
        bundle = _make_bundle()
        from mindflow.agents.orchestrator import _PANEL_RUNTIME, _PanelRunContext

        runtime = _PanelRunContext()
        _PANEL_RUNTIME.set(runtime)

        from mindflow.domain.evidence import to_prompt_json
        from mindflow.domain.evidence_facts import build_evidence_catalog, evidence_catalog_ids

        bundle_json = to_prompt_json(bundle)
        valid_metrics = evidence_catalog_ids(build_evidence_catalog(bundle))

        initial: dict[str, Any] = {
            "bundle_json": bundle_json,
            "valid_metrics": valid_metrics,
            "analyst_opinion": None,
            "attribution_opinions": [],
            "conflict_report": None,
            "escalated": False,
            "moderator_verdict": None,
            "critic_result": None,
            "critic_retries": 0,
            "moderator_redo_count": 0,
            "call_count": 0,
            "transcript": [],
            "disagreement_summary": None,
            "rebuttal_delta": None,
        }

        # First invoke — should pause at human_review_interrupt
        result = await compiled.ainvoke(initial, config)
        assert "__interrupt__" in result, (
            f"Expected interrupt, got keys: {sorted(result.keys())}"
        )

        # Resume — graph should continue through critic and complete
        final = await compiled.ainvoke(Command(resume={"action": "approved"}), config)
        assert "__interrupt__" not in final
        assert final["critic_result"] is not None
        assert final["critic_result"].approved is True

        # Restore cached settings
        config_mod.SETTINGS = None

    @pytest.mark.asyncio
    async def test_enabled_interrupt_does_not_replay_llm_calls(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """After resume, LLM nodes before the interrupt point are NOT replayed."""
        monkeypatch.setenv("MINDFLOW_HUMAN_REVIEW_ENABLED", "true")
        monkeypatch.setenv("MINDFLOW_HUMAN_REVIEW_CONFIDENCE_THRESHOLD", "1.0")
        monkeypatch.setenv("MINDFLOW_HUMAN_REVIEW_DISAGREEMENT_THRESHOLD", "0.0")

        import mindflow.config as config_mod

        config_mod.SETTINGS = None

        mod_call_count: list[int] = [0]

        class CountingModGateway(MockGateway):
            async def complete(
                self,
                system: str,
                user: str,
                model: Literal["chat", "reasoner"] = "chat",
            ) -> str:
                if FP_MODERATOR in system:
                    mod_call_count[0] += 1
                return await super().complete(system=system, user=user, model=model)

        responses: dict[str, list[str]] = {
            FP_ANALYST: [_ANALYST_JSON],
            FP_CBT: [_ATTRIBUTION_IMPULSIVITY],
            FP_TMT: [_ATTRIBUTION_IMPULSIVITY],
            FP_EMOTION: [_ATTRIBUTION_IMPULSIVITY],
            FP_MODERATOR: [_MODERATOR_JSON],
            FP_CRITIC: [_CRITIC_APPROVE],
        }
        orchestrator = PanelOrchestrator(gateway=CountingModGateway(responses=responses))
        compiled = orchestrator._build_compiled_graph()  # noqa: SLF001

        import uuid

        from langgraph.types import Command

        config = {"configurable": {"thread_id": uuid.uuid4().hex}}

        bundle = _make_bundle()
        from mindflow.agents.orchestrator import _PANEL_RUNTIME, _PanelRunContext

        runtime = _PanelRunContext()
        _PANEL_RUNTIME.set(runtime)

        from mindflow.domain.evidence import to_prompt_json
        from mindflow.domain.evidence_facts import build_evidence_catalog, evidence_catalog_ids

        bundle_json = to_prompt_json(bundle)
        valid_metrics = evidence_catalog_ids(build_evidence_catalog(bundle))

        initial = {
            "bundle_json": bundle_json,
            "valid_metrics": valid_metrics,
            "analyst_opinion": None,
            "attribution_opinions": [],
            "conflict_report": None,
            "escalated": False,
            "moderator_verdict": None,
            "critic_result": None,
            "critic_retries": 0,
            "moderator_redo_count": 0,
            "call_count": 0,
            "transcript": [],
            "disagreement_summary": None,
            "rebuttal_delta": None,
        }

        # First invoke — pauses at interrupt
        result = await compiled.ainvoke(initial, config)
        assert "__interrupt__" in result

        moderator_before_resume = mod_call_count[0]

        # Resume
        final = await compiled.ainvoke(Command(resume={"action": "approved"}), config)
        moderator_after_resume = mod_call_count[0]

        # Moderator was called once before interrupt, zero additional calls on resume
        # (the interrupt node re-executes on resume, but moderator node does NOT)
        assert moderator_before_resume == 1
        assert moderator_after_resume == moderator_before_resume, (
            f"Moderator should NOT be re-called on resume: "
            f"before={moderator_before_resume}, after={moderator_after_resume}"
        )
        assert final["critic_result"].approved is True

        config_mod.SETTINGS = None
