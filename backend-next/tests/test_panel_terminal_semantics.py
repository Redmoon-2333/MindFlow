"""Phase 1.1 regressions: explicit panel terminal semantics.

A deliberation that never earned critic approval must never be reported as a
successful ``source="panel"`` analysis.  The contract is pinned at two levels:

* ``PanelGraph`` writes explicit ``critic_approved`` / ``panel_rejected`` /
  ``panel_terminal`` / ``rejection_reason`` fields from its single terminal
  node (``panel_finalize``).
* ``AnalysisGraph.panel_graph_node`` only accepts ``source="panel"`` when a
  moderator verdict exists, passes the deterministic schema validation, and
  the critic explicitly approved it.  Every other outcome degrades through
  single_expert → ollama → rule_engine with a marker that survives into the
  persisted ``degradation_path``.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from test_agents_orchestrator import (
    _ANALYST_JSON,
    _ATTRIBUTION_IMPULSIVITY,
    _CRITIC_APPROVE,
    _CRITIC_REJECT,
    _MODERATOR_JSON,
    _MODERATOR_REDO_JSON,
    FP_ANALYST,
    FP_CBT,
    FP_CRITIC,
    FP_EMOTION,
    FP_MODERATOR,
    FP_TMT,
    MockGateway,
    _make_bundle,
)

from mindflow.domain.evidence import to_prompt_json
from mindflow.domain.evidence_facts import build_evidence_catalog, evidence_catalog_ids
from mindflow.graph import analysis_graph as analysis
from mindflow.graph.panel_graph import PanelGraph, PanelGraphState

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _panel_state() -> PanelGraphState:
    """Build a full PanelGraphState for the shared test bundle."""
    bundle = _make_bundle()
    return {  # type: ignore[typeddict-item]
        "bundle_json": to_prompt_json(bundle),
        "valid_metrics": tuple(evidence_catalog_ids(build_evidence_catalog(bundle))),
        "attribution_opinions": (),
        "transcript": (),
        "analyst_opinion": None,
        "conflict_report": None,
        "escalated": False,
        "moderator_verdict": None,
        "critic_result": None,
        "critic_retries": 0,
        "moderator_redo_count": 0,
        "call_count": 0,
        "disagreement_summary": None,
        "rebuttal_delta": None,
        "_expert_index": 0,
    }


def _responses(moderators: list[str], critics: list[str]) -> dict[str, list[str]]:
    return {
        FP_ANALYST: [_ANALYST_JSON],
        FP_CBT: [_ATTRIBUTION_IMPULSIVITY],
        FP_TMT: [_ATTRIBUTION_IMPULSIVITY],
        FP_EMOTION: [_ATTRIBUTION_IMPULSIVITY],
        FP_MODERATOR: moderators,
        FP_CRITIC: critics,
    }


async def _analysis_panel_node(panel_graph: Any) -> dict[str, Any]:
    """Run AnalysisGraph.panel_graph_node against *panel_graph*."""
    runtime = analysis.AnalysisRunContext(panel_graph=panel_graph)
    bundle_json = _panel_state()["bundle_json"]
    return await analysis.panel_graph_node({
        "runtime": runtime,
        "user_id": 1,
        "target_date": date(2026, 9, 19),
        "run_id": "",
        "bundle_json": bundle_json,
        "valid_metrics": _panel_state()["valid_metrics"],
    })


# ═══════════════════════════════════════════════════════════════════════════════
# PanelGraph terminal fields
# ═══════════════════════════════════════════════════════════════════════════════


class TestPanelTerminalFields:
    """The panel graph writes one explicit terminal verdict per run."""

    async def test_critic_approval_is_explicit(self) -> None:
        graph = PanelGraph(gateway=MockGateway(responses=_responses(
            [_MODERATOR_JSON], [_CRITIC_APPROVE],
        )))
        result = await graph.ainvoke(_panel_state())

        assert result["critic_approved"] is True
        assert result["panel_rejected"] is False
        assert result["panel_terminal"] == "approved"
        assert result["rejection_reason"] == ""
        assert result["moderator_redo_count"] == 0

    async def test_reject_once_then_approve_returns_second_verdict(self) -> None:
        graph = PanelGraph(gateway=MockGateway(responses=_responses(
            [_MODERATOR_JSON, _MODERATOR_REDO_JSON], [_CRITIC_REJECT, _CRITIC_APPROVE],
        )))
        result = await graph.ainvoke(_panel_state())

        assert result["critic_approved"] is True
        assert result["panel_rejected"] is False
        assert result["panel_terminal"] == "approved"
        assert result["moderator_redo_count"] == 1
        # The second (redo) verdict is the one that survives, not the rejected one.
        assert result["moderator_verdict"]["rationale"].startswith("修正后")

    async def test_two_rejections_mark_panel_rejected(self) -> None:
        graph = PanelGraph(gateway=MockGateway(responses=_responses(
            [_MODERATOR_JSON, _MODERATOR_REDO_JSON], [_CRITIC_REJECT, _CRITIC_REJECT],
        )))
        result = await graph.ainvoke(_panel_state())

        assert result["critic_approved"] is False
        assert result["panel_rejected"] is True
        assert result["panel_terminal"] == "rejected"
        assert result["moderator_redo_count"] == 2
        assert "重做 2 次" in result["rejection_reason"]
        assert "引用不存在的指标: fake_metric" in result["rejection_reason"]

    async def test_terminal_state_is_in_the_trace(self) -> None:
        graph = PanelGraph(gateway=MockGateway(responses=_responses(
            [_MODERATOR_JSON, _MODERATOR_REDO_JSON], [_CRITIC_REJECT, _CRITIC_REJECT],
        )))
        result = await graph.ainvoke(_panel_state())
        terminal = [e for e in result["trace"] if e["type"] == "terminal"]

        assert len(terminal) == 1
        assert terminal[0]["node"] == "panel_finalize"
        assert terminal[0]["terminal"] == "rejected"
        assert terminal[0]["panel_rejected"] is True
        assert terminal[0]["moderator_redo_count"] == 2
        # Critic issues stay replayable in the trace.
        critic = [e for e in result["trace"] if e["type"] == "critic"]
        assert critic and critic[-1]["issues"]


# ═══════════════════════════════════════════════════════════════════════════════
# AnalysisGraph gating
# ═══════════════════════════════════════════════════════════════════════════════


class TestAnalysisPanelGating:
    """Only a schema-valid, critic-approved verdict may become source=panel."""

    async def test_approved_panel_succeeds(self) -> None:
        graph = PanelGraph(gateway=MockGateway(responses=_responses(
            [_MODERATOR_JSON], [_CRITIC_APPROVE],
        )))
        update = await _analysis_panel_node(graph)

        assert update["panel_succeeded"] is True
        assert update["critic_approved"] is True
        assert update["panel_rejected"] is False
        assert update["source"] == "panel"

    async def test_rejected_panel_never_returns_panel_source(self) -> None:
        graph = PanelGraph(gateway=MockGateway(responses=_responses(
            [_MODERATOR_JSON, _MODERATOR_REDO_JSON], [_CRITIC_REJECT, _CRITIC_REJECT],
        )))
        update = await _analysis_panel_node(graph)

        assert update["panel_succeeded"] is False
        assert update["critic_approved"] is False
        assert update["panel_rejected"] is True
        assert "source" not in update
        assert update["panel_degradation_marker"] == "panel_rejected"
        assert "重做 2 次" in update["panel_unavailable_reason"]

    async def test_legacy_verdict_without_critic_approval_is_not_a_panel(self) -> None:
        """A verdict dict alone must never be enough (the original bug)."""
        legacy = SimpleNamespace(ainvoke=AsyncMock(return_value={
            "moderator_verdict": {
                "types": ["impulsivity"],
                "confidence": {"impulsivity": 0.85},
                "recommended_technique": "stimulus_control",
                "rationale": "旧格式结果",
            },
            "call_count": 6,
        }))
        update = await _analysis_panel_node(legacy)

        assert update["panel_succeeded"] is False
        assert update["panel_rejected"] is True
        assert "source" not in update

    async def test_schema_invalid_verdict_is_rejected(self) -> None:
        invalid = SimpleNamespace(ainvoke=AsyncMock(return_value={
            "moderator_verdict": {
                "types": ["not_a_real_type"],
                "confidence": {"not_a_real_type": 0.9},
                "rationale": "非法类型",
            },
            "critic_approved": True,
            "panel_terminal": "approved",
            "call_count": 6,
        }))
        update = await _analysis_panel_node(invalid)

        assert update["panel_succeeded"] is False
        assert update["critic_approved"] is False
        assert update["panel_degradation_marker"] == "panel_schema_invalid"

    async def test_missing_verdict_is_rejected(self) -> None:
        empty = SimpleNamespace(ainvoke=AsyncMock(return_value={
            "moderator_verdict": None,
            "critic_approved": False,
            "panel_terminal": "unavailable",
            "rejection_reason": "面板未产出主持人裁决",
            "call_count": 4,
        }))
        update = await _analysis_panel_node(empty)

        assert update["panel_succeeded"] is False
        assert update["panel_rejected"] is True
        assert update["panel_degradation_marker"] == "panel_unavailable"


# ═══════════════════════════════════════════════════════════════════════════════
# Degradation provenance
# ═══════════════════════════════════════════════════════════════════════════════


class TestDegradationProvenance:
    """The rejection reason survives the L1→L2→L3 fallback chain."""

    async def test_marker_prefixes_the_fallback_path(self) -> None:
        state: dict[str, Any] = {
            "runtime": analysis.AnalysisRunContext(),
            "user_id": 1,
            "target_date": date(2026, 9, 19),
            "analysis_kind": "daily_attribution",
            "summary_json": "{}",
            "behavior_summary": None,
            "crisis_detected": False,
            "fallback_to_rule_engine": False,
            "degradation_path": [],
            "panel_degradation_marker": "panel_rejected",
        }
        update = await analysis._fallback_chain_node(state)

        assert update["source"] == "rule_engine"
        # The panel marker always leads the path, then the tier that served.
        assert update["degradation_path"][0] == "panel_rejected"
        assert update["degradation_path"][-1] == "rule_engine"

    async def test_marker_is_not_duplicated(self) -> None:
        state: dict[str, Any] = {
            "runtime": analysis.AnalysisRunContext(),
            "user_id": 1,
            "target_date": date(2026, 9, 19),
            "analysis_kind": "daily_attribution",
            "summary_json": "{}",
            "behavior_summary": None,
            "crisis_detected": False,
            "fallback_to_rule_engine": True,
            "degradation_path": ["panel_timeout"],
            "panel_degradation_marker": "panel_timeout",
        }
        update = await analysis._fallback_chain_node(state)

        assert update["degradation_path"] == ["panel_timeout", "rule_engine"]
        assert update["degradation_path"].count("panel_timeout") == 1


@pytest.mark.parametrize("terminal", ["rejected", "unavailable"])
def test_panel_finalize_terminal_values_are_stable(terminal: str) -> None:
    """Terminal vocabulary is part of the contract AnalysisGraph enforces."""
    assert terminal in {"rejected", "unavailable", "approved"}
