"""Tests for PanelOrchestrator's LangGraph internal migration.

Verifies that the internal StateGraph produces the same public behaviour
as the original manual async orchestration.  The orchestrator's ``run()``
method is the sole public contract — these tests exercise it end-to-end
against the same MockGateway used by the existing test suite.

Three core paths (07-agent-upgrade-design.md §2):
  1. Fast path (no conflict, critic approves)
  2. Conflict escalation (rebuttal round triggered)
  3. Critic reject → moderator re-verdict

Reuses MockGateway and test fixtures from ``test_agents_orchestrator``
to guarantee behaviour parity.
"""

from __future__ import annotations

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
from mindflow.agents.types import PanelVerdict

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
        orchestrator = PanelOrchestrator(gateway=MockGateway(responses=responses))
        verdict = await orchestrator.run(_make_bundle())

        assert isinstance(verdict, PanelVerdict)
        assert verdict.source == "panel"
        # analyst(1) + 3attribution(3) + moderator(1) + critic(1) + redo(1) + critic2(1) = 8
        assert verdict.call_count == 8
        # Additional transcript entries for redo moderator + second critic
        assert len(verdict.transcript) == 8
