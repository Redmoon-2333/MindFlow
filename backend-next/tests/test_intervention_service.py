"""Tests for services/intervention_service.py — orchestration logic.

Covers:
  - Deep-work guard (focus_score > 80 → skip)
  - Throttle rejection (daily cap reached)
  - Successful generation, broadcast, notification
  - Response recording
  - History query

Uses mocked dependencies for isolation.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from mindflow.domain.events import make_event
from mindflow.domain.intervention import (
    InterventionIntensity,
)
from mindflow.domain.procrastination import (
    CBTTechnique,
    ProcrastinationAssessment,
    ProcrastinationType,
)
from mindflow.services.intervention_service import (
    InterventionService,
    _deep_work_guard,
    _render_message,
    _select_intervention_type,
)
from mindflow.services.intervention_throttle import ThrottleDecision, ThrottleReason


class TestDeepWorkGuard:
    """_deep_work_guard — deep work detection."""

    def test_high_focus_returns_true(self) -> None:
        """Focus score > 80 should signal deep work."""
        events = []
        # 85 events with the same process_name = high focus
        for i in range(85):
            events.append(make_event(
                user_id=1,
                timestamp_utc=datetime(2026, 7, 17, 8, 0, 0, tzinfo=UTC)
                + timedelta(seconds=i * 10),
                duration_s=10.0,
                process_name="Code.exe",
            ))
        assert _deep_work_guard(events) is True

    def test_low_focus_returns_false(self) -> None:
        """Focus score <= 80 should not signal deep work."""
        events = []
        for i in range(30):
            events.append(make_event(
                user_id=1,
                timestamp_utc=datetime(2026, 7, 17, 8, 0, 0, tzinfo=UTC)
                + timedelta(seconds=i * 30),
                duration_s=5.0,
                process_name=f"App_{i % 3}.exe",
            ))
        assert _deep_work_guard(events) is False

    def test_empty_events_returns_false(self) -> None:
        """No events → not deep work."""
        assert _deep_work_guard([]) is False


class TestSelectInterventionType:
    """_select_intervention_type mapping."""

    def test_task_aversion_maps(self) -> None:
        assessment = ProcrastinationAssessment(
            types=(ProcrastinationType.TASK_AVERSION,),
            confidence={ProcrastinationType.TASK_AVERSION: 0.8},
            recommended_technique=CBTTechnique.GRADED_EXPOSURE,
            rationale="测试原因",
            source="rule_engine",
        )
        assert _select_intervention_type(assessment) == "task_breakdown"

    def test_impulsivity_maps(self) -> None:
        assessment = ProcrastinationAssessment(
            types=(ProcrastinationType.IMPULSIVITY,),
            confidence={ProcrastinationType.IMPULSIVITY: 0.8},
            recommended_technique=CBTTechnique.STIMULUS_CONTROL,
            rationale="测试",
            source="rule_engine",
        )
        assert _select_intervention_type(assessment) == "environment_optimization"

    def test_none_when_no_significant_pattern(self) -> None:
        assessment = ProcrastinationAssessment(
            types=(ProcrastinationType.IMPULSIVITY,),
            confidence={ProcrastinationType.IMPULSIVITY: 0.15},
            recommended_technique=None,
            rationale="无显著模式",
            source="rule_engine",
        )
        # confidence < 0.2 and no technique → None
        assert _select_intervention_type(assessment) is None

    def test_none_when_empty_types(self) -> None:
        assessment = ProcrastinationAssessment(
            types=(),
            confidence={},
            recommended_technique=None,
            rationale="无数据",
            source="rule_engine",
        )
        assert _select_intervention_type(assessment) is None


class TestRenderMessage:
    """_render_message template rendering."""

    def test_gentle_intensity(self) -> None:
        title, body = _render_message("nudge", InterventionIntensity.GENTLE)
        assert "小提示" in title
        assert "行动提示" in title
        assert "分心" in body or "延迟" in body

    def test_standard_intensity(self) -> None:
        title, body = _render_message("task_breakdown", InterventionIntensity.STANDARD)
        assert "MindFlow" in title
        assert "拆解" in body

    def test_strict_intensity(self) -> None:
        title, body = _render_message(
            "environment_optimization", InterventionIntensity.STRICT
        )
        assert "专注提醒" in title
        assert "干扰" in body

    def test_with_cbt_technique(self) -> None:
        title, body = _render_message(
            "task_breakdown", InterventionIntensity.STANDARD, cbt_technique="goal_setting"
        )
        # Uses Chinese label, not raw enum value (P2 requirement)
        assert "目标设定" in body
        assert "goal_setting" not in body


class TestInterventionService:
    """InterventionService orchestration tests."""

    @pytest.fixture
    def mock_repo(self) -> AsyncMock:
        repo = AsyncMock()
        repo.log_triggered = AsyncMock(return_value={"id": "mock-id"})
        repo.update_response = AsyncMock(
            return_value={"id": "mock-id", "user_response": "accepted"}
        )
        repo.query_range = AsyncMock(return_value=[])
        return repo

    @pytest.fixture
    def mock_throttle(self) -> MagicMock:
        throttle = MagicMock()
        throttle.can_intervene = AsyncMock(
            return_value=ThrottleDecision(ThrottleReason.OK, detail="通过")
        )
        return throttle

    @pytest.fixture
    def mock_notifier(self) -> AsyncMock:
        return AsyncMock()

    @pytest.fixture
    def mock_broadcast(self) -> AsyncMock:
        return AsyncMock(return_value=1)

    @pytest.fixture
    def service(
        self, mock_repo, mock_throttle, mock_notifier, mock_broadcast
    ) -> InterventionService:
        return InterventionService(
            intervention_repo=mock_repo,
            throttle=mock_throttle,
            notifier=mock_notifier,
            broadcast_fn=mock_broadcast,
        )

    @pytest.fixture
    def assessment(self) -> ProcrastinationAssessment:
        return ProcrastinationAssessment(
            types=(ProcrastinationType.IMPULSIVITY,),
            confidence={ProcrastinationType.IMPULSIVITY: 0.8},
            recommended_technique=CBTTechnique.STIMULUS_CONTROL,
            rationale="检测到冲动分心模式",
            source="rule_engine",
        )

    # ── Deep work guard ──────────────────────────────────────────────

    async def test_skipped_when_deep_work(
        self, service, assessment
    ) -> None:
        """Deep work (focus_score > 80) → skip."""
        events = []
        for i in range(85):
            events.append(make_event(
                user_id=1,
                timestamp_utc=datetime(2026, 7, 17, 8, 0, 0, tzinfo=UTC)
                + timedelta(seconds=i * 10),
                duration_s=10.0,
                process_name="Code.exe",
            ))
        result = await service.maybe_intervene(
            assessment=assessment,
            recent_events=events,
        )
        assert result.skipped
        assert "深度专注" in result.skip_reason

    # ── Throttle rejection ───────────────────────────────────────────

    async def test_skipped_when_throttled(
        self, mock_throttle, service, assessment
    ) -> None:
        """Throttle denies → skip."""
        mock_throttle.can_intervene = AsyncMock(
            return_value=ThrottleDecision(
                ThrottleReason.DAILY_CAP, detail="已达上限"
            )
        )
        result = await service.maybe_intervene(assessment=assessment)
        assert result.skipped
        assert result.throttle_decision is not None
        assert result.throttle_decision.reason == ThrottleReason.DAILY_CAP

    # ── Successful flow ──────────────────────────────────────────────

    async def test_success_generates_intervention(
        self, service, assessment
    ) -> None:
        """Happy path: generates, persists, broadcasts, notifies."""
        result = await service.maybe_intervene(assessment=assessment)
        assert not result.skipped
        assert result.intervention is not None
        assert result.intervention.intervention_type == "environment_optimization"
        assert result.intervention.cbt_technique == "stimulus_control"
        assert result.intervention.dismissible is True

    async def test_success_broadcasts(
        self, service, assessment, mock_broadcast, mock_notifier
    ) -> None:
        """Successful intervention triggers broadcast + notification."""
        result = await service.maybe_intervene(assessment=assessment)
        assert result.intervention is not None

        # Broadcast was called
        mock_broadcast.assert_awaited_once()
        # await_args is a _Call object — args[0] is the first positional arg
        call_args: tuple = mock_broadcast.await_args.args  # type: ignore[union-attr]
        assert call_args[0]["type"] == "intervention"
        assert call_args[0]["payload"]["intervention_type"] == "environment_optimization"

        # Notification was called
        mock_notifier.send.assert_awaited_once()

    async def test_success_persists_log(
        self, service, assessment, mock_repo
    ) -> None:
        """Successful intervention persists a log entry."""
        result = await service.maybe_intervene(assessment=assessment)
        assert result.intervention is not None
        mock_repo.log_triggered.assert_awaited_once()
        call_kwargs = mock_repo.log_triggered.await_args[1]
        assert call_kwargs["intervention_type"] == "environment_optimization"
        assert call_kwargs["intervention_id"] == result.intervention.id

    # ── Bypass throttle ──────────────────────────────────────────────

    async def test_bypass_throttle(
        self, mock_throttle, service, assessment
    ) -> None:
        """bypass_throttle=True skips throttle check."""
        mock_throttle.can_intervene = AsyncMock(
            return_value=ThrottleDecision(
                ThrottleReason.DAILY_CAP, detail="已达上限"
            )
        )
        result = await service.maybe_intervene(
            assessment=assessment,
            bypass_throttle=True,
        )
        assert not result.skipped
        assert result.intervention is not None

    # ── Record response ──────────────────────────────────────────────

    async def test_record_response(self, service, mock_repo) -> None:
        """record_response delegates to repo."""
        result = await service.record_response("some-id", "accepted", 3.0)
        assert result is not None
        mock_repo.update_response.assert_awaited_once_with(
            "some-id", "accepted", 3.0
        )

    async def test_record_response_not_found(self, mock_repo, service) -> None:
        """Non-existent ID returns None."""
        mock_repo.update_response = AsyncMock(return_value=None)
        result = await service.record_response("ghost-id", "accepted")
        assert result is None

    # ── History ──────────────────────────────────────────────────────

    async def test_get_history(self, service, mock_repo) -> None:
        """get_history delegates to repo.query_range."""
        mock_repo.query_range = AsyncMock(return_value=[
            {"id": "log-1"},
            {"id": "log-2"},
        ])
        history = await service.get_history(user_id=1, days=3)
        assert len(history) == 2
        mock_repo.query_range.assert_awaited_once()
