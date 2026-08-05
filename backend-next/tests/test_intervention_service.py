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
    _enrich_history_item,
    _generate_ollama_message,
    _parse_message_response,
    _render_template_message,
    _select_intervention_type,
)
from mindflow.services.intervention_throttle import ThrottleDecision, ThrottleReason
from mindflow.services.safety_guard import SafetyCheck, SafetyVerdict


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
    """_render_template_message template rendering."""

    def test_gentle_intensity(self) -> None:
        title, body = _render_template_message(
            "nudge", InterventionIntensity.GENTLE, variant_index=0
        )
        assert "小提示" in title
        assert "行动提示" in title
        assert "分心" in body or "延迟" in body

    def test_standard_intensity(self) -> None:
        title, body = _render_template_message(
            "task_breakdown", InterventionIntensity.STANDARD, variant_index=0
        )
        assert "MindFlow" in title
        assert "拆解" in body

    def test_strict_intensity(self) -> None:
        title, body = _render_template_message(
            "environment_optimization", InterventionIntensity.STRICT, variant_index=0
        )
        assert "专注提醒" in title
        assert "干扰" in body

    def test_title_variants_rotate_by_index(self) -> None:
        """Different variant_index yields different titles (deterministic)."""
        titles = {
            _render_template_message("nudge", InterventionIntensity.GENTLE, variant_index=i)[0]
            for i in range(3)
        }
        assert len(titles) == 3

    def test_with_cbt_technique(self) -> None:
        title, body = _render_template_message(
            "task_breakdown", InterventionIntensity.STANDARD, cbt_technique="goal_setting"
        )
        # Uses Chinese label, not raw enum value (P2 requirement)
        assert "目标设定" in body
        assert "goal_setting" not in body


@pytest.fixture
def mock_repo() -> AsyncMock:
    repo = AsyncMock()
    repo.log_triggered = AsyncMock(return_value={"id": "mock-id"})
    repo.update_response = AsyncMock(
        return_value={"id": "mock-id", "user_response": "accepted"}
    )
    repo.update_feedback = AsyncMock(
        return_value={"id": "mock-id", "feedback_rating": "helpful"}
    )
    repo.query_range = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def mock_throttle() -> MagicMock:
    throttle = MagicMock()
    throttle.can_intervene = AsyncMock(
        return_value=ThrottleDecision(ThrottleReason.OK, detail="通过")
    )
    throttle.reserve_slot = AsyncMock(return_value=1)
    return throttle


@pytest.fixture
def mock_notifier() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_broadcast() -> AsyncMock:
    return AsyncMock(return_value=1)


@pytest.fixture
def service(
    mock_repo, mock_throttle, mock_notifier, mock_broadcast
) -> InterventionService:
    return InterventionService(
        intervention_repo=mock_repo,
        throttle=mock_throttle,
        notifier=mock_notifier,
        broadcast_fn=mock_broadcast,
    )


@pytest.fixture
def assessment() -> ProcrastinationAssessment:
    return ProcrastinationAssessment(
        types=(ProcrastinationType.IMPULSIVITY,),
        confidence={ProcrastinationType.IMPULSIVITY: 0.8},
        recommended_technique=CBTTechnique.STIMULUS_CONTROL,
        rationale="检测到冲动分心模式",
        source="rule_engine",
    )


class TestInterventionService:
    """InterventionService orchestration tests."""

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

    async def test_manual_request_can_bypass_deep_work_guard(
        self, service, assessment
    ) -> None:
        events = [
            make_event(
                user_id=1,
                timestamp_utc=datetime(2026, 7, 17, 8, 0, 0, tzinfo=UTC)
                + timedelta(seconds=i * 10),
                duration_s=10.0,
                process_name="Code.exe",
            )
            for i in range(85)
        ]

        result = await service.maybe_intervene(
            assessment=assessment,
            recent_events=events,
            bypass_deep_work_guard=True,
        )

        assert not result.skipped
        assert result.intervention is not None

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

    async def test_context_json_contains_title_and_message(
        self, service, assessment, mock_repo
    ) -> None:
        """context_json persisted by log_triggered includes concrete title and message text."""
        result = await service.maybe_intervene(assessment=assessment)
        assert result.intervention is not None
        mock_repo.log_triggered.assert_awaited_once()
        call_kwargs = mock_repo.log_triggered.await_args[1]
        ctx = call_kwargs["context"]
        assert isinstance(ctx, dict)
        assert "title" in ctx, "context_json must contain the generated title"
        assert "message" in ctx, "context_json must contain the generated message"
        # The title/message should be concrete Chinese text, not a raw enum
        assert ctx["title"] == result.intervention.title
        assert ctx["message"] == result.intervention.message
        assert "environment_optimization" not in str(ctx["title"]), (
            "title must be concrete text, not a raw type enum"
        )

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

    # ── Notification urgency (B3) ────────────────────────────────────

    async def test_notification_urgency_maps_intensity(
        self, service, assessment, mock_notifier
    ) -> None:
        """Notification urgency follows intensity: gentle→low, standard→normal, strict→critical."""
        expected = {
            InterventionIntensity.GENTLE: "low",
            InterventionIntensity.STANDARD: "normal",
            InterventionIntensity.STRICT: "critical",
        }
        for intensity, urgency in expected.items():
            mock_notifier.send.reset_mock()
            result = await service.maybe_intervene(
                assessment=assessment, intensity=intensity
            )
            assert result.intervention is not None
            kwargs = mock_notifier.send.await_args.kwargs
            assert kwargs["urgency"] == urgency, f"intensity={intensity}"

    # ── L1→L2→L3 degradation chain (C2) ─────────────────────────────

    async def test_ollama_fallback_when_no_llm_client(
        self, mock_repo, mock_throttle, mock_notifier, mock_broadcast, assessment, monkeypatch
    ) -> None:
        """No DeepSeek key → Ollama is tried directly (L2)."""
        svc = InterventionService(
            intervention_repo=mock_repo,
            throttle=mock_throttle,
            notifier=mock_notifier,
            broadcast_fn=mock_broadcast,
            ollama_base_url="http://localhost:11434",
            ollama_model="qwen3:8b",
        )

        async def fake_ollama(**kwargs: object) -> tuple[str, str]:
            return ("Ollama标题", "Ollama消息")

        monkeypatch.setattr(
            "mindflow.services.intervention_service._generate_ollama_message",
            fake_ollama,
        )
        result = await svc.maybe_intervene(assessment=assessment)
        assert result.intervention is not None
        assert result.intervention.title == "Ollama标题"
        assert result.intervention.message == "Ollama消息"
        # Persisted context records the LLM as the message source
        mock_repo.log_triggered.assert_awaited_once()
        ctx = mock_repo.log_triggered.await_args.kwargs["context"]
        assert ctx["message_source"] == "llm"

    async def test_ollama_fallback_when_deepseek_fails(
        self, mock_repo, mock_throttle, mock_notifier, mock_broadcast, assessment, monkeypatch
    ) -> None:
        """DeepSeek (L1) fails → Ollama (L2) is tried."""
        svc = InterventionService(
            intervention_repo=mock_repo,
            throttle=mock_throttle,
            notifier=mock_notifier,
            broadcast_fn=mock_broadcast,
            llm_client=MagicMock(),  # non-None L1 client; _generate_llm_message is stubbed
            ollama_base_url="http://localhost:11434",
            ollama_model="qwen3:8b",
        )

        async def fake_llm(**kwargs: object) -> tuple[str, str] | None:
            return None

        async def fake_ollama(**kwargs: object) -> tuple[str, str]:
            return ("Ollama标题", "Ollama消息")

        monkeypatch.setattr(
            "mindflow.services.intervention_service._generate_llm_message",
            fake_llm,
        )
        monkeypatch.setattr(
            "mindflow.services.intervention_service._generate_ollama_message",
            fake_ollama,
        )
        result = await svc.maybe_intervene(assessment=assessment)
        assert result.intervention is not None
        assert result.intervention.title == "Ollama标题"

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

    async def test_get_history_enriches_with_title_message(
        self, service, mock_repo
    ) -> None:
        """get_history enriches new-style rows with concrete title and message."""
        mock_repo.query_range = AsyncMock(return_value=[
            {
                "id": "log-new",
                "intervention_type": "nudge",
                "cbt_technique": None,
                "context_json": {
                    "intensity": "gentle",
                    "title": "休息一下",
                    "message": "你已经工作45分钟了，起来走动两分钟吧",
                },
            },
        ])
        history = await service.get_history(user_id=1, days=7)
        assert len(history) == 1
        assert history[0]["title"] == "休息一下"
        assert history[0]["message"] == "你已经工作45分钟了，起来走动两分钟吧"

    async def test_get_history_provides_fallback_for_legacy_rows(
        self, service, mock_repo
    ) -> None:
        """get_history provides Chinese fallback for legacy rows without stored text."""
        mock_repo.query_range = AsyncMock(return_value=[
            {
                "id": "log-legacy",
                "intervention_type": "task_breakdown",
                "cbt_technique": None,
                "context_json": {"intensity": "standard"},
            },
        ])
        history = await service.get_history(user_id=1, days=7)
        assert len(history) == 1
        assert "title" in history[0]
        assert "message" in history[0]
        # Must be readable Chinese, not a raw enum
        assert "task_breakdown" not in str(history[0]["title"])
        assert "task_breakdown" not in str(history[0]["message"])

    # ── Record feedback ─────────────────────────────────────────────

    async def test_record_feedback(self, service, mock_repo) -> None:
        """record_feedback delegates to repo.update_feedback."""
        result = await service.record_feedback("some-id", "helpful", "很有帮助")
        assert result is not None
        mock_repo.update_feedback.assert_awaited_once_with(
            "some-id", "helpful", "很有帮助"
        )

    async def test_record_feedback_not_found(self, mock_repo, service) -> None:
        """Non-existent ID returns None."""
        mock_repo.update_feedback = AsyncMock(return_value=None)
        result = await service.record_feedback("ghost-id", "annoying")
        assert result is None

    async def test_record_feedback_without_comment(self, service, mock_repo) -> None:
        """Comment is optional."""
        result = await service.record_feedback("some-id", "neutral")
        assert result is not None
        mock_repo.update_feedback.assert_awaited_once_with(
            "some-id", "neutral", None
        )


class TestSafetyGuardWiring:
    """evaluate_safety() is wired after message rendering, before persistence.

    The service must not log/reserve/broadcast/notify when the verdict blocks,
    and must log warnings and continue when the verdict only warns.
    """

    @staticmethod
    def _patch_verdict(
        monkeypatch: pytest.MonkeyPatch,
        *,
        allowed: bool,
        blocked_by: str = "",
        warnings: tuple[str, ...] = (),
    ) -> None:
        level = "pass" if allowed and not warnings else ("warn" if allowed else "block")
        verdict = SafetyVerdict(
            allowed=allowed,
            checks=(SafetyCheck(level=level, reason="safety-reason", category=blocked_by),),
            blocked_by=blocked_by,
            warnings=warnings,
        )
        monkeypatch.setattr(
            "mindflow.services.intervention_service.evaluate_safety",
            lambda *args, **kwargs: verdict,
        )

    async def test_safety_block_returns_skipped_without_side_effects(
        self, service, assessment, mock_repo, mock_throttle, mock_broadcast,
        mock_notifier, monkeypatch,
    ) -> None:
        """Blocked verdict → skipped; no reservation, log, broadcast, or notification."""
        self._patch_verdict(monkeypatch, allowed=False, blocked_by="crisis_language")

        result = await service.maybe_intervene(assessment=assessment)

        assert result.skipped
        assert "安全" in result.skip_reason
        mock_throttle.reserve_slot.assert_not_awaited()
        mock_repo.log_triggered.assert_not_awaited()
        mock_broadcast.assert_not_awaited()
        mock_notifier.send.assert_not_awaited()

    async def test_safety_warning_logs_and_still_dispatches(
        self, service, assessment, mock_repo, mock_broadcast, mock_notifier, monkeypatch,
    ) -> None:
        """Warn-only verdict → continues: persists, broadcasts, and notifies."""
        self._patch_verdict(monkeypatch, allowed=True, warnings=("深夜时段，适合弱干预",))

        result = await service.maybe_intervene(assessment=assessment)

        assert not result.skipped
        assert result.intervention is not None
        mock_repo.log_triggered.assert_awaited_once()
        mock_broadcast.assert_awaited_once()
        mock_notifier.send.assert_awaited_once()


class TestSlotReservation:
    """Daily-slot reservation runs after can_intervene and before persistence."""

    async def test_reservation_contention_returns_skipped(
        self, service, assessment, mock_throttle, mock_repo,
        mock_broadcast, mock_notifier,
    ) -> None:
        """reserve_slot returns None → skipped; no log, broadcast, or notification."""
        mock_throttle.reserve_slot = AsyncMock(return_value=None)

        result = await service.maybe_intervene(assessment=assessment)

        assert result.skipped
        assert "槽位" in result.skip_reason
        mock_repo.log_triggered.assert_not_awaited()
        mock_broadcast.assert_not_awaited()
        mock_notifier.send.assert_not_awaited()

    async def test_reserve_slot_called_before_persist(
        self, service, assessment, mock_throttle, mock_repo,
    ) -> None:
        """reserve_slot is awaited with the user and type; the same flow persists."""
        result = await service.maybe_intervene(assessment=assessment)

        assert not result.skipped
        mock_throttle.reserve_slot.assert_awaited_once()
        mock_repo.log_triggered.assert_awaited_once()

    async def test_persistence_failure_releases_slot_and_returns_skipped(
        self, service, assessment, mock_throttle, mock_repo,
        mock_broadcast, mock_notifier,
    ) -> None:
        """log_triggered raises → reserved slot released, skipped, no broadcast/notify."""
        mock_repo.log_triggered = AsyncMock(side_effect=RuntimeError("db down"))

        result = await service.maybe_intervene(assessment=assessment)

        assert result.skipped
        assert "持久化" in result.skip_reason
        # Release is tied to the exact (user, slot_index, date) reservation.
        mock_repo.release_daily_slot.assert_awaited_once()
        call = mock_repo.release_daily_slot.await_args
        assert call.args == (1, 1)
        assert call.kwargs.get("date_str") is not None
        mock_broadcast.assert_not_awaited()
        mock_notifier.send.assert_not_awaited()

    async def test_bypass_throttle_skips_reservation(
        self, service, assessment, mock_throttle,
    ) -> None:
        """Manual bypass (bypass_throttle=True) does not reserve a slot."""
        result = await service.maybe_intervene(
            assessment=assessment, bypass_throttle=True
        )

        assert not result.skipped
        mock_throttle.reserve_slot.assert_not_awaited()


class TestEnrichHistoryItem:
    """_enrich_history_item — title/message extraction and legacy fallback."""

    def test_new_record_promotes_stored_title_and_message(self) -> None:
        """New-style record with title/message in context_json → promoted to top-level."""
        row: dict[str, object] = {
            "id": "test-1",
            "intervention_type": "nudge",
            "cbt_technique": None,
            "context_json": {
                "intensity": "gentle",
                "title": "该开始了",
                "message": "你已经停留了5分钟，试试番茄钟吧",
            },
        }
        result = _enrich_history_item(row)
        assert result["title"] == "该开始了"
        assert result["message"] == "你已经停留了5分钟，试试番茄钟吧"

    def test_legacy_record_derives_chinese_fallback_with_intensity(self) -> None:
        """Legacy record without stored text → derives fallback using recorded intensity."""
        row: dict[str, object] = {
            "id": "test-2",
            "intervention_type": "environment_optimization",
            "cbt_technique": "stimulus_control",
            "context_json": {
                "intensity": "strict",
                "procrastination_types": ["impulsivity"],
            },
        }
        result = _enrich_history_item(row)
        assert "title" in result
        assert "message" in result
        # Strict intensity uses "专注提醒" title
        assert "专注提醒" in str(result["title"])
        # Should contain Chinese text, NOT a raw type enum
        assert "environment_optimization" not in str(result["title"])
        assert "environment_optimization" not in str(result["message"])

    def test_legacy_record_falls_back_to_standard_when_intensity_missing(self) -> None:
        """Legacy record with no intensity → falls back to STANDARD."""
        row: dict[str, object] = {
            "id": "test-3",
            "intervention_type": "task_breakdown",
            "cbt_technique": None,
            "context_json": {"procrastination_types": ["task_aversion"]},
        }
        result = _enrich_history_item(row)
        assert "title" in result
        assert "message" in result
        # Standard intensity uses "MindFlow" in the title
        assert "MindFlow" in str(result["title"])

    def test_legacy_record_no_context_at_all(self) -> None:
        """Entirely missing context_json → fallback with STANDARD intensity."""
        row: dict[str, object] = {
            "id": "test-4",
            "intervention_type": "smart_prioritization",
            "cbt_technique": None,
            "context_json": None,
        }
        result = _enrich_history_item(row)
        assert "title" in result
        assert "message" in result
        # Must be readable Chinese, not a raw enum
        assert "smart_prioritization" not in str(result["title"])
        assert "smart_prioritization" not in str(result["message"])

    def test_legacy_fallback_never_returns_raw_enum_labels(self) -> None:
        """No fallback path should expose a raw enum like 'environment_optimization'."""
        for itype in (
            "task_breakdown", "nudge", "environment_optimization", "smart_prioritization"
        ):
            row: dict[str, object] = {
                "id": f"test-{itype}",
                "intervention_type": itype,
                "cbt_technique": None,
                "context_json": None,
            }
            result = _enrich_history_item(row)
            assert itype not in str(result["title"]), (
                f"title should not contain raw enum {itype!r}"
            )
            assert itype not in str(result["message"]), (
                f"message should not contain raw enum {itype!r}"
            )


class _FakeResponse:
    """Minimal stand-in for an httpx.Response in Ollama tests."""

    status_code = 200

    def __init__(self, content: str) -> None:
        self._content = content

    def json(self) -> dict[str, object]:
        return {"choices": [{"message": {"content": self._content}}]}


class _FakeClient:
    """Async context manager that records the Ollama POST."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.url: str | None = None
        self.json_payload: dict[str, object] | None = None

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def post(self, url: str, json: dict[str, object]) -> _FakeResponse:
        self.url = url
        self.json_payload = json
        return _FakeResponse(self._content)


class TestParseMessageResponse:
    """_parse_message_response — JSON parsing and length caps."""

    def test_strips_code_fences(self) -> None:
        content = (
            '```json\n{"title": "专注一下", "message": "建议这样做", '
            '"urgency": "medium"}\n```'
        )
        title, message = _parse_message_response(content)
        assert title == "专注一下"
        assert message == "建议这样做"

    def test_plain_json(self) -> None:
        title, message = _parse_message_response(
            '{"title": "专注一下", "message": "建议这样做"}'
        )
        assert title == "专注一下"
        assert message == "建议这样做"

    def test_enforces_title_hard_cap(self) -> None:
        content = '{"title": "' + "长" * 30 + '", "message": "ok"}'
        title, _ = _parse_message_response(content)
        assert len(title) == 15

    def test_enforces_message_cap(self) -> None:
        content = '{"title": "t", "message": "' + "x" * 250 + '"}'
        _, message = _parse_message_response(content)
        assert len(message) == 200  # 197 + "..."
        assert message.endswith("...")

    def test_missing_title_returns_none(self) -> None:
        assert _parse_message_response('{"message": "m"}') is None

    def test_missing_message_returns_none(self) -> None:
        assert _parse_message_response('{"title": "t"}') is None

    def test_invalid_json_returns_none(self) -> None:
        assert _parse_message_response("not json") is None


class TestGenerateOllamaMessage:
    """_generate_ollama_message — L2 fallback HTTP shape and parsing."""

    async def test_posts_to_v1_chat_completions(self, monkeypatch) -> None:
        content = '{"title": "专注一下", "message": "建议这样做", "urgency": "medium"}'
        fake = _FakeClient(content)
        monkeypatch.setattr(
            "mindflow.services.intervention_service.httpx.AsyncClient",
            lambda *args, **kwargs: fake,
        )
        result = await _generate_ollama_message(
            ollama_base_url="http://localhost:11434",
            model="qwen3:8b",
            summary_json="{}",
            intervention_type="nudge",
            intensity="standard",
            cbt_technique=None,
        )
        assert result == ("专注一下", "建议这样做")
        assert fake.url == "/v1/chat/completions"
        assert fake.json_payload is not None
        assert fake.json_payload["model"] == "qwen3:8b"
        assert fake.json_payload["stream"] is False

    async def test_non_200_returns_none(self, monkeypatch) -> None:
        async def fail_post(url: str, json: dict[str, object]) -> _FakeResponse:
            resp = _FakeResponse("{}")
            resp.status_code = 500
            return resp

        class _FailingClient(_FakeClient):
            def __init__(self) -> None:
                super().__init__("{}")

            async def post(self, url: str, json: dict[str, object]) -> _FakeResponse:
                return await fail_post(url, json)

        fake = _FailingClient()
        monkeypatch.setattr(
            "mindflow.services.intervention_service.httpx.AsyncClient",
            lambda *args, **kwargs: fake,
        )
        result = await _generate_ollama_message(
            ollama_base_url="http://localhost:11434",
            model="qwen3:8b",
            summary_json="{}",
            intervention_type="nudge",
            intensity="standard",
        )
        assert result is None
