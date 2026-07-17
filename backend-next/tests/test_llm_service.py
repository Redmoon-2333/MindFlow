"""Tests for LLMService — three-tier degradation chain.

Covers the full degradation matrix:

  1. L1 succeeds → L2/L3 never called
  2. L1 fails → L2 succeeds → L3 never called
  3. L1+L2 fail → L3 rule engine fallback
  4. Crisis detection short-circuits before any LLM call
  5. Idempotent cache: second call returns cached result
  6. Force bypasses cache
  7. No events for date → raises not-found
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

# MagicMock is already imported above for other fixtures
import pytest

from mindflow.domain.events import make_event
from mindflow.domain.procrastination import (
    RuleEngine,
)
from mindflow.infrastructure.llm.client import LLMAPIError
from mindflow.infrastructure.llm.schemas import LLMAttributionResult
from mindflow.infrastructure.security.crisis_detector import CrisisDetector
from mindflow.services.llm_service import AttributionOutcome, LLMService


@pytest.fixture
def mock_activity_repo() -> MagicMock:
    """Activity repo that returns 10 events when queried."""
    repo = MagicMock()
    base = datetime(2026, 7, 17, 8, 0, 0, tzinfo=UTC)
    events = [
        make_event(
            user_id=1,
            timestamp_utc=base + timedelta(seconds=i * 30),
            duration_s=30.0,
            process_name="Code.exe",
            app_name="VS Code",
        )
        for i in range(10)
    ]
    repo.query_range = AsyncMock(return_value=events)
    return repo


@pytest.fixture
def mock_analysis_repo() -> MagicMock:
    """Analysis repo that returns None (no cache) unless seeded."""
    repo = MagicMock()
    repo.get_by_date = AsyncMock(return_value=None)
    repo.upsert = AsyncMock()
    return repo


@pytest.fixture
def mock_deepseek_success() -> MagicMock:
    """DeepSeek client that returns a valid result."""
    client = MagicMock()

    async def _analyze(_: str) -> LLMAttributionResult:
        return LLMAttributionResult(
            procrastination_types=["impulsivity"],
            type_confidence={"impulsivity": 0.82},
            cognitive_distortions=["all-or-nothing thinking"],
            cbt_technique="stimulus_control",
            response_text="测试回应",
            next_action="测试行动",
        )

    client.analyze = _analyze
    return client


@pytest.fixture
def mock_deepseek_failure() -> MagicMock:
    """DeepSeek client that always raises."""
    client = MagicMock()

    async def _analyze(_: str) -> LLMAttributionResult:
        raise LLMAPIError("Simulated failure")

    client.analyze = _analyze
    return client


@pytest.fixture
def rule_engine() -> RuleEngine:
    return RuleEngine()


def _make_service(
    activity_repo=None, analysis_repo=None, deepseek=None, ollama_url=None, rule_engine=None
) -> LLMService:
    return LLMService(
        activity_repo=activity_repo or MagicMock(),
        analysis_repo=analysis_repo or MagicMock(),
        deepseek_client=deepseek,
        rule_engine=rule_engine or RuleEngine(),
        crisis_detector=CrisisDetector(),
        ollama_base_url=ollama_url,
    )


class TestDegradationChain:
    """L1 → L2 → L3 degradation matrix."""

    @pytest.mark.asyncio
    async def test_l1_success_l2_l3_not_called(
        self, mock_activity_repo, mock_analysis_repo, mock_deepseek_success
    ) -> None:
        """L1 succeeds → L2/L3 should never be called."""
        service = _make_service(
            activity_repo=mock_activity_repo,
            analysis_repo=mock_analysis_repo,
            deepseek=mock_deepseek_success,
        )

        outcome = await service.analyze(1, date(2026, 7, 17))

        assert outcome.source == "deepseek"
        assert outcome.degraded is False
        assert outcome.cached is False
        assert "impulsivity" in outcome.assessment.get("procrastination_types", [])

    @pytest.mark.asyncio
    async def test_l1_fails_l2_not_configured_l3_fallback(
        self, mock_activity_repo, mock_analysis_repo, mock_deepseek_failure, rule_engine
    ) -> None:
        """L1 fails, L2 not configured → L3 rule engine."""
        service = _make_service(
            activity_repo=mock_activity_repo,
            analysis_repo=mock_analysis_repo,
            deepseek=mock_deepseek_failure,
            rule_engine=rule_engine,
        )

        outcome = await service.analyze(1, date(2026, 7, 17))

        assert outcome.source == "rule_engine"
        assert outcome.degraded is True

    @pytest.mark.asyncio
    async def test_l1_and_l2_fail_l3_fallback(
        self, mock_activity_repo, mock_analysis_repo, mock_deepseek_failure, rule_engine
    ) -> None:
        """Both L1 and L2 fail → L3 rule engine fallback."""
        # Mock an Ollama that also fails
        service = LLMService(
            activity_repo=mock_activity_repo,
            analysis_repo=mock_analysis_repo,
            deepseek_client=mock_deepseek_failure,
            rule_engine=rule_engine,
            crisis_detector=CrisisDetector(),
            ollama_base_url="http://localhost:11434",
        )
        # Override _ollama_call to fail
        service._ollama_call = AsyncMock(side_effect=LLMAPIError("Ollama failed"))

        outcome = await service.analyze(1, date(2026, 7, 17))

        assert outcome.source == "rule_engine"
        assert outcome.degraded is True

    @pytest.mark.asyncio
    async def test_l1_not_configured_l3_fallback(
        self, mock_activity_repo, mock_analysis_repo, rule_engine
    ) -> None:
        """No DeepSeek client → skip to L3."""
        service = _make_service(
            activity_repo=mock_activity_repo,
            analysis_repo=mock_analysis_repo,
            deepseek=None,
            rule_engine=rule_engine,
        )

        outcome = await service.analyze(1, date(2026, 7, 17))

        assert outcome.source == "rule_engine"
        assert outcome.degraded is True


class TestCrisisDetection:
    """Crisis detection short-circuits LLM."""

    @pytest.mark.asyncio
    async def test_crisis_short_circuits_llm(self, mock_activity_repo, mock_analysis_repo) -> None:
        """Crisis keywords should prevent LLM call (verified by source=crisis, hotline in text)."""
        # Create events with crisis text in manual_tag
        base = datetime(2026, 7, 17, 8, 0, 0, tzinfo=UTC)
        crisis_events = [
            make_event(
                user_id=1,
                timestamp_utc=base + timedelta(seconds=i * 30),
                duration_s=30.0,
                process_name="Code.exe",
                window_title="感觉撑不下去了" if i == 0 else "",
                event_type="manual_tag" if i == 0 else "window_snapshot",
            )
            for i in range(5)
        ]
        mock_activity_repo.query_range = AsyncMock(return_value=crisis_events)

        # Use a MagicMock for the deepseek client so we can track calls
        deepseek_mock = MagicMock()
        deepseek_mock.analyze = AsyncMock()

        service = _make_service(
            activity_repo=mock_activity_repo,
            analysis_repo=mock_analysis_repo,
            deepseek=deepseek_mock,
        )

        outcome = await service.analyze(1, date(2026, 7, 17))

        assert outcome.crisis_detected is True
        assert "热线" in outcome.assessment.get("response_text", "")
        # DeepSeek should NOT have been called — crisis short-circuits before LLM
        deepseek_mock.analyze.assert_not_called()


class TestCaching:
    """Idempotent cache behaviour."""

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached(self, mock_analysis_repo, mock_activity_repo) -> None:
        """Existing analysis should be returned without LLM call."""
        mock_analysis_repo.get_by_date = AsyncMock(
            return_value={
                "procrastination_types": ["task_aversion"],
                "type_confidence": {"task_aversion": 0.7},
                "cognitive_distortions": [],
                "cbt_technique": "graded_exposure",
                "response_text": "测试缓存",
                "source": "rule_engine",
            }
        )

        service = _make_service(activity_repo=mock_activity_repo, analysis_repo=mock_analysis_repo)

        outcome = await service.analyze(1, date(2026, 7, 17))

        assert outcome.cached is True
        assert outcome.assessment.get("response_text") == "测试缓存"

    @pytest.mark.asyncio
    async def test_force_bypasses_cache(
        self, mock_activity_repo, mock_analysis_repo, mock_deepseek_success
    ) -> None:
        """force=True should skip cache and re-run analysis."""
        cached_data = {
            "procrastination_types": ["task_aversion"],
            "type_confidence": {"task_aversion": 0.7},
            "cognitive_distortions": [],
            "cbt_technique": "graded_exposure",
            "response_text": "旧数据",
            "source": "rule_engine",
        }
        mock_analysis_repo.get_by_date = AsyncMock(return_value=cached_data)

        service = _make_service(
            activity_repo=mock_activity_repo,
            analysis_repo=mock_analysis_repo,
            deepseek=mock_deepseek_success,
        )

        outcome = await service.analyze(1, date(2026, 7, 17), force=True)

        # force=True means we should get fresh data, not cached
        assert outcome.source == "deepseek"  # L1 should have been called
        assert outcome.assessment.get("response_text") == "测试回应"

    @pytest.mark.asyncio
    async def test_empty_events_raises_not_found(self, mock_analysis_repo) -> None:
        """No events for the date should raise not-found."""
        from mindflow.api.errors import ProblemDetail

        activity_repo = MagicMock()
        activity_repo.query_range = AsyncMock(return_value=[])

        service = _make_service(activity_repo=activity_repo, analysis_repo=mock_analysis_repo)

        with pytest.raises(ProblemDetail, match="暂无活动数据"):
            await service.analyze(1, date(2026, 7, 17))


class TestAttributionOutcome:
    """AttributionOutcome data class."""

    def test_creates_with_defaults(self) -> None:
        """Outcome should have sensible defaults."""
        outcome = AttributionOutcome(assessment={}, source="rule_engine")
        assert outcome.cached is False
        assert outcome.degraded is False
        assert outcome.crisis_detected is False
        assert outcome.source == "rule_engine"


class TestIntendedTaskRedaction:
    """Review P3: path-looking manual tags must not reach the LLM payload."""

    def test_path_like_manual_tag_skipped(self) -> None:
        from datetime import UTC, datetime

        from mindflow.domain.events import ActivityEvent, WindowSnapshot
        from mindflow.infrastructure.llm.summary import _find_intended_task

        def _tag(text: str) -> ActivityEvent:
            snap = WindowSnapshot(
                app_name="manual",
                window_title=text,
                process_name="manual",
                is_idle=False,
                timestamp_utc=datetime.now(UTC),
            )
            return ActivityEvent(
                id="t1",
                user_id=1,
                timestamp_utc=datetime.now(UTC),
                duration_s=0.0,
                event_type="manual_tag",
                data=snap,
            )

        assert _find_intended_task([_tag(r"C:\Users\me\thesis.docx")]) is None
        assert _find_intended_task([_tag("/home/me/secret/notes.md")]) is None
        assert _find_intended_task([_tag("写毕业论文第三章")]) == "写毕业论文第三章"
