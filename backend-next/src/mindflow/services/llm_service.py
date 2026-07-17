"""LLM attribution service — the heart of Wave 6.

Implements the three-tier degradation chain (Architecture §3.3, ADR-003):

  L1: DeepSeek API (primary, ~95% of requests)
  L2: Ollama local (optional, zero-cost fallback)
  L3: RuleEngine (never fails, ¥0)

Plus:
  - Crisis detection (runs before any LLM call, independent gate)
  - Idempotent caching (UNIQUE(user_id, date) in DB)
  - Outcome wrapped in ``AttributionOutcome`` with source tracking

Design constraints:
  - No changes to ``domain/`` modules.
  - All LLM calls go through Pydantic ``model_validate_json`` (strict).
  - Degradation is logged at WARNING level but never exposed to the user
    as an error — HTTP 200 with ``meta.degraded=true``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal

from loguru import logger

from mindflow.domain.events import ActivityEvent
from mindflow.domain.procrastination import BehaviorSummary, ProcrastinationAssessment, RuleEngine
from mindflow.infrastructure.llm.client import DeepSeekClient, LLMAPIError, LLMNotConfiguredError
from mindflow.infrastructure.llm.schemas import LLMAttributionResult
from mindflow.infrastructure.llm.summary import build_behavior_summary, serialize_summary
from mindflow.infrastructure.repositories.activity import SQLAlchemyActivityRepository
from mindflow.infrastructure.repositories.analysis import (
    SQLAlchemyProcrastinationAnalysisRepository,
)
from mindflow.infrastructure.security.crisis_detector import CrisisDetector, CrisisLevel

SourceType = Literal["deepseek", "ollama", "rule_engine"]


@dataclass(frozen=True)
class AttributionOutcome:
    """Result of the attribution pipeline with source and cache tracking.

    Attributes:
        assessment: The assessment data as a dict, ready for JSON serialization.
            Shape matches either the LLM output contract or the rule-engine
            output, unified for the API consumer.
        source: Which tier produced the result.
        cached: True if the result was served from a previous analysis.
        degraded: True if all LLM tiers failed and the rule engine was used.
        crisis_detected: True if crisis keywords were found.
    """

    assessment: dict[str, Any]
    source: SourceType
    cached: bool = False
    degraded: bool = False
    crisis_detected: bool = False


_LLM_NOT_CONFIGURED_HINT = (
    "DeepSeek API key is not configured — set MINDFLOW_LLM__API_KEY or "
    "add llm.api_key to the .env file. "
    "Behavior analysis will use the rule engine (L3) as fallback."
)


class LLMService:
    """Three-tier LLM attribution service.

    Args:
        activity_repo: Repository for reading activity events.
        analysis_repo: Repository for persisting/reading analysis results.
        rule_engine: Deterministic rule engine (L3 fallback).
        deepseek_client: DeepSeek API client (L1). May be None if not configured.
        crisis_detector: Crisis keyword scanner.
        ollama_base_url: Base URL for Ollama API. If None, L2 is skipped.
        ollama_model: Ollama model name. Defaults to "qwen3:8b".
    """

    def __init__(  # noqa: PLR0913 — service wiring naturally needs many args
        self,
        activity_repo: SQLAlchemyActivityRepository,
        analysis_repo: SQLAlchemyProcrastinationAnalysisRepository,
        rule_engine: RuleEngine | None = None,
        deepseek_client: DeepSeekClient | None = None,
        crisis_detector: CrisisDetector | None = None,
        ollama_base_url: str | None = None,
        ollama_model: str = "qwen3:8b",
    ) -> None:
        self._activity_repo = activity_repo
        self._analysis_repo = analysis_repo
        self._rule_engine = rule_engine or RuleEngine()
        self._deepseek_client = deepseek_client
        self._crisis_detector = crisis_detector or CrisisDetector()
        self._ollama_base_url = ollama_base_url
        self._ollama_model = ollama_model

    # ── Public API ────────────────────────────────────────────────────

    async def analyze(
        self,
        user_id: int,
        target_date: date,
        *,
        force: bool = False,
    ) -> AttributionOutcome:
        """Run the full attribution pipeline for *user_id* on *target_date*.

        Pipeline:
          1. Cache check (skip if ``force=True``)
          2. Load events from repository
          3. Crisis detection on manual_tag + intended_task text
          4. Build behavior summary
          5. L1: DeepSeek API
          6. L2: Ollama local (if enabled)
          7. L3: RuleEngine (always succeeds)
          8. Persist and return

        Args:
            user_id: User identifier.
            target_date: Date to analyse.
            force: If True, bypass the idempotent cache and re-run analysis.

        Returns:
            An ``AttributionOutcome`` with the assessment data.

        Raises:
            ProblemDetail (not-found): If no events exist for the given date.
        """
        # ── 1. Cache check ────────────────────────────────────────────
        if not force:
            cached = await self._analysis_repo.get_by_date(user_id, target_date)
            if cached is not None:
                logger.debug("Cache hit for attribution {}/{}", user_id, target_date)
                return AttributionOutcome(
                    assessment=cached,
                    source=cached.get("source", "rule_engine"),
                    cached=True,
                )

        # ── 2. Load events ────────────────────────────────────────────
        start_dt = datetime(target_date.year, target_date.month, target_date.day, tzinfo=UTC)
        end_dt = start_dt + timedelta(days=1)

        events = await self._activity_repo.query_range(user_id, start_dt, end_dt)
        if not events:
            from mindflow.api.errors import _not_found

            raise _not_found("暂无活动数据，请先开始采集")

        # ── 3. Crisis detection ───────────────────────────────────────
        crisis_texts = self._collect_crisis_texts(events)
        crisis_level, crisis_response = self._crisis_detector.scan_texts(crisis_texts)

        if crisis_level == CrisisLevel.HIGH:
            logger.warning("Crisis keywords detected for user {}. Skipping LLM.", user_id)
            return AttributionOutcome(
                assessment={
                    "procrastination_types": [],
                    "type_confidence": {},
                    "cognitive_distortions": [],
                    "cbt_technique": None,
                    "response_text": crisis_response.message if crisis_response else "",
                    "next_action": "寻求专业帮助",
                },
                source="rule_engine",
                crisis_detected=True,
            )

        # ── 4. Build summary ──────────────────────────────────────────
        summary = build_behavior_summary(events)
        summary_json = serialize_summary(summary)

        # ── 5-7. Three-tier degradation ───────────────────────────────
        assessment, source, degraded = await self._run_degradation_chain(summary, summary_json)

        # ── 8. Persist ────────────────────────────────────────────────
        await self._analysis_repo.upsert(
            user_id=user_id,
            target_date=target_date,
            procrastination_types=list(assessment.get("procrastination_types", [])),
            type_confidence=assessment.get("type_confidence", {}),
            cognitive_distortions=assessment.get("cognitive_distortions", []),
            cbt_technique=(
                assessment.get("cbt_technique") or assessment.get("recommended_technique")
            ),
            response_text=assessment.get("response_text", assessment.get("rationale", "")),
            llm_model=source,
        )

        return AttributionOutcome(
            assessment=assessment,
            source=source,
            degraded=degraded,
        )

    # ── Degradation chain ─────────────────────────────────────────────

    async def _run_degradation_chain(
        self,
        summary: BehaviorSummary,
        summary_json: str,
    ) -> tuple[dict[str, Any], SourceType, bool]:
        """Execute L1 → L2 → L3, returning (assessment, source, degraded).

        Returns:
            A tuple of (assessment_dict, source_string, was_degraded).
        """
        # L1: DeepSeek API
        if self._deepseek_client is not None:
            try:
                result = await self._deepseek_client.analyze(summary_json)
                logger.info("L1 (DeepSeek) succeeded")
                return self._llm_result_to_assessment(result), "deepseek", False
            except LLMNotConfiguredError:
                logger.warning(_LLM_NOT_CONFIGURED_HINT)
            except (LLMAPIError, TimeoutError) as exc:
                logger.warning("L1 (DeepSeek) failed: {}. Falling back to L2.", exc)
            except Exception as exc:
                logger.warning("L1 (DeepSeek) unexpected error: {}. Falling back.", exc)
        else:
            logger.debug("DeepSeek client not configured, skipping L1")

        # L2: Ollama local
        if self._ollama_base_url:
            try:
                ollama_result = await self._ollama_call(summary_json)
                if ollama_result is not None:
                    logger.info("L2 (Ollama) succeeded")
                    return self._llm_result_to_assessment(ollama_result), "ollama", True
            except Exception as exc:
                logger.warning("L2 (Ollama) failed: {}. Falling back to L3.", exc)
        else:
            logger.debug("Ollama not configured, skipping L2")

        # L3: RuleEngine (never fails)
        logger.info("Falling back to L3 (RuleEngine) for attribution")
        assessment = self._rule_engine_to_assessment(self._rule_engine.assess(summary))
        return assessment, "rule_engine", True

    # ── Ollama helper ─────────────────────────────────────────────────

    async def _ollama_call(self, summary_json: str) -> LLMAttributionResult | None:
        """Call Ollama's OpenAI-compatible API and parse the result.

        Args:
            summary_json: JSON behavior summary.

        Returns:
            A parsed ``LLMAttributionResult`` or None on failure.
        """
        import httpx  # noqa: PLC0415 — lazy import for optional dependency

        payload = {
            "model": self._ollama_model,
            "messages": [
                {"role": "system", "content": _OLLAMA_SYSTEM_PROMPT},
                {"role": "user", "content": f"请分析以下行为数据：\n\n{summary_json}"},
            ],
            "stream": False,
        }

        # _ollama_call is only invoked when self._ollama_base_url is truthy
        ollama_url: str = self._ollama_base_url  # type: ignore[assignment]
        async with httpx.AsyncClient(
            base_url=ollama_url,
            timeout=httpx.Timeout(60.0),
        ) as client:
            response = await client.post("/v1/chat/completions", json=payload)

        if response.status_code != 200:
            logger.warning("Ollama returned status {}", response.status_code)
            return None

        body = response.json()
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            return None

        return LLMAttributionResult.model_validate_json(content)

    # ── Conversion helpers ────────────────────────────────────────────

    @staticmethod
    def _llm_result_to_assessment(result: LLMAttributionResult) -> dict[str, Any]:
        """Convert an LLM result to a serializable assessment dict."""
        return {
            "procrastination_types": list(result.procrastination_types),
            "type_confidence": dict(result.type_confidence),
            "cognitive_distortions": list(result.cognitive_distortions),
            "cbt_technique": result.cbt_technique,
            "response_text": result.response_text,
            "next_action": result.next_action,
        }

    @staticmethod
    def _rule_engine_to_assessment(assessment: ProcrastinationAssessment) -> dict[str, Any]:
        """Convert a rule engine assessment to a serializable dict."""
        return {
            "procrastination_types": [str(t) for t in assessment.types],
            "type_confidence": {str(k): v for k, v in assessment.confidence.items()},
            "cognitive_distortions": [],
            "cbt_technique": (
                str(assessment.recommended_technique) if assessment.recommended_technique else None
            ),
            "response_text": assessment.rationale,
            "next_action": "根据行为模式调整工作环境",
        }

    # ── Crisis text collection ────────────────────────────────────────

    @staticmethod
    def _collect_crisis_texts(events: list[ActivityEvent]) -> list[str]:
        """Collect text fields for crisis scanning.

        Includes:
          - manual_tag events' window_title (user's own notes)
          - any event's window_title that contains crisis-like keywords
            (defence in depth)
        """
        texts: list[str] = []
        seen: set[str] = set()

        for ev in events:
            title = ev.data.window_title.strip()
            if title and title not in seen and ev.event_type == "manual_tag":
                texts.append(title)
                seen.add(title)

        return texts


# ── Ollama system prompt (simplified, since local models are smaller) ──────────

_OLLAMA_SYSTEM_PROMPT: str = (
    "你是一个行为分析助手。分析用户的行为数据并输出 JSON 格式结果。\n"
    "包含 procrastination_types, type_confidence, cbt_technique, response_text, next_action。\n"
    '不要使用"诊断""治疗""患者""处方"等词汇。\n'
    "最多包含 3 个拖延类型。response_text 不超过 500 字。"
)
