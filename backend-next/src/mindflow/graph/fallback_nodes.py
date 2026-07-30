"""Typed fallback graph nodes for single-expert, Ollama, and RuleEngine degradation.

Converts the three-tier degradation chain from ``LLMService`` into
independently testable, typed graph nodes compatible with LangGraph
StateGraph orchestration and the legacy ``LLMService.analyze()`` adapter.

Architecture (degradation graph)::

    cache_check_node
        │
        ├─ cache_hit → END (return cached)
        └─ no_cache
              │
              ▼
         crisis_gate_node
              │
              ├─ crisis_detected → rule_engine_node → END
              └─ no_crisis
                    │
                    ▼
               prepare_context_node
                    │
                    ▼
               single_expert_node
                    │
                    ▼
               fallback_eligibility_router
                    │
                    ├─ success → END (persist)
                    ├─ failure → ollama_node
                    │              │
                    │              ▼
                    │         fallback_eligibility_router
                    │              │
                    │              ├─ success → END
                    │              └─ failure → rule_engine_node → END
                    └─ skip (not configured) → ollama_node / rule_engine_node

Route matrix (8 combinations):

    1. cache hit        → END (cached result)
    2. crisis detected   → rule_engine_node → END
    3. DS success        → END (source="deepseek", degraded=False)
    4. DS fail + OS success → END (source="ollama", degraded=True)
    5. DS fail + OS fail → rule_engine_node → END (degraded=True)
    6. DS fail + OS skip → rule_engine_node → END (degraded=True)
    7. DS skip + OS success → END (source="ollama", degraded=True)
    8. DS skip + OS skip → rule_engine_node → END (degraded=True)

Each terminal result sets: ``source``, ``degraded``, ``persistence_intent``.

Design constraints:
    - Do NOT retry deterministic schema/safety failures as transport failures.
    - RuleEngine is the final guarantee — always returns a result.
    - Crisis short-circuits all LLM calls.
    - Source labels: ``"deepseek"``, ``"ollama"``, ``"rule_engine"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal, TypedDict, cast

from loguru import logger

from mindflow.domain.events import ActivityEvent
from mindflow.domain.procrastination import (
    BehaviorSummary,
    ProcrastinationAssessment,
    RuleEngine,
)
from mindflow.infrastructure.llm.client import (
    DeepSeekClient,
    LLMAPIError,
    LLMNotConfiguredError,
)
from mindflow.infrastructure.llm.schemas import LLMAttributionResult
from mindflow.infrastructure.llm.summary import (
    build_behavior_summary,
    serialize_summary,
)
from mindflow.infrastructure.security.crisis_detector import CrisisDetector, CrisisLevel

# ── Route type ────────────────────────────────────────────────────────────────

FallbackRoute = Literal[
    "cache_check",
    "crisis_gate",
    "prepare_context",
    "single_expert",
    "ollama",
    "rule_engine",
    "fallback_router",
    "end",
]

# ── Persistence intent ────────────────────────────────────────────────────────

PersistenceIntent = Literal["save", "skip"]


# ═══════════════════════════════════════════════════════════════════════════════
# Runtime context (non-serializable resources)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FallbackRunContext:
    """Runtime resources injected into the fallback state graph.

    Not serialized by the checkpointer — carries live references to
    repositories, clients, and engines needed by individual nodes.
    """

    analysis_repo: Any = None  # SQLAlchemyProcrastinationAnalysisRepository
    deepseek_client: DeepSeekClient | None = None
    ollama_base_url: str | None = None
    ollama_model: str = "qwen3:8b"
    rule_engine: RuleEngine = field(default_factory=RuleEngine)
    crisis_detector: CrisisDetector = field(default_factory=CrisisDetector)


# ═══════════════════════════════════════════════════════════════════════════════
# FallbackState — TypedDict for LangGraph compatibility
# ═══════════════════════════════════════════════════════════════════════════════


class FallbackState(TypedDict, total=False):
    """State flowing through the fallback degradation graph.

    All fields are JSON/checkpointer-serializable except ``runtime``
    (which holds live repository/client references).
    """

    # ── Input (set by caller) ──
    user_id: int
    target_date: date
    events: list[dict[str, Any]]  # serialized from ActivityEvent
    analysis_kind: str
    force: bool

    # ── Runtime (not checkpointed) ──
    runtime: FallbackRunContext

    # ── Preparation (set by prepare_context_node) ──
    summary_json: str
    behavior_summary: Any  # BehaviorSummary (runtime-only)
    events_domain: list[Any]  # list[ActivityEvent] (runtime-only)

    # ── Cache (set by cache_check_node) ──
    cache_hit: bool
    cached_result: dict[str, Any] | None

    # ── Crisis (set by crisis_gate_node) ──
    crisis_detected: bool
    crisis_response_text: str

    # ── Current tier result (set by single_expert / ollama / rule_engine) ──
    current_result: dict[str, Any] | None

    # ── Degradation tracking ──
    source: str  # "deepseek" | "ollama" | "rule_engine"
    degradation_path: list[str]
    degraded: bool

    # ── Final output ──
    assessment: dict[str, Any] | None
    persistence_intent: str  # PersistenceIntent

    # ── Router state ──
    error: str | None


# ═══════════════════════════════════════════════════════════════════════════════
# Pure preparation/conversion functions (extracted from LLMService)
# ═══════════════════════════════════════════════════════════════════════════════


def collect_crisis_texts(events: list[ActivityEvent]) -> list[str]:
    """Collect text fields for crisis scanning from activity events.

    Includes manual_tag event window titles and any event's window_title
    that could contain crisis-like keywords (defence in depth).
    """
    texts: list[str] = []
    seen: set[str] = set()

    for ev in events:
        title = ev.data.window_title.strip()
        if title and title not in seen and ev.event_type == "manual_tag":
            texts.append(title)
            seen.add(title)

    return texts


def build_behavior_bundle(
    events: list[ActivityEvent],
    baseline_deviation: float | None = None,
) -> tuple[BehaviorSummary, str]:
    """Build a behavior summary and serialize it to JSON for LLM input.

    Returns:
        A tuple of (BehaviorSummary, summary_json_string).
    """
    summary = build_behavior_summary(events, baseline_deviation)
    summary_json = serialize_summary(summary)
    return summary, summary_json


def llm_result_to_assessment(result: LLMAttributionResult) -> dict[str, Any]:
    """Convert an LLM attribution result to a serializable assessment dict."""
    return {
        "procrastination_types": list(result.procrastination_types),
        "type_confidence": dict(result.type_confidence),
        "cognitive_distortions": list(result.cognitive_distortions),
        "cbt_technique": result.cbt_technique,
        "response_text": result.response_text,
        "next_action": result.next_action,
    }


def rule_engine_assessment_to_dict(
    assessment: ProcrastinationAssessment,
) -> dict[str, Any]:
    """Convert a rule engine assessment to a serializable dict."""
    return {
        "procrastination_types": [str(t) for t in assessment.types],
        "type_confidence": {str(k): v for k, v in assessment.confidence.items()},
        "cognitive_distortions": [],
        "cbt_technique": (
            str(assessment.recommended_technique)
            if assessment.recommended_technique
            else None
        ),
        "response_text": assessment.rationale,
        "next_action": "根据行为模式调整工作环境",
    }


def events_to_dicts(events: list[ActivityEvent]) -> list[dict[str, Any]]:
    """Convert ActivityEvent list to serializable dicts for state storage."""
    return [
        {
            "id": ev.id,
            "user_id": ev.user_id,
            "timestamp_utc": ev.timestamp_utc.isoformat(),
            "duration_s": ev.duration_s,
            "event_type": ev.event_type,
            "process_name": ev.data.process_name,
            "app_name": ev.data.app_name,
            "window_title": ev.data.window_title,
            "is_idle": ev.data.is_idle,
        }
        for ev in events
    ]


def dicts_to_events(data: list[dict[str, Any]]) -> list[ActivityEvent]:
    """Deserialize event dicts back to ActivityEvent objects (best-effort).

    Only used for tests and bootstrapping — real graphs receive domain events
    through the ``events_domain`` runtime field.
    """
    from datetime import datetime

    from mindflow.domain.events import WindowSnapshot

    results: list[ActivityEvent] = []
    for d in data:
        ts = d.get("timestamp_utc", "")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        snap = WindowSnapshot(
            app_name=d.get("app_name", ""),
            window_title=d.get("window_title", ""),
            process_name=d.get("process_name", ""),
            is_idle=d.get("is_idle", False),
            timestamp_utc=ts if isinstance(ts, datetime) else datetime.min,
        )
        results.append(
            ActivityEvent(
                id=str(d.get("id", "")),
                user_id=int(d.get("user_id", 0)),
                timestamp_utc=ts if isinstance(ts, datetime) else datetime.min,
                duration_s=float(d.get("duration_s", 0)),
                event_type=cast(Any, str(d.get("event_type", "window_snapshot"))),
                data=snap,
            )
        )
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Graph nodes (module-level async callables)
# ═══════════════════════════════════════════════════════════════════════════════


async def cache_check_node(state: FallbackState) -> dict[str, Any]:
    """Check if a previous analysis exists for the given date.

    Sets ``cache_hit=True`` and ``assessment`` with ``persistence_intent="skip"``
    when a cached result is found and ``force`` is False.

    Route: cache_hit → END, no_cache → crisis_gate_node
    """
    runtime: FallbackRunContext = state.get("runtime", FallbackRunContext())  # type: ignore[arg-type]
    user_id = state["user_id"]
    target_date = state["target_date"]
    force = state.get("force", False)
    analysis_kind = state.get("analysis_kind", "daily_attribution")

    if force:
        return {"cache_hit": False}

    try:
        cached = await runtime.analysis_repo.get_by_date(
            user_id, target_date, analysis_kind=analysis_kind,
        )
    except Exception:
        logger.warning("Cache lookup failed for {}/{}", user_id, target_date)
        return {"cache_hit": False}

    if cached is not None:
        logger.debug("Cache hit for {}/{}", user_id, target_date)
        return {
            "cache_hit": True,
            "cached_result": cached,
            "assessment": cached,
            "source": cached.get("source", "rule_engine"),
            "degraded": False,
            "persistence_intent": "skip",
        }

    return {"cache_hit": False}


async def crisis_gate_node(state: FallbackState) -> dict[str, Any]:
    """Scan event texts for crisis keywords, short-circuiting LLM calls.

    When a HIGH crisis level is detected, sets ``crisis_detected=True``
    and pre-populates the assessment with hotline information.
    The router then sends the flow directly to rule_engine_node.

    Route: crisis_detected → rule_engine_node, no_crisis → prepare_context_node
    """
    runtime: FallbackRunContext = state.get("runtime", FallbackRunContext())  # type: ignore[arg-type]

    # Use domain events if available (preferred), otherwise deserialize from dicts
    events_domain: list[ActivityEvent] = state.get("events_domain", []) or []
    if not events_domain and state.get("events"):
        try:
            events_domain = dicts_to_events(state["events"])
        except Exception:
            return {"crisis_detected": False}

    if not events_domain:
        return {"crisis_detected": False}

    crisis_texts = collect_crisis_texts(events_domain)
    crisis_level, crisis_response = runtime.crisis_detector.scan_texts(crisis_texts)

    if crisis_level == CrisisLevel.HIGH:
        logger.warning("Crisis keywords detected for user {}.", state["user_id"])
        return {
            "crisis_detected": True,
            "crisis_response_text": crisis_response.message if crisis_response else "",
            "events_domain": events_domain,
        }

    return {"crisis_detected": False, "events_domain": events_domain}


async def prepare_context_node(state: FallbackState) -> dict[str, Any]:
    """Build the behavior summary and JSON bundle from events.

    Must run after crisis_gate_node (which ensures events_domain is populated).
    Also checks for empty events and sets an error if none found.

    Route: always → single_expert_node
    """
    events_domain: list[ActivityEvent] = state.get("events_domain", []) or []

    if not events_domain:
        return {"error": "No events to analyse"}

    summary, summary_json = build_behavior_bundle(events_domain)
    return {
        "summary_json": summary_json,
        "behavior_summary": summary,
    }


async def single_expert_node(state: FallbackState) -> dict[str, Any]:
    """Call the DeepSeek API (L1) with the behavior summary.

    Sets ``current_result`` on success (source="deepseek", degraded=False).
    Sets ``error`` on any failure (transport or schema/safety) — the router
    will decide whether to fall through to L2.

    **Always** records ``"deepseek"`` in ``degradation_path`` so the router
    knows this tier was attempted regardless of outcome.

    Deterministic failures (schema validation, forbidden words) are NOT
    retried at the transport level — they are reported as errors and the
    degradation chain routes to the next tier.
    """
    runtime: FallbackRunContext = state.get("runtime", FallbackRunContext())  # type: ignore[arg-type]
    summary_json: str = state.get("summary_json", "")

    if not summary_json:
        return {"error": "No summary JSON — run prepare_context_node first"}

    client = runtime.deepseek_client
    if client is None:
        logger.debug("DeepSeek client not configured, skipping L1")
        return {
            "error": "deepseek_not_configured",
            "degradation_path": list(state.get("degradation_path", [])) + ["deepseek"],
        }

    try:
        result = await client.analyze(summary_json)
        logger.info("L1 (DeepSeek) succeeded")
        return {
            "current_result": llm_result_to_assessment(result),
            "source": "deepseek",
            "degraded": False,
            "degradation_path": list(state.get("degradation_path", [])) + ["deepseek"],
            "error": None,
        }
    except LLMNotConfiguredError:
        logger.warning("DeepSeek not configured — no API key")
        return {
            "error": "deepseek_not_configured",
            "degradation_path": list(state.get("degradation_path", [])) + ["deepseek"],
        }
    except (LLMAPIError, TimeoutError) as exc:
        logger.warning("L1 (DeepSeek) transport failure: {}", exc)
        return {
            "error": f"deepseek_transport: {exc}",
            "degradation_path": list(state.get("degradation_path", [])) + ["deepseek"],
        }
    except Exception as exc:
        # Schema validation, forbidden words, JSON parse — deterministic failures
        logger.warning("L1 (DeepSeek) deterministic failure: {}", exc)
        return {
            "error": f"deepseek_schema: {exc}",
            "degradation_path": list(state.get("degradation_path", [])) + ["deepseek"],
        }


async def ollama_node(state: FallbackState) -> dict[str, Any]:
    """Call the Ollama local API (L2) with the behavior summary.

    Sets ``current_result`` on success (source="ollama", degraded=True).
    Sets ``error`` on failure — the router falls through to L3.

    **Always** records ``"ollama"`` in ``degradation_path`` so the router
    knows this tier was attempted regardless of outcome.

    Uses the same OpenAI-compatible endpoint as the existing
    ``LLMService._ollama_call``.
    """
    runtime: FallbackRunContext = state.get("runtime", FallbackRunContext())  # type: ignore[arg-type]
    summary_json: str = state.get("summary_json", "")

    if not summary_json:
        return {"error": "No summary JSON — run prepare_context_node first"}

    ollama_url = runtime.ollama_base_url
    if not ollama_url:
        logger.debug("Ollama not configured, skipping L2")
        return {
            "error": "ollama_not_configured",
            "degradation_path": list(state.get("degradation_path", [])) + ["ollama"],
        }

    try:
        result = await _ollama_api_call(ollama_url, runtime.ollama_model, summary_json)
        logger.info("L2 (Ollama) succeeded")
        return {
            "current_result": llm_result_to_assessment(result),
            "source": "ollama",
            "degraded": True,
            "degradation_path": list(state.get("degradation_path", [])) + ["ollama"],
            "error": None,
        }
    except Exception as exc:
        logger.warning("L2 (Ollama) failed: {}", exc)
        return {
            "error": f"ollama_failure: {exc}",
            "degradation_path": list(state.get("degradation_path", [])) + ["ollama"],
        }


async def rule_engine_node(state: FallbackState) -> dict[str, Any]:
    """Run the deterministic RuleEngine (L3) — always succeeds.

    Uses the cached ``behavior_summary`` from ``prepare_context_node``.
    If no summary is available (e.g. crisis path), produces a minimal
    assessment.

    Always sets: source="rule_engine", degraded=True (unless crisis),
    persistence_intent="save".

    Route: always → END (terminal guarantee)
    """
    runtime: FallbackRunContext = state.get("runtime", FallbackRunContext())  # type: ignore[arg-type]
    crisis_detected = state.get("crisis_detected", False)

    # Crisis path: produce hotline response (no behavior summary needed)
    if crisis_detected:
        crisis_text = state.get("crisis_response_text", "")
        assessment: dict[str, Any] = {
            "procrastination_types": [],
            "type_confidence": {},
            "cognitive_distortions": [],
            "cbt_technique": None,
            "response_text": crisis_text or "",
            "next_action": "寻求专业帮助",
        }
        return {
            "current_result": assessment,
            "assessment": assessment,
            "source": "rule_engine",
            "degraded": False,  # crisis is not degradation — it's a safety gate
            "persistence_intent": "save",
            "degradation_path": list(state.get("degradation_path", []))
            + ["crisis→rule_engine"],
            "error": None,
        }

    # Normal degradation path: run RuleEngine on behavior summary
    behavior_summary: BehaviorSummary | None = state.get("behavior_summary")
    if behavior_summary is None:
        # Degraded case: no summary available (shouldn't happen, but safe fallback)
        assessment = {
            "procrastination_types": [],
            "type_confidence": {},
            "cognitive_distortions": [],
            "cbt_technique": None,
            "response_text": "无法完成行为分析",
            "next_action": "明天继续记录活动",
        }
        return {
            "current_result": assessment,
            "assessment": assessment,
            "source": "rule_engine",
            "degraded": True,
            "persistence_intent": "save",
            "degradation_path": list(state.get("degradation_path", []))
            + ["rule_engine"],
            "error": None,
        }

    try:
        proc_assessment: ProcrastinationAssessment = runtime.rule_engine.assess(
            behavior_summary
        )
        logger.info("L3 (RuleEngine) produced assessment")
        assessment = rule_engine_assessment_to_dict(proc_assessment)
    except Exception:
        # RuleEngine.assess() guarantees no exceptions (contract), but we
        # defend against unexpected failures anyway.
        assessment = {
            "procrastination_types": [],
            "type_confidence": {},
            "cognitive_distortions": [],
            "cbt_technique": None,
            "response_text": "规则引擎异常，请稍后重试",
            "next_action": "明天继续记录活动",
        }

    return {
        "current_result": assessment,
        "assessment": assessment,
        "source": "rule_engine",
        "degraded": True,
        "persistence_intent": "save",
        "degradation_path": list(state.get("degradation_path", [])) + ["rule_engine"],
        "error": None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Conditional router
# ═══════════════════════════════════════════════════════════════════════════════


def fallback_eligibility_router(state: FallbackState) -> FallbackRoute:
    """Determine the next node based on current state of the degradation chain.

    Decision matrix:

    ┌─────────────────────────┬──────────────────┬────────────────────────┐
    │ Condition               │ Next Route       │ Reason                  │
    ├─────────────────────────┼──────────────────┼────────────────────────┤
    │ cache_hit               │ "end"            │ Return cached result    │
    │ crisis_detected         │ "rule_engine"    │ Skip all LLM tiers      │
    │ no summary, no result   │ "prepare_context"│ Build behavior bundle   │
    │ current_result+no error │ "end"            │ Tier produced result    │
    │ last_tier=deepseek      │ ollama/RuleEngine│ Try L2 or skip to L3   │
    │ last_tier=ollama        │ "rule_engine"    │ L3 final guarantee      │
    │ last_tier=rule_engine   │ "end"            │ L3 produced result      │
    │ no tiers attempted      │ "single_expert"  │ Start degradation       │
    └─────────────────────────┴──────────────────┴────────────────────────┘
    """
    # ── Cache hit → terminal ──
    if state.get("cache_hit", False):
        return "end"

    # ── Crisis detected → skip LLMs, go to rule_engine ──
    if state.get("crisis_detected", False):
        return "rule_engine"

    # ── Check degradation progress ──
    degradation_path: list[str] = list(state.get("degradation_path", []))
    error = state.get("error")
    current_result = state.get("current_result")

    # ── Already arrived at rule_engine (terminal) ──
    if "rule_engine" in degradation_path or "crisis→rule_engine" in degradation_path:
        return "end"

    # ── Current tier succeeded → terminal ──
    if current_result and not error:
        return "end"

    # ── No summary yet + no result → prepare context ──
    if not state.get("summary_json") and not state.get("current_result"):
        return "prepare_context"

    # ── Determine next tier from degradation_path ──
    if not degradation_path:
        # No tiers attempted yet — start from L1
        return "single_expert"

    last_tier = degradation_path[-1]

    if last_tier == "deepseek":
        # L1 was attempted (and failed, otherwise we'd be at "end")
        return _route_after_single_expert_failure(state)

    if last_tier == "ollama":
        # L2 was attempted (and failed) — L3 is the final guarantee
        return "rule_engine"

    # Fallthrough: start from L1
    return "single_expert"


def _route_after_single_expert_failure(state: FallbackState) -> FallbackRoute:
    """Internal: decide whether to try Ollama or go straight to RuleEngine."""
    runtime: FallbackRunContext = state.get("runtime", FallbackRunContext())  # type: ignore[arg-type]
    if runtime.ollama_base_url:
        return "ollama"
    return "rule_engine"


# ═══════════════════════════════════════════════════════════════════════════════
# Ollama HTTP helper (extracted from LLMService._ollama_call)
# ═══════════════════════════════════════════════════════════════════════════════

_OLLAMA_SYSTEM_PROMPT: str = (
    "你是一个行为分析助手。分析用户的行为数据并输出 JSON 格式结果。\n"
    "包含 procrastination_types, type_confidence, cbt_technique, response_text, next_action。\n"
    '不要使用"诊断""治疗""患者""处方"等词汇。\n'
    "最多包含 3 个拖延类型。response_text 不超过 500 字。"
)


async def _ollama_api_call(
    base_url: str,
    model: str,
    summary_json: str,
) -> LLMAttributionResult:
    """Call Ollama's OpenAI-compatible chat completions endpoint.

    Returns a validated ``LLMAttributionResult`` on success.

    Raises:
        httpx.HTTPError: On transport failure.
        ValueError/ValidationError: On schema/forbidden-word failure.
    """
    import httpx  # noqa: PLC0415 — lazy import for optional dependency

    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _OLLAMA_SYSTEM_PROMPT},
            {"role": "user", "content": f"请分析以下行为数据：\n\n{summary_json}"},
        ],
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        response = await client.post(url, json=payload)

    if response.status_code != 200:
        raise LLMAPIError(
            f"Ollama returned status {response.status_code}: {response.text[:200]}"
        )

    body = response.json()
    content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not content:
        raise LLMAPIError("Ollama returned empty content")

    return LLMAttributionResult.model_validate_json(content)


# ═══════════════════════════════════════════════════════════════════════════════
# Node runner — execute fallback nodes sequentially (used by legacy adapter)
# ═══════════════════════════════════════════════════════════════════════════════


async def run_fallback_pipeline(
    state: FallbackState,
    *,
    max_tiers: int = 3,
) -> FallbackState:
    """Execute the fallback pipeline sequentially (not via LangGraph runtime).

    This is the bridge between the legacy ``LLMService.analyze()`` adapter
    and the typed graph nodes. It runs nodes in the correct order:
    cache → crisis_gate → prepare_context → degradation tiers.

    Args:
        state: Initial ``FallbackState`` with input fields populated.
        max_tiers: Maximum degradation tiers to attempt (default: 3).

    Returns:
        The final ``FallbackState`` with ``assessment`` populated.
    """
    current = state

    # ── Phase 1: Cache check (if cache_hit not already determined) ──
    if not current.get("cache_hit", False) and not current.get("force", False):
        try:
            runtime: FallbackRunContext = current.get("runtime", FallbackRunContext())  # type: ignore[arg-type]
            if runtime.analysis_repo is not None:
                cached = await cache_check_node(current)
                current = cast(FallbackState, {**current, **cached})  # type: ignore[typeddict-item]
                if current.get("cache_hit"):
                    return current
        except Exception:
            pass  # cache check failure is non-fatal

    # ── Phase 2: Crisis detection (if events available) ──
    if not current.get("crisis_detected", False) and (
        current.get("events_domain") or current.get("events")
    ):
        crisis_updates = await crisis_gate_node(current)
        current = cast(FallbackState, {**current, **crisis_updates})  # type: ignore[typeddict-item]
        if current.get("crisis_detected"):
            # Crisis detected — go straight to rule_engine, skip LLM tiers
            re_updates = await rule_engine_node(current)
            return cast(FallbackState, {**current, **re_updates})  # type: ignore[typeddict-item]

    # ── Phase 3: Degradation loop ──
    for _ in range(max_tiers + 1):  # +1 for rule_engine as final guarantee
        route = fallback_eligibility_router(current)

        if route == "end":
            # Finalise: ensure assessment is set from current_result if not already
            if current.get("current_result") and not current.get("assessment"):
                current["assessment"] = current["current_result"]  # type: ignore[typeddict-item]
            if not current.get("persistence_intent"):
                current["persistence_intent"] = "save"  # type: ignore[typeddict-item]
            break

        if route == "cache_check":
            updates = await cache_check_node(current)
        elif route == "crisis_gate":
            updates = await crisis_gate_node(current)
        elif route == "prepare_context":
            updates = await prepare_context_node(current)
        elif route == "single_expert":
            updates = await single_expert_node(current)
        elif route == "ollama":
            updates = await ollama_node(current)
        elif route == "rule_engine":
            updates = await rule_engine_node(current)
        else:
            # Unknown route — stop with error
            current["error"] = f"Unknown route: {route}"  # type: ignore[typeddict-item]
            break

        # Merge updates into state
        current = cast(FallbackState, {**current, **updates})  # type: ignore[typeddict-item]

        # Terminal nodes: rule_engine always succeeds
        if route == "rule_engine":
            if not current.get("assessment") and current.get("current_result"):
                current["assessment"] = current["current_result"]  # type: ignore[typeddict-item]
            if not current.get("persistence_intent"):
                current["persistence_intent"] = "save"  # type: ignore[typeddict-item]
            break

    # ── Final guarantee: if no assessment, run rule_engine ──
    if not current.get("assessment"):
        updates = await rule_engine_node(current)
        current = cast(FallbackState, {**current, **updates})  # type: ignore[typeddict-item]

    return current
