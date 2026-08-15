"""AnalysisGraph — top-level workflow composition root.

Composes the full analysis pipeline as a LangGraph StateGraph:
  1. cache_idempotency_check  — skip if analysis already exists for (user_id, date, kind)
  2. evidence_preparation     — build EvidenceBundle from activity events
  3. crisis_gate              — scan for crisis keywords, short-circuit LLM calls
  4. PanelGraph (subgraph)    — multi-expert panel deliberation (Todo 9)
  5. fallback_chain           — single_expert → ollama → rule_engine (Todo 11)
  6. result_conversion        — assessment dict → PanelVerdict
  7. terminal_persistence     — save analysis + mark run completed + release budget

Implements ``AnalysisWorkflowPort`` from Todo 2.  Distinct idempotency keys
per origin and analysis kind: ``{origin}:{user_id}:{date}:{analysis_kind}``.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, TypedDict, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from mindflow.agents.types import PanelSource, PanelUnavailableError
from mindflow.domain.evidence import EvidenceBundle, to_prompt_json
from mindflow.errors import NoActivityDataError
from mindflow.graph.fallback_nodes import (
    FallbackRunContext,
    FallbackState,
    ollama_node,
    rule_engine_node,
    single_expert_node,
)
from mindflow.graph.panel_graph import PanelGraph, PanelGraphState
from mindflow.ports import (
    AnalysisRequest,
    AnalysisResult,
    BudgetReservationPort,
    OriginType,
    WorkflowRunStorePort,
)
from mindflow.services.panel_service import analysis_dict_to_panel_verdict

# ═══════════════════════════════════════════════════════════════════════════════
# Runtime context — holds live dependencies injected at graph invocation time
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class AnalysisRunContext:
    """Live dependencies injected into the analysis graph at invocation time.

    NOT stored in checkpointable state — LangGraph cannot serialize
    repository references, HTTP clients, or compiled subgraphs.

    All fields default to None so node functions can use
    ``state.get("runtime", AnalysisRunContext())`` as a safe fallback
    during testing.
    """

    # ── Repositories ──
    analysis_repo: Any = None  # ProcrastinationAnalysisRepositoryPort
    workflow_run_repo: WorkflowRunStorePort | None = None
    budget_repo: BudgetReservationPort | None = None

    # ── Reservation ownership ──
    # True only when THIS run won the budget reservation (set by
    # budget_reserve_node on a successful try_reserve).  Used by the
    # run_analysis failure handler so a failing non-owner concurrent run
    # never deletes the winner's reservation.
    budget_owned: bool = False

    # ── Evidence ──
    evidence_builder: Any = None  # EvidenceBundleBuilder

    # ── Crisis ──
    crisis_detector: Any = None  # CrisisDetector

    # ── Panel subgraph ──
    panel_graph: PanelGraph | None = None

    # ── Fallback tier dependencies (shared with FallbackRunContext) ──
    deepseek_client: Any = None  # DeepSeekClient | None
    ollama_base_url: str | None = None
    ollama_model: str = "qwen3:8b"
    rule_engine: Any = None  # RuleEngine

    # ── Timezone ──
    timezone: str = "local"


# ═══════════════════════════════════════════════════════════════════════════════
# Graph state — checkpointable TypedDict
# ═══════════════════════════════════════════════════════════════════════════════


class AnalysisGraphState(TypedDict, total=False):
    """State flowing through the top-level analysis graph.

    All fields are JSON/checkpointer-serializable except ``runtime``
    (which holds live repository/client references).
    """

    # ── Input (set by caller) ──
    user_id: int
    target_date: date
    analysis_kind: str
    origin: OriginType
    force: bool
    idempotency_key: str

    # ── Runtime (not checkpointed) ──
    runtime: AnalysisRunContext

    # ── Run tracking ──
    run_id: str
    budget_reserved: bool

    # ── Cache ──
    cache_hit: bool
    cached_result: dict[str, Any] | None

    # ── Evidence ──
    bundle_json: str
    # tuple (not frozenset) so the state stays JSON/checkpointer-serializable
    # (LangGraph checkpoints serialize state; frozenset is not JSON-safe).
    valid_metrics: tuple[str, ...]

    # ── Crisis ──
    crisis_detected: bool
    crisis_response_text: str
    events_domain: list[Any]  # list[ActivityEvent]

    # ── Panel result ──
    panel_succeeded: bool
    panel_unavailable_reason: str
    fallback_to_rule_engine: bool

    # ── Fallback state (mirrors FallbackState subset needed here) ──
    summary_json: str
    behavior_summary: Any  # BehaviorSummary
    current_result: dict[str, Any] | None
    assessment: dict[str, Any] | None
    source: str
    degradation_path: list[str]
    degraded: bool

    # ── Final conversion ──
    verdict_json: dict[str, Any] | None  # serialized PanelVerdict-friendly dict

    # ── Error ──
    error: str | None
    persistence_failed: bool


# ═══════════════════════════════════════════════════════════════════════════════
# Idempotency key construction
# ═══════════════════════════════════════════════════════════════════════════════


def build_idempotency_key(
    origin: OriginType,
    user_id: int,
    target_date: date,
    analysis_kind: str,
) -> str:
    """Construct a globally-unique idempotency key.

    Format: ``{origin}:{user_id}:{date}:{analysis_kind}``

    ADR-005 prescribes distinct keys per origin so Scheduler/API/Chat
    origins do not block each other, yet converge on the same
    (user_id, date, analysis_kind) storage row.
    """
    return f"{origin}:{user_id}:{target_date.isoformat()}:{analysis_kind}"


def _cached_analysis_meta(cached: dict[str, Any]) -> tuple[str, bool, list[str]]:
    """Derive source, degraded, and degradation path from a stored analysis row.

    A cached fallback result must never be re-labelled as a successful panel.
    """
    source = str(cached.get("source") or cached.get("llm_model") or "rule_engine")
    degraded = bool(cached.get("degraded", source != "panel"))
    path = list(cached.get("degradation_path") or [])
    if not path:
        path = [source]
    return source, degraded, path


_PANEL_WORKFLOW_TIMEOUT_S = 8.0
_COMPETING_ANALYSIS_WAIT_TIMEOUT_S = 5.0
_COMPETING_ANALYSIS_POLL_INTERVAL_S = 0.05


async def _wait_for_competing_analysis(
    runtime: AnalysisRunContext,
    *,
    user_id: int,
    target_date: date,
    analysis_kind: str,
    run_id: str,
) -> dict[str, Any] | None:
    """Wait for the owner of a duplicate key to publish its cached result.

    ``try_reserve(False)`` only tells us that another run owns the key; the
    owner's analysis row may not exist yet.  Returning from the graph at that
    point produces a misleading empty verdict.  The idempotent workflow run
    row is the coordination signal: only a pending/running owner warrants a
    short wait, while missing or terminal rows keep the existing non-blocking
    behavior.
    """
    workflow_repo = runtime.workflow_run_repo
    if workflow_repo is None or not run_id:
        return None

    try:
        owner_run = await workflow_repo.get_run(run_id)
    except Exception as exc:
        logger.debug("Could not inspect competing workflow run {}: {}", run_id, exc)
        return None

    status = getattr(owner_run, "status", None)
    if status not in {"pending", "running"}:
        return None

    deadline = asyncio.get_running_loop().time() + _COMPETING_ANALYSIS_WAIT_TIMEOUT_S
    while True:
        try:
            cached = await runtime.analysis_repo.get_by_date(
                user_id,
                target_date,
                analysis_kind=analysis_kind,
            )
        except Exception as exc:
            logger.debug(
                "Competing analysis cache lookup failed for {}: {}",
                run_id,
                exc,
            )
            cached = None
        if isinstance(cached, dict):
            logger.info("Replayed analysis published by competing run {}", run_id)
            return cached

        try:
            owner_run = await workflow_repo.get_run(run_id)
        except Exception as exc:
            logger.debug("Could not poll competing workflow run {}: {}", run_id, exc)
            return None
        status = getattr(owner_run, "status", None)
        if status not in {"pending", "running"}:
            # A completed owner should have written the analysis before its
            # terminal status.  Do one final cache read, then stop waiting.
            if status == "completed":
                try:
                    cached = await runtime.analysis_repo.get_by_date(
                        user_id,
                        target_date,
                        analysis_kind=analysis_kind,
                    )
                    return cached if isinstance(cached, dict) else None
                except Exception:
                    return None
            return None

        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            logger.warning(
                "Timed out waiting for competing analysis run {} to publish",
                run_id,
            )
            return None
        await asyncio.sleep(min(_COMPETING_ANALYSIS_POLL_INTERVAL_S, remaining))


# ═══════════════════════════════════════════════════════════════════════════════
# Graph nodes
# ═══════════════════════════════════════════════════════════════════════════════


async def cache_idempotency_check_node(
    state: AnalysisGraphState,
) -> dict[str, Any]:
    """Check if a previous analysis exists for the given date and kind.

    When a cached result is found, sets ``cache_hit=True`` and populates
    ``assessment`` so downstream nodes (result_conversion, persistence)
    can complete the run without re-running the analysis.

    Route: cache_hit → result_conversion, no_cache → evidence_preparation
    """
    runtime: AnalysisRunContext = state.get("runtime", AnalysisRunContext())
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
        logger.debug("AnalysisGraph cache hit for {}/{}", user_id, target_date)
        source, degraded, path = _cached_analysis_meta(cached)
        return {
            "cache_hit": True,
            "cached_result": cached,
            "assessment": cached,
            "source": source,
            "degraded": degraded,
            "degradation_path": path,
            "error": None,
        }

    return {"cache_hit": False}


async def budget_reserve_node(state: AnalysisGraphState) -> dict[str, Any]:
    """Atomically reserve budget for this run via idempotency_key.

    Uses ``BudgetReservationPort.try_reserve`` which INSERTs with
    ON CONFLICT DO NOTHING — first caller wins, subsequent return False.

    When reservation fails (another run already claimed this key),
    we re-check the cache as a fallback — the prior run may have already
    completed the analysis.
    """
    runtime: AnalysisRunContext = state.get("runtime", AnalysisRunContext())
    idempotency_key = state.get("idempotency_key", "")
    user_id = state["user_id"]
    target_date = state["target_date"]
    analysis_kind = state.get("analysis_kind", "daily_attribution")

    if not idempotency_key:
        logger.warning("No idempotency_key provided; skipping budget reservation")
        return {"budget_reserved": False}

    budget_repo = runtime.budget_repo
    if budget_repo is None:
        logger.warning("No budget repository configured; skipping budget reservation")
        return {"budget_reserved": False}

    reserved = await budget_repo.try_reserve(idempotency_key)

    if not reserved:
        # Another run already claimed this key.  Check if the analysis
        # was already completed by that prior run.
        logger.info(
            "Budget reservation failed for {} — checking if analysis already exists",
            idempotency_key,
        )
        try:
            cached = await runtime.analysis_repo.get_by_date(
                user_id, target_date, analysis_kind=analysis_kind,
            )
            if cached is not None:
                source, degraded, path = _cached_analysis_meta(cached)
                return {
                    "budget_reserved": False,
                    "cache_hit": True,
                    "cached_result": cached,
                    "assessment": cached,
                    "source": source,
                    "degraded": degraded,
                    "degradation_path": path,
                }
        except Exception:
            cached = None

        # A duplicate request can observe the reservation before the owner
        # reaches terminal persistence.  Wait for that owner to publish the
        # cache rather than routing directly to END (which becomes an empty
        # verdict in run_analysis()).
        if cached is None:
            cached = await _wait_for_competing_analysis(
                runtime,
                user_id=user_id,
                target_date=target_date,
                analysis_kind=analysis_kind,
                run_id=state.get("run_id", ""),
            )
        if cached is not None:
            source, degraded, path = _cached_analysis_meta(cached)
            return {
                "budget_reserved": False,
                "cache_hit": True,
                "cached_result": cached,
                "assessment": cached,
                "source": source,
                "degraded": degraded,
                "degradation_path": path,
            }

        return {
            "budget_reserved": False,
            "fallback_to_rule_engine": True,
            "degradation_path": ["duplicate_timeout"],
        }

    logger.debug("Budget reserved for {}", idempotency_key)
    runtime.budget_owned = True
    return {"budget_reserved": True}


async def evidence_preparation_node(
    state: AnalysisGraphState,
) -> dict[str, Any]:
    """Build the evidence bundle for the target date.

    Delegates to ``EvidenceBundleBuilder.build()`` to fetch activity
    events, compute features, build the behavior summary, and produce
    a JSON-serializable bundle for the panel subgraph.
    """
    from mindflow.time_utils import business_day_bounds_utc

    runtime: AnalysisRunContext = state.get("runtime", AnalysisRunContext())
    user_id = state["user_id"]
    target_date = state["target_date"]

    window_start, window_end = business_day_bounds_utc(
        target_date, runtime.timezone,
    )

    bundle: EvidenceBundle = await runtime.evidence_builder.build(
        user_id, window_start, window_end,
    )

    if isinstance(bundle, EvidenceBundle) and not bundle.events:
        raise NoActivityDataError("暂无活动数据，请先开始采集")

    # Build the JSON bundle for the panel subgraph
    bundle_json = to_prompt_json(bundle)

    # Collect valid metrics from the evidence catalog
    from mindflow.domain.evidence_facts import build_evidence_catalog, evidence_catalog_ids

    catalog = build_evidence_catalog(bundle)
    valid_metrics = tuple(evidence_catalog_ids(catalog))

    return {
        "bundle_json": bundle_json,
        "valid_metrics": valid_metrics,
        "events_domain": list(getattr(bundle, "events", [])),
    }


async def crisis_gate_node(state: AnalysisGraphState) -> dict[str, Any]:
    """Scan event texts for crisis keywords, short-circuiting LLM calls.

    When a HIGH crisis level is detected, sets ``crisis_detected=True``
    and pre-populates the crisis response text.  The router then sends
    the flow directly to the fallback chain's rule_engine_node.
    """
    from mindflow.graph.fallback_nodes import collect_crisis_texts, dicts_to_events
    from mindflow.infrastructure.security.crisis_detector import CrisisLevel

    runtime: AnalysisRunContext = state.get("runtime", AnalysisRunContext())

    # Use domain events if available, otherwise deserialize from dicts
    events_domain: list[Any] = state.get("events_domain", []) or []
    if not events_domain:
        return {"crisis_detected": False, "events_domain": []}

    # Filter to ActivityEvent instances; deserialize dicts if needed
    from mindflow.domain.events import ActivityEvent

    typed_events: list[ActivityEvent] = [
        e for e in events_domain if isinstance(e, ActivityEvent)
    ]
    if not typed_events and events_domain:
        try:
            typed_events = dicts_to_events(
                [e if isinstance(e, dict) else {} for e in events_domain]
            )
        except Exception:
            return {"crisis_detected": False, "events_domain": events_domain}

    if not typed_events:
        return {"crisis_detected": False, "events_domain": events_domain}

    crisis_texts = collect_crisis_texts(typed_events)
    crisis_level, crisis_response = runtime.crisis_detector.scan_texts(crisis_texts)

    if crisis_level == CrisisLevel.HIGH:
        logger.warning(
            "AnalysisGraph: Crisis keywords detected for user {}.",
            state["user_id"],
        )
        return {
            "crisis_detected": True,
            "crisis_response_text": crisis_response.message if crisis_response else "",
            "events_domain": events_domain,
        }

    return {"crisis_detected": False, "events_domain": events_domain}


async def panel_graph_node(state: AnalysisGraphState) -> dict[str, Any]:
    """Run the expert panel subgraph (PanelGraph from Todo 9).

    Invokes the compiled PanelGraph with a PanelGraphState containing
    the evidence bundle.  On success, extracts the moderator verdict
    as the assessment.  On failure (PanelUnavailableError), sets
    ``panel_succeeded=False`` so the router sends the flow to the
    fallback chain.
    """
    runtime: AnalysisRunContext = state.get("runtime", AnalysisRunContext())
    bundle_json = state.get("bundle_json", "")
    valid_metrics = state.get("valid_metrics", ())

    if not bundle_json:
        return {
            "panel_succeeded": False,
            "panel_unavailable_reason": "No evidence bundle available",
        }

    panel_graph = runtime.panel_graph
    if panel_graph is None:
        return {
            "panel_succeeded": False,
            "panel_unavailable_reason": "PanelGraph not configured",
        }

    # Build input state for the panel subgraph
    panel_state: PanelGraphState = {
        "bundle_json": bundle_json,
        "valid_metrics": valid_metrics,
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

    try:
        result = await asyncio.wait_for(
            panel_graph.ainvoke(panel_state),
            timeout=_PANEL_WORKFLOW_TIMEOUT_S,
        )
    except TimeoutError:
        logger.warning(
            "PanelGraph timed out after {}s; using RuleEngine fallback",
            _PANEL_WORKFLOW_TIMEOUT_S,
        )
        return {
            "panel_succeeded": False,
            "panel_unavailable_reason": "PanelGraph timed out",
            "fallback_to_rule_engine": True,
            "degradation_path": ["panel_timeout"],
        }
    except PanelUnavailableError as exc:
        logger.warning("PanelGraph unavailable: {}", exc)
        return {
            "panel_succeeded": False,
            "panel_unavailable_reason": str(exc),
        }
    except Exception as exc:
        logger.warning("PanelGraph unexpected error: {}", exc)
        return {
            "panel_succeeded": False,
            "panel_unavailable_reason": f"PanelGraph error: {exc}",
        }

    verdict_dict = result.get("moderator_verdict") if isinstance(result, dict) else None
    escalated = result.get("escalated", False) if isinstance(result, dict) else False
    call_count = result.get("call_count", 0) if isinstance(result, dict) else 0

    # Persist per-node trace payloads so every panel run is replayable.
    trace = result.get("trace", []) if isinstance(result, dict) else []
    run_repo = runtime.workflow_run_repo
    if trace and state.get("run_id") and run_repo is not None:
        try:
            for entry in trace:
                await run_repo.save_node_event(
                    state["run_id"],
                    str(entry.get("node", "panel")),
                    payload=entry,
                )
        except Exception as exc:
            logger.warning("Failed to persist panel trace: {}", exc)

    if verdict_dict is None or not isinstance(verdict_dict, dict):
        return {
            "panel_succeeded": False,
            "panel_unavailable_reason": "Moderator did not produce a verdict",
        }

    logger.info(
        "PanelGraph succeeded ({} calls, escalated={})",
        call_count,
        escalated,
    )

    return {
        "panel_succeeded": True,
        "assessment": verdict_dict,
        "source": "panel",
        "degraded": False,
        "degradation_path": ["panel"],
        "error": None,
    }


async def prepare_fallback_context_node(
    state: AnalysisGraphState,
) -> dict[str, Any]:
    """Build the behavior summary for the fallback chain.

    Converts events_domain into a behavior summary suitable for the
    L1→L2→L3 degradation chain.  Must run after panel failure (or
    crisis detection) before entering the fallback chain.
    """
    from mindflow.graph.fallback_nodes import build_behavior_bundle

    events_domain: list[Any] = state.get("events_domain", []) or []

    if not events_domain:
        # No events — produce a minimal summary for rule_engine
        return {
            "summary_json": "{}",
            "behavior_summary": None,
        }

    summary, summary_json = build_behavior_bundle(events_domain)
    return {
        "summary_json": summary_json,
        "behavior_summary": summary,
    }


async def result_conversion_node(state: AnalysisGraphState) -> dict[str, Any]:
    """Convert the assessment dict to a PanelVerdict-friendly dict.

    Uses ``analysis_dict_to_panel_verdict``, the single source of truth
    for verdict construction from raw dicts.  The resulting verdict dict
    is stored in ``verdict_json`` for terminal persistence.
    """
    assessment = state.get("assessment")
    source = cast(PanelSource, state.get("source", "rule_engine"))
    degraded = state.get("degraded", True)

    if assessment is None:
        # No assessment — produce a minimal fallback
        assessment = {
            "procrastination_types": [],
            "type_confidence": {},
            "cognitive_distortions": [],
            "cbt_technique": None,
            "response_text": "无法完成分析",
            "source": source,
        }

    # Ensure source is in the assessment dict for proper verdict construction
    if "source" not in assessment:
        assessment = dict(assessment)
        assessment["source"] = source

    try:
        verdict = analysis_dict_to_panel_verdict(assessment, source=source)
    except Exception as exc:
        logger.warning("Verdict conversion failed: {}", exc)
        return {
            "verdict_json": None,
            "error": f"verdict_conversion: {exc}",
        }

    # Serialize the verdict back to a dict for checkpointable state
    verdict_dict: dict[str, Any] = {
        "types": [t.value for t in verdict.types],
        "confidence": {k.value: v for k, v in verdict.confidence.items()},
        "recommended_technique": (
            verdict.recommended_technique.value
            if verdict.recommended_technique is not None
            else None
        ),
        "rationale": verdict.rationale,
        "dissent": list(verdict.dissent),
        "transcript": [
            {"role": e.role, "content": e.content, "round": e.round}
            for e in verdict.transcript
        ],
        "escalated": verdict.escalated,
        "call_count": verdict.call_count,
        "source": verdict.source,
        "degraded": degraded,
        "degradation_path": (
            list(verdict.degradation_path) or list(state.get("degradation_path", []) or [])
        ),
        "cached": bool(state.get("cache_hit", False)),
        "insufficient_data": verdict.insufficient_data,
        "uncertainty": verdict.uncertainty,
        "evidence_gaps": list(verdict.evidence_gaps),
    }

    return {
        "verdict_json": verdict_dict,
        "degraded": degraded,
    }


async def terminal_persistence_node(
    state: AnalysisGraphState,
) -> dict[str, Any]:
    """Persist analysis, mark run completed, and release budget — atomically.

    This is THE SINGLE terminal persistence node.  It does ALL three:
      1. Save analysis (upsert — idempotent via ON CONFLICT DO UPDATE)
      2. Mark workflow run as completed
      3. Release budget reservation

    Side-effect nodes are idempotent: calling this twice with the same
    state is safe (upsert is a no-op on same data, update_status is a
    no-op when already "completed", release is idempotent).

    Crash-after-persist guarantee: if the analysis was saved but the
    run was not marked completed, the next invocation's
    ``cache_idempotency_check`` will find the existing analysis and
    route through this node again, completing the run.
    """
    runtime: AnalysisRunContext = state.get("runtime", AnalysisRunContext())
    user_id = state["user_id"]
    target_date = state["target_date"]
    analysis_kind = state.get("analysis_kind", "daily_attribution")
    assessment = state.get("assessment")
    source = cast(PanelSource, state.get("source", "rule_engine"))
    degraded = state.get("degraded", True)
    run_id = state.get("run_id", "")
    idempotency_key = state.get("idempotency_key", "")
    verdict_json = state.get("verdict_json")

    if assessment is None:
        assessment = {
            "procrastination_types": [],
            "type_confidence": {},
            "cognitive_distortions": [],
            "cbt_technique": None,
            "response_text": "无分析结果",
            "source": source,
        }

    # ── 1. Persist analysis (upsert — idempotent) ──────────────────────
    # Extract types and confidence from either verdict_json (preferred)
    # or the raw assessment dict.
    procrastination_types: list[str] = []
    type_confidence: dict[str, float] = {}
    response_text: str = ""
    cbt_technique: str | None = None
    panel_transcript: dict[str, Any] | None = None

    if verdict_json is not None:
        procrastination_types = list(verdict_json.get("types", []))
        type_confidence = dict(verdict_json.get("confidence", {}))
        response_text = str(verdict_json.get("rationale", ""))
        cbt_technique = verdict_json.get("recommended_technique")
        if verdict_json.get("transcript"):
            panel_transcript = {
                "transcript": list(verdict_json["transcript"]),
                "dissent": list(verdict_json.get("dissent", [])),
                "escalated": bool(verdict_json.get("escalated", False)),
                "call_count": int(verdict_json.get("call_count", 0)),
                "degradation_path": list(state.get("degradation_path", []) or []),
                "cached": bool(verdict_json.get("cached", state.get("cache_hit", False))),
                "insufficient_data": bool(verdict_json.get("insufficient_data", False)),
                "uncertainty": verdict_json.get("uncertainty"),
                "evidence_gaps": list(verdict_json.get("evidence_gaps", [])),
            }
    else:
        procrastination_types = list(
            assessment.get("types", assessment.get("procrastination_types", []))
        )
        type_confidence = dict(
            assessment.get("confidence", assessment.get("type_confidence", {}))
        )
        response_text = str(
            assessment.get("rationale", assessment.get("response_text", ""))
        )
        cbt_technique = (
            assessment.get("recommended_technique")
            or assessment.get("cbt_technique")
        )

    try:
        await runtime.analysis_repo.upsert(
            user_id=user_id,
            target_date=target_date,
            procrastination_types=procrastination_types,
            type_confidence=type_confidence,
            cognitive_distortions=list(
                assessment.get("cognitive_distortions", [])
            ),
            cbt_technique=cbt_technique,
            response_text=response_text,
            llm_model=source,
            analysis_kind=analysis_kind,
            source=source,
            panel_transcript=panel_transcript,
            degraded=degraded,
            degradation_path=list(state.get("degradation_path", []) or []),
        )
        logger.info(
            "Analysis persisted for user {} on {} (source={}, kind={})",
            user_id,
            target_date,
            source,
            analysis_kind,
        )
    except Exception as exc:
        logger.error("Analysis persistence failed: {}", exc)
        return {
            "error": f"persistence: {exc}",
            "persistence_failed": True,
        }

    # ── 2. Mark workflow run completed ─────────────────────────────────
    if run_id:
        try:
            # Convert verdict_json to a verdict if available
            from mindflow.agents.types import PanelVerdict

            verdict_obj: PanelVerdict | None = None
            if verdict_json is not None:
                verdict_obj = analysis_dict_to_panel_verdict(
                    verdict_json, source=source,
                )

            result = AnalysisResult(
                verdict=verdict_obj or _empty_verdict(),
                run_id=run_id,
                created_at=datetime.now(UTC),
            )
            run_repo = runtime.workflow_run_repo
            if run_repo is not None:
                await run_repo.update_status(
                    run_id, "completed", result=result,
                )
        except Exception as exc:
            logger.warning("Failed to mark run {} as completed: {}", run_id, exc)
            # Non-fatal: analysis is saved, run can be reconciled later

    # ── 3. Release budget ──────────────────────────────────────────────
    budget_repo = runtime.budget_repo
    if idempotency_key and state.get("budget_reserved") and budget_repo is not None:
        try:
            await budget_repo.release(idempotency_key)
        except Exception as exc:
            logger.warning("Failed to release budget for {}: {}", idempotency_key, exc)

    return {"error": None, "persistence_failed": False}


async def handle_persistence_failure_node(
    state: AnalysisGraphState,
) -> dict[str, Any]:
    """Mark the workflow run as failed when persistence fails.

    This ensures a failed run is never left in a "running" or
    "pending" state — it transitions to "failed" so the retry
    infrastructure can pick it up.
    """
    runtime: AnalysisRunContext = state.get("runtime", AnalysisRunContext())
    run_id = state.get("run_id", "")
    error = state.get("error", "persistence_failed")
    idempotency_key = state.get("idempotency_key", "")

    if run_id:
        try:
            run_repo = runtime.workflow_run_repo
            if run_repo is not None:
                await run_repo.update_status(
                    run_id, "failed", error=error,
                )
        except Exception as exc:
            logger.warning("Failed to mark run {} as failed: {}", run_id, exc)

    # Release budget even on failure so the key can be retried
    budget_repo = runtime.budget_repo
    if idempotency_key and state.get("budget_reserved") and budget_repo is not None:
        try:
            await budget_repo.release(idempotency_key)
        except Exception as exc:
            logger.warning("Failed to release budget on failure: {}", exc)

    return {"persistence_failed": True}


# ═══════════════════════════════════════════════════════════════════════════════
# Conditional routers
# ═══════════════════════════════════════════════════════════════════════════════


def cache_router(state: AnalysisGraphState) -> str:
    """Route: cache_hit → result_conversion, no_cache → budget_reserve."""
    if state.get("cache_hit", False):
        return "result_conversion"
    return "budget_reserve"


def budget_router(state: AnalysisGraphState) -> str:
    """Route after budget reservation.

    If budget was NOT reserved AND we got a cache hit from the re-check,
    route to result_conversion.  If the competing run does not publish in
    time, continue through the deterministic fallback instead of returning
    an empty verdict.
    """
    if state.get("budget_reserved", False):
        return "evidence_preparation"
    if state.get("cache_hit", False):
        # Re-check found cached result from a prior completed run
        return "result_conversion"
    if state.get("fallback_to_rule_engine", False):
        return "evidence_preparation"
    # Budget taken, no cache — another run owns it, let it finish
    logger.info("Budget already reserved and no cache — exiting")
    return "end"


def crisis_router(state: AnalysisGraphState) -> str:
    """Route: crisis_detected → prepare_fallback, no_crisis → panel_graph."""
    if state.get("crisis_detected", False) or state.get(
        "fallback_to_rule_engine", False
    ):
        return "prepare_fallback_context"
    return "panel_graph"


def panel_result_router(state: AnalysisGraphState) -> str:
    """Route: panel_succeeded → result_conversion, failed → prepare_fallback."""
    if state.get("panel_succeeded", False):
        return "result_conversion"
    return "prepare_fallback_context"


def persistence_router(state: AnalysisGraphState) -> str:
    """Route: persistence_failed → handle_failure, success → END."""
    if state.get("persistence_failed", False):
        return "handle_persistence_failure"
    return END


# ═══════════════════════════════════════════════════════════════════════════════
# AnalysisGraph — top-level composition root
# ═══════════════════════════════════════════════════════════════════════════════


class AnalysisGraph:
    """Top-level workflow composition root.

    Composes the full analysis pipeline as a LangGraph StateGraph,
    integrating the PanelGraph subgraph (Todo 9) and fallback chain
    (Todo 11) with exactly-once terminal persistence.

    Implements ``AnalysisWorkflowPort`` from Todo 2.

    Args:
        analysis_repo: Repository for analysis persistence.
        workflow_run_repo: Repository for workflow run status tracking.
        budget_repo: Repository for atomic budget reservation.
        evidence_builder: Builder for evidence bundles.
        crisis_detector: Pre-LLM crisis keyword detector.
        panel_graph: Compiled expert panel subgraph (optional).
        deepseek_client: L1 LLM client.
        ollama_base_url: L2 Ollama endpoint.
        ollama_model: L2 Ollama model name.
        rule_engine: L3 deterministic rule engine.
        timezone: Business timezone.
    """

    def __init__(  # noqa: PLR0913 — service wiring naturally needs many args
        self,
        analysis_repo: Any,
        workflow_run_repo: WorkflowRunStorePort,
        budget_repo: BudgetReservationPort,
        evidence_builder: Any,
        crisis_detector: Any,
        panel_graph: PanelGraph | None = None,
        deepseek_client: Any = None,
        ollama_base_url: str | None = None,
        ollama_model: str = "qwen3:8b",
        rule_engine: Any = None,
        timezone: str = "local",
        checkpointer: Any = None,
    ) -> None:
        from mindflow.domain.procrastination import RuleEngine as _RuleEngine

        self._analysis_repo = analysis_repo
        self._workflow_run_repo = workflow_run_repo
        self._budget_repo = budget_repo
        self._evidence_builder = evidence_builder
        self._crisis_detector = crisis_detector
        self._panel_graph = panel_graph
        self._deepseek_client = deepseek_client
        self._ollama_base_url = ollama_base_url
        self._ollama_model = ollama_model
        self._rule_engine = rule_engine or _RuleEngine()
        self._timezone = timezone
        # Optional LangGraph checkpointer (ApplicationCheckpointer or None).
        # When provided, the compiled graph persists state after every node
        # transition so a crash can resume instead of re-running paid LLM
        # calls (architecture plan item A). The object exposes ``.saver``.
        self._checkpointer = checkpointer
        self._compiled: CompiledStateGraph[Any, Any, Any, Any] | None = None

    # ── AnalysisWorkflowPort implementation ────────────────────────────

    async def run_analysis(self, request: AnalysisRequest) -> AnalysisResult:
        """Run the full analysis workflow for the given request.

        Implements ``AnalysisWorkflowPort.run_analysis``.
        """
        # Build idempotency key if not provided
        idempotency_key = request.idempotency_key or build_idempotency_key(
            origin=request.origin,
            user_id=request.user_id,
            target_date=request.target_date,
            analysis_kind=request.analysis_kind,
        )

        # Create workflow run
        from mindflow.ports import WorkflowRunRequest

        run_request = WorkflowRunRequest(
            user_id=request.user_id,
            target_date=request.target_date,
            force_refresh=request.force or request.retry_if_degraded,
            origin=request.origin,
            idempotency_key=idempotency_key,
        )
        run_id = await self._workflow_run_repo.save_run(run_request)

        # Build runtime context
        runtime = AnalysisRunContext(
            analysis_repo=self._analysis_repo,
            workflow_run_repo=self._workflow_run_repo,
            budget_repo=self._budget_repo,
            evidence_builder=self._evidence_builder,
            crisis_detector=self._crisis_detector,
            panel_graph=self._panel_graph,
            deepseek_client=self._deepseek_client,
            ollama_base_url=self._ollama_base_url,
            ollama_model=self._ollama_model,
            rule_engine=self._rule_engine,
            timezone=self._timezone,
        )

        # Build initial state
        initial_state: AnalysisGraphState = {
            "user_id": request.user_id,
            "target_date": request.target_date,
            "analysis_kind": request.analysis_kind,
            "origin": request.origin,
            "force": request.force or request.retry_if_degraded,
            "idempotency_key": idempotency_key,
            "runtime": runtime,
            "run_id": run_id,
            "budget_reserved": False,
            "cache_hit": False,
            "cached_result": None,
            "bundle_json": "",
            "valid_metrics": (),
            "crisis_detected": False,
            "crisis_response_text": "",
            "events_domain": [],
            "panel_succeeded": False,
            "panel_unavailable_reason": "",
            "fallback_to_rule_engine": False,
            "summary_json": "",
            "behavior_summary": None,
            "current_result": None,
            "assessment": None,
            "source": "",
            "degradation_path": [],
            "degraded": False,
            "verdict_json": None,
            "error": None,
            "persistence_failed": False,
        }

        # Mark the run as running before executing the graph so stale-run
        # recovery can observe it if the process dies mid-flight.
        try:
            await self._workflow_run_repo.update_status(run_id, "running")
        except Exception as exc:
            logger.warning("Failed to mark run {} as running: {}", run_id, exc)

        # Run the graph. With a checkpointer wired, use a stable thread_id so a
        # crash mid-run can resume from the last checkpoint (LLM cost savings).
        graph = self._get_compiled_graph()
        invoke_config: RunnableConfig | None = None
        if self._checkpointer is not None:
            invoke_config = {
                "configurable": {
                    "thread_id": (
                        f"analysis_{request.user_id}_"
                        f"{request.target_date.isoformat()}"
                    )
                }
            }
        try:
            final_state = await graph.ainvoke(
                initial_state,
                config=invoke_config,
            )
        except NoActivityDataError:
            with contextlib.suppress(Exception):
                await self._workflow_run_repo.update_status(
                    run_id, "failed", error="暂无活动数据，请先开始采集",
                )
            if runtime.budget_owned and self._budget_repo is not None:
                try:
                    await self._budget_repo.release(idempotency_key)
                except Exception as exc:
                    logger.warning(
                        "Failed to release budget for {}: {}", idempotency_key, exc
                    )
            raise
        except Exception as exc:
            logger.error("AnalysisGraph invocation failed: {}", exc)
            # Mark run as failed
            with contextlib.suppress(Exception):
                await self._workflow_run_repo.update_status(
                    run_id, "failed", error=str(exc),
                )
            # Release the reservation ONLY if this run actually won it — a
            # failing non-owner concurrent run must never delete the winner's
            # reservation.
            if runtime.budget_owned and self._budget_repo is not None:
                try:
                    await self._budget_repo.release(idempotency_key)
                except Exception as exc:
                    logger.warning(
                        "Failed to release budget for {}: {}", idempotency_key, exc
                    )
            # Return an empty verdict
            return AnalysisResult(
                verdict=_empty_verdict(),
                run_id=run_id,
                created_at=datetime.now(UTC),
            )

        if isinstance(final_state, dict) and final_state.get("persistence_failed"):
            raise RuntimeError(final_state.get("error") or "analysis persistence failed")

        # Convert final state to AnalysisResult
        verdict_json = final_state.get("verdict_json") if isinstance(final_state, dict) else None
        if verdict_json is not None:
            verdict = analysis_dict_to_panel_verdict(
                verdict_json,
                source=final_state.get("source", "rule_engine"),
            )
        else:
            # Fallback: use assessment directly
            assessment = final_state.get("assessment") if isinstance(final_state, dict) else None
            if assessment is not None:
                verdict = analysis_dict_to_panel_verdict(
                    assessment,
                    source=final_state.get("source", "rule_engine"),
                )
            else:
                verdict = _empty_verdict()

        return AnalysisResult(
            verdict=verdict,
            run_id=run_id,
            created_at=datetime.now(UTC),
        )

    # ── Graph construction ─────────────────────────────────────────────

    def _build_compiled_graph(self) -> CompiledStateGraph[Any, Any, Any, Any]:
        """Build and compile the LangGraph StateGraph once.

        Graph topology:

        START → cache_idempotency_check
            ├── cache_hit → result_conversion → terminal_persistence
            │                → [ok→END | fail→handle_failure→END]
            └── no_cache → budget_reserve
                ├── budget_reserved → evidence_preparation → crisis_gate
                │   ├── crisis_detected → prepare_fallback_context
                │   └── no_crisis → panel_graph
                │       ├── success → result_conversion
                │       │           → terminal_persistence
                │       │           → [ok→END | fail→handle_failure→END]
                │       └── failure → prepare_fallback_context
                └── budget_failed → [cache_recheck_hit
                                    → result_conversion
                                    → terminal_persistence | END]

        The fallback chain (prepare_fallback_context → fallback_chain) runs after:
          - Crisis detection (crisis_detected=True)
          - Panel failure (panel_succeeded=False)

        The fallback chain is a loop: prepare_fallback_context → fallback_chain
        The fallback_chain node manages the L1→L2→L3 degradation internally.
        """
        graph = StateGraph(AnalysisGraphState)

        # ── Nodes ───────────────────────────────────────────────────
        graph.add_node("cache_idempotency_check", cache_idempotency_check_node)
        graph.add_node("budget_reserve", budget_reserve_node)
        graph.add_node("evidence_preparation", evidence_preparation_node)
        graph.add_node("crisis_gate", crisis_gate_node)
        graph.add_node("panel_graph", panel_graph_node)
        graph.add_node("prepare_fallback_context", prepare_fallback_context_node)
        graph.add_node("fallback_chain", _fallback_chain_node)
        graph.add_node("result_conversion", result_conversion_node)
        graph.add_node("terminal_persistence", terminal_persistence_node)
        graph.add_node("handle_persistence_failure", handle_persistence_failure_node)

        # ── Wiring ─────────────────────────────────────────────────
        graph.set_entry_point("cache_idempotency_check")

        # Cache check → result_conversion (hit) or budget_reserve (miss)
        graph.add_conditional_edges(
            "cache_idempotency_check",
            cache_router,
            {
                "result_conversion": "result_conversion",
                "budget_reserve": "budget_reserve",
            },
        )

        # Budget reserve → evidence_preparation (won) or result_conversion (re-check hit) or END
        graph.add_conditional_edges(
            "budget_reserve",
            budget_router,
            {
                "evidence_preparation": "evidence_preparation",
                "result_conversion": "result_conversion",
                "end": END,
            },
        )

        # Evidence preparation → crisis gate
        graph.add_edge("evidence_preparation", "crisis_gate")

        # Crisis gate → prepare_fallback (crisis) or panel_graph (no crisis)
        graph.add_conditional_edges(
            "crisis_gate",
            crisis_router,
            {
                "prepare_fallback_context": "prepare_fallback_context",
                "panel_graph": "panel_graph",
            },
        )

        # Panel graph → result_conversion (success) or prepare_fallback (failure)
        graph.add_conditional_edges(
            "panel_graph",
            panel_result_router,
            {
                "result_conversion": "result_conversion",
                "prepare_fallback_context": "prepare_fallback_context",
            },
        )

        # Prepare fallback context → fallback chain
        graph.add_edge("prepare_fallback_context", "fallback_chain")

        # Fallback chain → result_conversion (always — L3 guarantees output)
        graph.add_edge("fallback_chain", "result_conversion")

        # Result conversion → terminal persistence
        graph.add_edge("result_conversion", "terminal_persistence")

        # Terminal persistence → END (success) or handle_failure
        graph.add_conditional_edges(
            "terminal_persistence",
            persistence_router,
            {
                "handle_persistence_failure": "handle_persistence_failure",
                END: END,
            },
        )

        # Handle persistence failure → END
        graph.add_edge("handle_persistence_failure", END)

        # Compile with the optional checkpointer (architecture plan item A).
        # When checkpointing_enabled is True the ApplicationCheckpointer
        # provides a SQLite-backed saver; None keeps in-memory behaviour
        # identical to before.
        self._compiled = graph.compile(
            checkpointer=(
                self._checkpointer.saver
                if self._checkpointer is not None
                else None
            )
        )
        return self._compiled

    def _get_compiled_graph(self) -> CompiledStateGraph[Any, Any, Any, Any]:
        """Return the compiled graph, building it on first access (lazy)."""
        if self._compiled is None:
            self._compiled = self._build_compiled_graph()
        return self._compiled


# ═══════════════════════════════════════════════════════════════════════════════
# Fallback chain node — runs L1→L2→L3 degradation inside a single graph node
# ═══════════════════════════════════════════════════════════════════════════════


async def _fallback_chain_node(state: AnalysisGraphState) -> dict[str, Any]:
    """Run the L1→L2→L3 degradation chain within a single graph node.

    Uses the typed fallback nodes from ``graph/fallback_nodes.py``
    (Todo 11).  L3 (rule_engine) always succeeds, so this node never
    produces an ``error`` on the final state.
    """

    runtime: AnalysisRunContext = state.get("runtime", AnalysisRunContext())
    summary_json: str = state.get("summary_json", "")
    behavior_summary = state.get("behavior_summary")
    crisis_detected = state.get("crisis_detected", False)
    fallback_to_rule_engine = state.get("fallback_to_rule_engine", False)
    crisis_response_text = state.get("crisis_response_text", "")

    # Build FallbackRunContext from AnalysisRunContext
    fallback_runtime = FallbackRunContext(
        analysis_repo=runtime.analysis_repo,
        deepseek_client=runtime.deepseek_client,
        crisis_detector=runtime.crisis_detector,
        ollama_base_url=runtime.ollama_base_url,
        ollama_model=runtime.ollama_model,
        rule_engine=runtime.rule_engine,
    )

    fb_state: FallbackState = {
        "user_id": state["user_id"],
        "target_date": state["target_date"],
        "analysis_kind": state.get("analysis_kind", "daily_attribution"),
        "force": False,
        "runtime": fallback_runtime,
        "summary_json": summary_json,
        "behavior_summary": behavior_summary,
        "cache_hit": False,
        "crisis_detected": crisis_detected,
        "crisis_response_text": crisis_response_text,
        "current_result": None,
        "assessment": None,
        "source": "",
        "degradation_path": [],
        "degraded": False,
        "persistence_intent": "save",
        "error": None,
    }

    if fallback_to_rule_engine:
        fb_state["degradation_path"] = list(state.get("degradation_path", []) or [])
        re_update = await rule_engine_node(fb_state)
        return {
            "assessment": re_update.get("assessment", {}),
            "source": re_update.get("source", "rule_engine"),
            "degraded": re_update.get("degraded", True),
            "degradation_path": re_update.get("degradation_path", ["rule_engine"]),
            "error": None,
        }

    # ── Crisis path: go straight to rule_engine ───────────────────────
    if crisis_detected:
        re_update = await rule_engine_node(fb_state)
        return {
            "assessment": re_update.get("assessment", {}),
            "source": re_update.get("source", "rule_engine"),
            "degraded": re_update.get("degraded", False),
            "degradation_path": re_update.get("degradation_path", ["crisis→rule_engine"]),
            "error": None,
        }

    # ── L1: DeepSeek ──────────────────────────────────────────────────
    ds_update = await single_expert_node(fb_state)
    if ds_update.get("current_result"):
        return {
            "assessment": ds_update["current_result"],
            "source": ds_update.get("source", "deepseek"),
            "degraded": ds_update.get("degraded", False),
            "degradation_path": ds_update.get("degradation_path", ["deepseek"]),
            "error": None,
        }

    # ── L2: Ollama ────────────────────────────────────────────────────
    fb_state.update(ds_update)  # type: ignore[typeddict-item]
    os_update = await ollama_node(fb_state)
    if os_update.get("current_result"):
        return {
            "assessment": os_update["current_result"],
            "source": os_update.get("source", "ollama"),
            "degraded": os_update.get("degraded", True),
            "degradation_path": os_update.get("degradation_path", ["deepseek", "ollama"]),
            "error": None,
        }

    # ── L3: RuleEngine (never fails) ──────────────────────────────────
    fb_state.update(os_update)  # type: ignore[typeddict-item]
    re_update = await rule_engine_node(fb_state)
    return {
        "assessment": re_update.get("assessment", {}),
        "source": re_update.get("source", "rule_engine"),
        "degraded": re_update.get("degraded", True),
        "degradation_path": re_update.get("degradation_path", []),
        "error": None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _empty_verdict() -> Any:
    """Return an empty PanelVerdict for error/unavailable cases."""
    from mindflow.agents.types import PanelVerdict
    from mindflow.domain.procrastination import ProcrastinationType

    return PanelVerdict(
        types=(ProcrastinationType.TASK_AVERSION,),
        confidence={ProcrastinationType.TASK_AVERSION: 0.5},
        recommended_technique=None,
        rationale="分析暂时不可用",
        dissent=(),
        transcript=(),
        escalated=False,
        call_count=0,
        source="rule_engine",
    )
