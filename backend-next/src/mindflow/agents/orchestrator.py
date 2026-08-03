"""PanelOrchestrator — the expert panel deliberation kernel.

Implements the full orchestration flow from 07-agent-upgrade-design.md §2 and §4,
now using LangGraph StateGraph internally:

```
快速通道（默认 ~6 次调用）: analyst → 归因×3并行 → [冲突检测] → moderator → critic
冲突升级（+3 次）: 每位归因专家收到其他两位完整论证 → 反驳修正 → moderator → critic
```

On unrecoverable failure, raises ``PanelUnavailableError`` for the caller (G003)
to catch and fall through the four-layer degradation chain:
  panel → single_expert (existing llm_service) → ollama → rule_engine
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypedDict, cast

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from loguru import logger

from mindflow.agents.conflict import ConflictReport, detect_conflict
from mindflow.agents.disagreement import (
    DisagreementSummary,
    analyze_disagreement,
    compute_rebuttal_delta,
)
from mindflow.agents.experts import (
    ANALYST,
    ATTRIBUTION_EXPERTS,
    CRITIC,
    MODERATOR,
    ExpertDef,
)
from mindflow.agents.llm_gateway import PanelLLMGateway
from mindflow.agents.schemas import (
    AnalystOutput,
    AttributionOutput,
    CriticOutput,
    ModeratorOutput,
)
from mindflow.agents.types import (
    CriticResult,
    ExpertOpinion,
    PanelBudgetExceededError,
    PanelUnavailableError,
    PanelVerdict,
    TranscriptEntry,
    _contains_forbidden_words,
)
from mindflow.domain.evidence import EvidenceBundle, to_prompt_json
from mindflow.domain.evidence_facts import build_evidence_catalog, evidence_catalog_ids
from mindflow.domain.procrastination import CBTTechnique, ProcrastinationType

# ── Parsing helpers ────────────────────────────────────────────────────────────


def _strip_markdown_fences(raw: str) -> str:
    """Strip optional Markdown code-fence markers from *raw*."""
    text = raw.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3].strip()
    return text


def _safe_parse_json(raw: str, context: str) -> dict[str, Any] | None:
    """Parse *raw* as JSON, returning None on failure.

    Strips Markdown fence markers if present.
    """
    text = _strip_markdown_fences(raw)
    try:
        result: dict[str, Any] = json.loads(text)
        return result
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse failed for {}: {}", context, exc)
        return None


def _parse_with_pydantic(
    raw: str,
    schema_class: type[AnalystOutput | AttributionOutput | ModeratorOutput | CriticOutput],
    context: str,
) -> AnalystOutput | AttributionOutput | ModeratorOutput | CriticOutput | None:
    """Parse *raw* LLM output with a Pydantic schema, returning None on failure."""
    text = _strip_markdown_fences(raw)
    try:
        return schema_class.model_validate_json(text)
    except Exception as exc:
        logger.warning("Pydantic parse failed for {}: {}", context, exc)
        return None


def _make_skipped_opinion(expert: ExpertDef, raw: str = "") -> ExpertOpinion:
    """Return a skipped ``ExpertOpinion`` for *expert*."""
    return ExpertOpinion(
        role=expert.role,
        perspective=expert.perspective,
        attribution_types=(),
        confidence={},
        evidence_citations=(),
        argument="",
        raw_json=raw,
        skipped=True,
    )


_CITATION_PATTERN = re.compile(r"\[证据[:：]\s*([A-Za-z0-9_.]+)\s*\]")


def validate_citations(
    opinion: ExpertOpinion,
    valid_metrics: frozenset[str],
) -> tuple[str, ...]:
    """Code-level citation validation — never trust the LLM critic alone.

    Extracts every ``[证据: metric]`` reference from the argument plus the
    structured ``evidence_citations`` field, and returns the subset that does
    NOT exist in the bundle's citation IDs.

    Supports controlled alias resolution: bare names that uniquely match
    one canonical ID are auto-normalized (Codex recommendation).
    """
    cited: set[str] = set(opinion.evidence_citations)
    cited.update(_CITATION_PATTERN.findall(opinion.argument))

    # Build bare-name → canonical-ID lookup for alias resolution
    bare_to_canonical: dict[str, str | None] = {}
    for vid in valid_metrics:
        if "." in vid:
            bare = vid.rsplit(".", 1)[-1]
            if bare not in bare_to_canonical:
                bare_to_canonical[bare] = vid
            else:
                bare_to_canonical[bare] = None  # ambiguous: multiple matches

    # Resolve aliases: bare name with exactly one match → canonicalize
    resolved: set[str] = set()
    unresolved: set[str] = set()
    for cite in cited:
        if cite in valid_metrics:
            resolved.add(cite)
        elif cite in bare_to_canonical and bare_to_canonical[cite] is not None:
            resolved.add(bare_to_canonical[cite])
        else:
            unresolved.add(cite)

    return tuple(sorted(unresolved))


def validate_verdict_schema(verdict: dict[str, Any]) -> list[str]:
    """Deterministically validate a moderator verdict before the critic call."""
    issues: list[str] = []
    verdict = normalize_verdict_types(verdict)
    valid_types = {t.value for t in ProcrastinationType}
    valid_techniques = {t.value for t in CBTTechnique}

    types_raw = verdict.get("types")
    if not isinstance(types_raw, list):
        issues.append("types 必须为数组")
    else:
        if len(types_raw) > 3:
            issues.append("types 最多 3 个")
        for t in types_raw:
            if str(t) not in valid_types:
                issues.append(f"未知拖延类型: {t}")

    confidence = verdict.get("confidence")
    if not isinstance(confidence, dict):
        issues.append("confidence 必须为对象")
    else:
        for key, value in confidence.items():
            if str(key) not in valid_types:
                issues.append(f"置信度键不是合法类型: {key}")
            try:
                number = float(value)
                if not 0.0 <= number <= 1.0:
                    issues.append(f"置信度越界: {key}={value}")
            except (TypeError, ValueError):
                issues.append(f"置信度不是数字: {key}={value}")

    technique = verdict.get("recommended_technique")
    if technique is not None and str(technique) not in valid_techniques:
        issues.append(f"未知 CBT 技术: {technique}")
    return issues


TYPE_ALIASES: dict[str, str] = {
    "决策性拖延": "decisional",
    "任务价值感知不足型拖延": "task_aversion",
    "冲动型拖延": "impulsivity",
    "完美主义拖延": "perfectionism",
    "情绪调节型拖延": "emotional_regulation",
}


def normalize_verdict_types(verdict: dict[str, Any]) -> dict[str, Any]:
    """Map Chinese/verbose moderator labels back to canonical enum values."""
    normalized = dict(verdict)
    raw_types = normalized.get("types")
    if isinstance(raw_types, list):
        normalized["types"] = [TYPE_ALIASES.get(str(t), str(t)) for t in raw_types]
    raw_conf = normalized.get("confidence")
    if isinstance(raw_conf, dict):
        normalized["confidence"] = {
            TYPE_ALIASES.get(str(k), str(k)): v for k, v in raw_conf.items()
        }
    return normalized


def _parse_expert_opinion(
    raw: str,
    expert: ExpertDef,
    skipped: bool = False,
    valid_metrics: frozenset[str] | None = None,
) -> ExpertOpinion:
    """Parse an expert's raw LLM response into ``ExpertOpinion``.

    Uses ``AttributionOutput.model_validate_json`` for type-safe parsing.
    If JSON parsing fails, returns a skipped opinion (graceful degradation).
    Forbidden-word and hallucinated-citation checks are still enforced at
    the code level (semantic validation that Pydantic cannot express).
    """
    if skipped:
        return _make_skipped_opinion(expert, raw)

    parsed = _parse_with_pydantic(raw, AttributionOutput, expert.role)
    if parsed is None:
        return _make_skipped_opinion(expert, raw)

    attribution_types = tuple(parsed.attribution_types)
    confidence = {
        k: float(v) for k, v in parsed.confidence.items()
        if isinstance(v, (int, float))
    }
    evidence_citations = tuple(parsed.evidence_citations)
    argument = parsed.argument

    # Check forbidden words
    forbidden = _contains_forbidden_words(argument)
    if forbidden:
        logger.warning("Forbidden word {!r} in {} opinion — skipping", forbidden, expert.role)
        return _make_skipped_opinion(expert, raw)

    opinion = ExpertOpinion(
        role=expert.role,
        perspective=expert.perspective,
        attribution_types=attribution_types,
        confidence=confidence,
        evidence_citations=evidence_citations,
        argument=argument,
        raw_json=raw,
    )

    # Code-enforced citation check (review P1): hallucinated metric references
    # disqualify the opinion regardless of what the LLM critic later says.
    if valid_metrics is not None:
        bogus = validate_citations(opinion, valid_metrics)
        if bogus:
            logger.warning(
                "Hallucinated citations {} in {} opinion — skipping",
                bogus,
                expert.role,
            )
            return _make_skipped_opinion(expert, raw)

    return opinion


def _parse_analyst_opinion(
    raw: str,
    expert: ExpertDef,
) -> ExpertOpinion:
    """Parse analyst output using Pydantic ``AnalystOutput``.

    The analyst outputs ``patterns`` / ``anomalies`` / ``top_concerns``
    rather than ``attribution_types`` / ``confidence``. We map those
    into the generic ``ExpertOpinion`` shape.
    """
    parsed = _parse_with_pydantic(raw, AnalystOutput, expert.role)
    if parsed is None:
        return _make_skipped_opinion(expert, raw)

    evidence_citations = tuple(parsed.evidence_citations)

    # Build argument text from patterns + anomalies
    parts: list[str] = []
    for p in parsed.patterns:
        if isinstance(p, dict):
            parts.append(f"[{p.get('severity', 'info')}] {p.get('description', '')}")
    for a in parsed.anomalies:
        if isinstance(a, dict):
            parts.append(f"异常-{a.get('metric', '')}: {a.get('detail', '')}")
    argument = "\n".join(parts) if parts else ""

    # Check forbidden words
    forbidden = _contains_forbidden_words(argument)
    if forbidden:
        logger.warning("Forbidden word {!r} in analyst opinion — skipping", forbidden)
        return _make_skipped_opinion(expert, raw)

    return ExpertOpinion(
        role=expert.role,
        perspective=expert.perspective,
        attribution_types=(),
        confidence={},
        evidence_citations=evidence_citations,
        argument=argument,
        raw_json=raw,
    )


def _parse_verdict(raw: str) -> dict[str, Any] | None:
    """Parse the moderator's JSON output into a raw dict.

    Uses ``ModeratorOutput.model_validate_json`` for type-safe parsing
    then converts back to dict for downstream compatibility.
    Returns None on parse failure.
    """
    parsed = _parse_with_pydantic(raw, ModeratorOutput, "moderator")
    if parsed is None:
        return None
    result: dict[str, Any] = {
        "types": parsed.types,
        "confidence": parsed.confidence,
        "recommended_technique": parsed.recommended_technique,
        "rationale": parsed.rationale,
        "dissent": parsed.dissent,
        "insufficient_data": parsed.insufficient_data,
        "uncertainty": parsed.uncertainty,
        "evidence_gaps": list(parsed.evidence_gaps),
    }
    return normalize_verdict_types(result)


def _parse_critic(raw: str) -> CriticResult:
    """Parse the critic's JSON output using Pydantic for type safety.

    Uses ``CriticOutput.model_validate_json`` which correctly parses
    JSON ``true``/``false`` — fixing the previous bug where
    ``bool("false") == True`` silently approved rejected verdicts.

    Returns a safe default (not approved, with an explanation) on failure.
    """
    text = _strip_markdown_fences(raw)

    try:
        parsed = CriticOutput.model_validate_json(text)
    except Exception:
        logger.warning("Critic JSON parse failed")
        return CriticResult(approved=False, issues=("批评家输出解析失败",))

    return CriticResult(
        approved=parsed.approved,
        issues=tuple(parsed.issues) if parsed.issues else (),
    )


def _verdict_dict_to_panel_verdict(
    data: dict[str, Any],
    escalated: bool,
    transcript: tuple[TranscriptEntry, ...],
    call_count: int,
) -> PanelVerdict:
    """Convert a moderator's JSON dict into a ``PanelVerdict``.

    Delegates to the shared :func:`mindflow.services.panel_service.analysis_dict_to_panel_verdict`.
    The lazy import avoids a circular dependency (``panel_service`` imports ``PanelOrchestrator``).
    """
    from mindflow.services.panel_service import analysis_dict_to_panel_verdict

    return analysis_dict_to_panel_verdict(
        data,
        escalated=escalated,
        transcript=transcript,
        call_count=call_count,
        source="panel",
    )


# ── Prompt builders ────────────────────────────────────────────────────────────


def _build_moderator_user_prompt(
    bundle_json: str,
    analyst: ExpertOpinion,
    attribution_opinions: Sequence[ExpertOpinion],
    conflict: ConflictReport,
    disagreement_summary: Any | None = None,
) -> str:
    """Build the moderator's user prompt with all expert opinions."""
    parts: list[str] = [
        "## 用户行为数据",
        bundle_json,
        "",
        "## 数据分析师报告",
        f"角色：{analyst.role}（{analyst.perspective}）",
        analyst.argument or "（无输出）",
        "",
        "## 归因专家意见",
    ]

    for i, op in enumerate(attribution_opinions):
        status = "（已跳过）" if op.skipped else ""
        parts.extend([
            f"### 专家{i + 1}：{op.role}（{op.perspective}）{status}",
            op.argument or "（无输出）",
            f"证据引用：{', '.join(op.evidence_citations) if op.evidence_citations else '无'}",
            "",
        ])

    if conflict.has_conflict:
        parts.extend([
            "## 冲突检测报告",
            conflict.details,
            "",
        ])

    if disagreement_summary is not None:
        parts.extend([
            "## 共识强度",
            f"agreement_strength={disagreement_summary.agreement_strength:.3f}, stability={disagreement_summary.stability}",
            "共识强度低时请降低置信度，或设置 insufficient_data=true。",
            "",
        ])


    return "\n".join(parts)


def _build_rebuttal_prompt(
    bundle_json: str,
    all_opinions: Sequence[ExpertOpinion],
    target_index: int,
) -> str:
    """Build a rebuttal prompt for one attribution expert,
    showing the other two experts' arguments.
    """
    target = all_opinions[target_index]
    others = [o for i, o in enumerate(all_opinions) if i != target_index]

    parts: list[str] = [
        "## 用户行为数据",
        bundle_json,
        "",
        f"## 你的原始分析（{target.role}）",
        target.argument or "（无输出）",
        "",
        "## 其他专家的分析——请阅读并给出回应",
    ]

    for _i, other in enumerate(others):
        parts.extend([
            f"### 专家：{other.role}（{other.perspective}）",
            other.argument or "（无输出）",
            "他们认为的类型："
            + (", ".join(other.attribution_types) if other.attribution_types else "未指定"),
            "",
        ])

    parts.append(
        "## 你的任务\n"
        "阅读其他两位专家的分析。请决定：\n"
        "1. 你是否同意他们的部分观点？\n"
        "2. 看完他们的分析后，你是否要修正自己的判断？\n"
        "3. 如果不同意，请用证据和数据反驳。\n\n"
        "输出与前一次相同的 JSON 格式"
        "（attribution_types + confidence + argument + evidence_citations）。"
    )

    return "\n".join(parts)


def _build_critic_user_prompt(
    bundle_json: str,
    verdict: PanelVerdict,
    all_opinions: Sequence[ExpertOpinion],
    valid_metrics: frozenset[str],
) -> str:
    """Build the critic's user prompt with verdict + opinions + valid metrics."""
    metrics_str = ", ".join(sorted(valid_metrics)) if valid_metrics else "（无）"
    # evidence_catalog is embedded in bundle_json by to_prompt_json()

    dissent_str = "\n".join(verdict.dissent) if verdict.dissent else "（无分歧）"

    opinions_lines: list[str] = []
    for op in all_opinions:
        status = "（已跳过）" if op.skipped else ""
        citations = ", ".join(op.evidence_citations) if op.evidence_citations else "无"
        opinions_lines.append(f"- {op.role}{status}：引用[{citations}]")

    op_text = "\n".join(opinions_lines)

    return (
        f"## 用户行为数据\n{bundle_json}\n\n"
        f"## 合法指标清单\n{metrics_str}\n\n"
        f"## 专家意见摘要\n{op_text}\n\n"
        f"## 主持人裁决\n"
        f"类型：{[str(t) for t in verdict.types]}\n"
        f"置信度：{ {str(k): v for k, v in verdict.confidence.items()} }\n"
        f"推荐技术：{verdict.recommended_technique}\n"
        f"理由：{verdict.rationale}\n"
        f"分歧：{dissent_str}\n\n"
        "请检查：\n"
        "1. 每个 [证据: X] 引用中的 X 是否在合法指标清单中？\n"
        "2. 是否有逻辑跳跃或过度诊断？\n"
        "3. 是否有禁词？\n"
        "4. 置信度是否与证据强度匹配？"
    )


def _build_moderator_redo_prompt(
    bundle_json: str,
    analyst: ExpertOpinion,
    attribution_opinions: Sequence[ExpertOpinion],
    conflict: ConflictReport,
    critic_issues: tuple[str, ...],
) -> str:
    """Build a moderator re-verdict prompt after critic rejection."""
    base = _build_moderator_user_prompt(bundle_json, analyst, attribution_opinions, conflict)
    issues_text = "\n".join(f"- {issue}" for issue in critic_issues)
    return (
        f"{base}\n\n"
        f"## 批评家打回意见\n"
        f"以下问题需要修正，请重新裁决：\n{issues_text}\n\n"
        f"请输出修正后的裁决 JSON。"
    )


# ── Transcript helpers ─────────────────────────────────────────────────────────


def _opinion_summary(opinion: ExpertOpinion) -> str:
    """Produce a short transcript summary for an expert opinion."""
    if opinion.skipped:
        return "（已跳过）"
    types_str = ", ".join(opinion.attribution_types) if opinion.attribution_types else "未归因"
    return f"类型={types_str}, 证据={len(opinion.evidence_citations)}项"


def _verdict_summary(verdict: dict[str, Any]) -> str:
    """Produce a short transcript summary for a moderator verdict."""
    types = verdict.get("types", [])
    types_str = (
        ", ".join(str(t) for t in types)
        if isinstance(types, list)
        else str(types)
    )
    return f"裁决类型={types_str}"


def _critic_summary(result: CriticResult) -> str:
    """Produce a short transcript summary for a critic result."""
    if result.approved:
        return "通过"
    return f"打回：{'；'.join(result.issues[:2])}"


# ── LangGraph State Schema ───────────────────────────────────────────────────


class PanelState(TypedDict):  # noqa: UP035 — TypedDict with `from __future__ import annotations`
    """State flowing through the LangGraph deliberation graph.

    All fields are required per the TypedDict contract; None-valued fields
    indicate data not yet produced by the corresponding graph node.
    """

    bundle_json: str
    valid_metrics: frozenset[str]
    analyst_opinion: ExpertOpinion | None
    attribution_opinions: list[ExpertOpinion]
    conflict_report: ConflictReport | None
    escalated: bool
    moderator_verdict: dict[str, Any] | None
    critic_result: CriticResult | None
    critic_retries: int
    moderator_redo_count: int
    call_count: int
    transcript: list[TranscriptEntry]
    disagreement_summary: DisagreementSummary | None
    rebuttal_delta: object | None  # RebuttalDelta — lazy import to avoid circular


@dataclass
class _PanelRunContext:
    """Mutable state owned by exactly one panel invocation."""

    call_count: int = 0
    transcript: list[TranscriptEntry] = field(default_factory=list)
    budget_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# Context variable to carry the mutable per-invocation runtime through the
# LangGraph StateGraph without including it in the checkpointable state.
# Set by ``_run_graph`` before ``ainvoke`` and read by all graph nodes.
_PANEL_RUNTIME: contextvars.ContextVar[_PanelRunContext] = contextvars.ContextVar(
    "_PANEL_RUNTIME",
)


# ═══════════════════════════════════════════════════════════════════════════════
# PanelOrchestrator
# ═══════════════════════════════════════════════════════════════════════════════


class PanelOrchestrator:
    """Expert panel deliberation orchestrator — uses LangGraph StateGraph internally.

    Manages the full expert panel lifecycle: calling experts, detecting conflicts,
    synthesising verdicts, and validating via the critic.

    The public API (``run(bundle) -> PanelVerdict``) is unchanged; the internal
    orchestration was migrated from manual async flow to a LangGraph ``StateGraph``.

    Args:
        gateway: The LLM gateway for calling experts.
    """

    def __init__(self, gateway: PanelLLMGateway) -> None:
        self._gateway = gateway
        self._compiled_graph: CompiledStateGraph[Any, Any, Any, Any] | None = None

    # ── Public API ────────────────────────────────────────────────────────

    async def run(self, bundle: EvidenceBundle) -> PanelVerdict:
        """Run a full expert panel deliberation on an evidence bundle.

        Args:
            bundle: The evidence bundle from the ML sensing layer.

        Returns:
            A ``PanelVerdict`` with the deliberation outcome.

        Raises:
            PanelUnavailableError: If the panel cannot produce a verdict
                (caller should fall through to single-expert tier).
            PanelBudgetExceededError: If the panel would exceed 12 LLM calls
                (hard safety guard, should never trigger on normal paths).
        """
        runtime = _PanelRunContext()
        try:
            return await self._run_graph(bundle, runtime)
        except (PanelBudgetExceededError, PanelUnavailableError):
            raise
        except Exception as exc:
            logger.error("Panel orchestrator unexpected error: {}", exc)
            raise PanelUnavailableError(
                reason=f"编排器异常：{exc}",
                call_count=runtime.call_count,
            ) from exc

    # ── LangGraph orchestration ───────────────────────────────────────────

    def _build_compiled_graph(
        self,
    ) -> CompiledStateGraph[Any, Any, Any, Any]:
        """Build and compile the LangGraph StateGraph once.

        Graph nodes: analyst → attribution → conflict_detection
          → [rebuttal (if escalated) | moderator]
          → human_review_interrupt → critic
          → [END (approved) | moderator (retry) | END (exhausted)]
        """
        graph = StateGraph(PanelState)

        # ── Node: analyst ──────────────────────────────────────────────
        async def analyst_node(state: PanelState) -> dict[str, Any]:
            rt = _PANEL_RUNTIME.get()
            logger.info("Panel round 0: Analyst")
            raw = await self._call_with_budget(rt, ANALYST, state["bundle_json"])
            analyst = _parse_analyst_opinion(raw, ANALYST)
            bogus = validate_citations(analyst, state["valid_metrics"])
            if bogus:
                logger.warning("Hallucinated citations {} in analyst — marking", bogus)
                analyst = ExpertOpinion(
                    role=ANALYST.role, perspective=ANALYST.perspective,
                    attribution_types=(), confidence={}, evidence_citations=(),
                    argument="", raw_json=raw, skipped=True,
                )
            rt.transcript.append(TranscriptEntry(role=ANALYST.role, content=_opinion_summary(analyst), round=0))
            return {
                "analyst_opinion": analyst,
                "transcript": list(rt.transcript),
                "call_count": rt.call_count,
            }

        # ── Node: attribution ──────────────────────────────────────────
        async def attribution_node(state: PanelState) -> dict[str, Any]:
            rt = _PANEL_RUNTIME.get()
            logger.info("Panel round 1: Attribution experts (parallel)")

            async def _call_and_parse(exp: ExpertDef) -> ExpertOpinion:
                raw = await self._safe_call_with_budget(rt, exp, state["bundle_json"])
                op = _parse_expert_opinion(raw, exp, valid_metrics=state["valid_metrics"])
                if op.skipped and _contains_forbidden_words(raw):
                    logger.warning("{} triggered forbidden words, retrying once", exp.role)
                    retry_msg = "你的上一条回复包含禁用词汇（诊断、治疗、患者、处方）。请用中文重新输出，严格遵守禁用词规则。"
                    raw2 = await self._safe_call_with_budget(rt, exp, retry_msg)
                    op2 = _parse_expert_opinion(raw2, exp, valid_metrics=state["valid_metrics"])
                    if not op2.skipped:
                        return op2
                    logger.warning("{} retry still failed, using original", exp.role)
                return op

            results = await asyncio.gather(*[_call_and_parse(exp) for exp in ATTRIBUTION_EXPERTS])
            opinions = list(results)
            for op in opinions:
                rt.transcript.append(TranscriptEntry(role=op.role, content=_opinion_summary(op), round=1))

            non_skipped = [o for o in opinions if not o.skipped]
            if len(non_skipped) < 2:
                raise PanelUnavailableError(
                    reason=f"仅{len(non_skipped)}份归因意见有效，需至少2份",
                    call_count=rt.call_count,
                )
            return {
                "attribution_opinions": opinions,
                "transcript": list(rt.transcript),
                "call_count": rt.call_count,
            }

        # ── Node: conflict_detection ───────────────────────────────────
        async def conflict_detection_node(state: PanelState) -> dict[str, Any]:
            logger.info("Conflict detection")
            conflict = detect_conflict(state["attribution_opinions"])
            escalated = conflict.has_conflict
            if escalated:
                logger.info("Conflict detected: {}", conflict.details)
            else:
                logger.info("No conflict among attribution experts")
            ds = analyze_disagreement(state["attribution_opinions"], conflict.details, conflict.max_confidence_gap, rebuttal_delta=None)
            logger.info("Disagreement analytics: agreement={:.3f}, stability={}", ds.agreement_strength, ds.stability)
            return {"conflict_report": conflict, "escalated": escalated, "disagreement_summary": ds}

        # ── Node: rebuttal ─────────────────────────────────────────────
        async def rebuttal_node(state: PanelState) -> dict[str, Any]:
            rt = _PANEL_RUNTIME.get()
            logger.info("Panel round 2a: Attribution rebuttal (parallel)")
            opinions = state["attribution_opinions"]
            prompts = [(ATTRIBUTION_EXPERTS[i], _build_rebuttal_prompt(state["bundle_json"], opinions, i)) for i in range(len(ATTRIBUTION_EXPERTS))]
            responses = await asyncio.gather(*[self._safe_call_with_budget(rt, exp, msg) for exp, msg in prompts])
            new_opinions = []
            for raw, exp in zip(responses, ATTRIBUTION_EXPERTS, strict=True):
                op = _parse_expert_opinion(raw, exp, valid_metrics=state["valid_metrics"])
                if op.skipped and _contains_forbidden_words(raw):
                    logger.warning("{} rebuttal triggered forbidden words, retrying", exp.role)
                    retry_msg = "你的上一条回复包含禁用词汇（诊断、治疗、患者、处方）。请用中文重新输出，严格遵守禁用词规则并回到推理内容。"
                    raw2 = await self._safe_call_with_budget(rt, exp, retry_msg)
                    op2 = _parse_expert_opinion(raw2, exp, valid_metrics=state["valid_metrics"])
                    if not op2.skipped:
                        op = op2
                new_opinions.append(op)
            for op in new_opinions:
                rt.transcript.append(TranscriptEntry(role=op.role, content=_opinion_summary(op), round=2))
            non_skipped = [o for o in new_opinions if not o.skipped]
            if len(non_skipped) < 2:
                raise PanelUnavailableError(reason=f"辩论后仅{len(non_skipped)}份归因意见有效", call_count=rt.call_count)
            delta = compute_rebuttal_delta(opinions, new_opinions)
            logger.info("Rebuttal delta: agreement {:.3f}→{:.3f}, delta={:+.3f}, converged={}", delta.before_agreement, delta.after_agreement, delta.agreement_delta, delta.converged)
            return {"attribution_opinions": new_opinions, "transcript": list(rt.transcript), "call_count": rt.call_count, "rebuttal_delta": delta}

        # ── Node: moderator ────────────────────────────────────────────
        async def moderator_node(state: PanelState) -> dict[str, Any]:
            rt = _PANEL_RUNTIME.get()
            is_redo = state["moderator_redo_count"] > 0
            analyst = state["analyst_opinion"]
            conflict = state["conflict_report"]
            assert analyst is not None
            assert conflict is not None
            if is_redo:
                round_num = 4
                prompt = _build_moderator_redo_prompt(state["bundle_json"], analyst, state["attribution_opinions"], conflict, cast(CriticResult, state["critic_result"]).issues)
            else:
                round_num = 2 if not state["escalated"] else 3
                prompt = _build_moderator_user_prompt(state["bundle_json"], analyst, state["attribution_opinions"], conflict)
                prompt = _build_moderator_user_prompt(state["bundle_json"], analyst, state["attribution_opinions"], conflict, state.get("disagreement_summary"))
            logger.info("Panel round {}: Moderator (redo_count={})", round_num, state["moderator_redo_count"])
            raw = await self._call_with_budget(rt, MODERATOR, prompt)
            verdict = _parse_verdict(raw)
            if verdict is None:
                raise PanelUnavailableError(reason="主持人输出解析失败", call_count=rt.call_count)
            rt.transcript.append(TranscriptEntry(role=MODERATOR.role, content=_verdict_summary(verdict), round=round_num))
            return {"moderator_verdict": verdict, "transcript": list(rt.transcript), "call_count": rt.call_count}

        # ── Node: human_review_interrupt ────────────────────────────────
        async def human_review_interrupt_node(state: PanelState) -> dict[str, Any]:
            """Optional human review gate — disabled by default (Todo 10)."""
            from mindflow.config import get_settings

            settings = get_settings()
            if not settings.human_review_enabled:
                return {}
            verdict = state.get("moderator_verdict")
            if verdict is None:
                return {}
            confidence: dict[str, float] = verdict.get("confidence", {})
            min_conf = min(confidence.values()) if confidence else 1.0
            ds = state.get("disagreement_summary")
            agreement_strength: float = float(ds.agreement_strength) if ds is not None else 1.0
            disagreement_strength = 1.0 - agreement_strength
            if min_conf < settings.human_review_confidence_threshold or disagreement_strength > settings.human_review_disagreement_threshold:
                logger.warning("Human review interrupt triggered: min_confidence={:.2f}, disagreement={:.2f}", min_conf, disagreement_strength)
                interrupt({"verdict": verdict, "min_confidence": min_conf, "agreement_strength": agreement_strength})
                logger.info("Human review interrupt resumed")
            return {}

        # ── Node: critic ───────────────────────────────────────────────
        async def critic_node(state: PanelState) -> dict[str, Any]:
            rt = _PANEL_RUNTIME.get()
            base_round = 2 if not state["escalated"] else 3
            round_num = base_round + 1 + state["critic_retries"]
            logger.info("Panel round {}: Critic", round_num)
            pending_verdict = _verdict_dict_to_panel_verdict(cast(dict[str, Any], state["moderator_verdict"]), state["escalated"], tuple(rt.transcript), rt.call_count)
            all_opinions: list[ExpertOpinion] = [cast(ExpertOpinion, state["analyst_opinion"]), *state["attribution_opinions"]]
            prompt = _build_critic_user_prompt(state["bundle_json"], pending_verdict, all_opinions, state["valid_metrics"])
            raw = await self._call_with_budget(rt, CRITIC, prompt)
            result = _parse_critic(raw)
            rt.transcript.append(TranscriptEntry(role=CRITIC.role, content=_critic_summary(result), round=round_num))
            updates: dict[str, Any] = {"critic_result": result, "transcript": list(rt.transcript), "call_count": rt.call_count}
            if not result.approved:
                updates["critic_retries"] = state["critic_retries"] + 1
                updates["moderator_redo_count"] = state["moderator_redo_count"] + 1
            return updates

        # ── Routers ────────────────────────────────────────────────────
        def should_escalate(state: PanelState) -> str:
            return "rebuttal" if state["escalated"] else "moderator"

        def critic_verdict(state: PanelState) -> str:
            if cast(CriticResult, state["critic_result"]).approved:
                return "approved"
            if state["moderator_redo_count"] < 2:
                return "retry"
            return "exhausted"

        # ── Wire graph ──────────────────────────────────────────────────
        graph.add_node("analyst", analyst_node)
        graph.add_node("attribution", attribution_node)
        graph.add_node("conflict_detection", conflict_detection_node)
        graph.add_node("rebuttal", rebuttal_node)
        graph.add_node("moderator", moderator_node)
        graph.add_node("human_review_interrupt", human_review_interrupt_node)
        graph.add_node("critic", critic_node)

        graph.set_entry_point("analyst")
        graph.add_edge("analyst", "attribution")
        graph.add_edge("attribution", "conflict_detection")
        graph.add_conditional_edges("conflict_detection", should_escalate, {"rebuttal": "rebuttal", "moderator": "moderator"})
        graph.add_edge("rebuttal", "moderator")
        graph.add_edge("moderator", "human_review_interrupt")
        graph.add_edge("human_review_interrupt", "critic")
        graph.add_conditional_edges("critic", critic_verdict, {"approved": END, "retry": "moderator", "exhausted": END})

        from mindflow.config import get_settings

        checkpointer = MemorySaver() if get_settings().human_review_enabled else None
        return graph.compile(checkpointer=checkpointer)

    def _get_compiled_graph(self) -> CompiledStateGraph[Any, Any, Any, Any]:
        """Return the compiled graph, building it on first access (lazy)."""
        if self._compiled_graph is None:
            self._compiled_graph = self._build_compiled_graph()
        return self._compiled_graph

    async def _run_graph(
        self,
        bundle: EvidenceBundle,
        runtime: _PanelRunContext,
    ) -> PanelVerdict:
        """Run the compiled LangGraph StateGraph for this session."""

        bundle_json = to_prompt_json(bundle)
        valid_metrics = evidence_catalog_ids(build_evidence_catalog(bundle))

        compiled = self._get_compiled_graph()

        initial: PanelState = {
            "bundle_json": bundle_json,
            "valid_metrics": valid_metrics,
            "analyst_opinion": None,
            "attribution_opinions": (),
            "conflict_report": None,
            "escalated": False,
            "moderator_verdict": None,
            "critic_result": None,
            "critic_retries": 0,
            "moderator_redo_count": 0,
            "call_count": 0,
            "transcript": (),
            "disagreement_summary": None,
            "rebuttal_delta": None,
        }

        # Set runtime in context var so graph nodes can access it without
        # including it in the checkpointable state (avoids msgpack error).
        _PANEL_RUNTIME.set(runtime)

        final = await compiled.ainvoke(initial)
        critic_result = cast(CriticResult, final["critic_result"])
        if not critic_result.approved:
            issues = "；".join(critic_result.issues) or "未提供拒绝原因"
            raise PanelUnavailableError(
                reason=f"批评家复核未通过：{issues}",
                call_count=final["call_count"],
            )

        return _verdict_dict_to_panel_verdict(
            cast(dict[str, Any], final["moderator_verdict"]),
            final["escalated"],
            tuple(final["transcript"]),
            final["call_count"],
        )

    # ── Gateway helpers ───────────────────────────────────────────────────

    async def _call_with_budget(
        self,
        runtime: _PanelRunContext,
        expert: ExpertDef,
        user_message: str,
    ) -> str:
        """Atomic budget check then gateway call.

        Args:
            expert: The expert definition (system prompt + role).
            user_message: The user message content.

        Returns:
            Raw response text from the LLM.

        Raises:
            PanelBudgetExceededError: If budget (12 calls) would be exceeded.
        """
        async with runtime.budget_lock:
            runtime.call_count += 1
            if runtime.call_count > 12:
                raise PanelBudgetExceededError(call_count=runtime.call_count)
        return await self._gateway.complete(
            system=expert.system_prompt,
            user=user_message,
            model=expert.model,
        )

    async def _safe_call_with_budget(
        self,
        runtime: _PanelRunContext,
        expert: ExpertDef,
        user_message: str,
    ) -> str:
        """Like ``_call_with_budget`` but returns empty string on failure.

        Used in parallel batches so a single failed call doesn't abort the group.
        """
        try:
            return await self._call_with_budget(runtime, expert, user_message)
        except PanelBudgetExceededError:
            raise
        except Exception as exc:
            logger.error("Parallel call to {} failed: {}", expert.role, exc)
            return ""
