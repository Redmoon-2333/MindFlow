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
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypedDict, cast

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
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
    }
    return result


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
    call_count: int
    transcript: list[TranscriptEntry]
    disagreement_summary: DisagreementSummary | None
    rebuttal_delta: object | None  # RebuttalDelta — lazy import to avoid circular
    runtime: _PanelRunContext


@dataclass
class _PanelRunContext:
    """Mutable state owned by exactly one panel invocation."""

    call_count: int = 0
    transcript: list[TranscriptEntry] = field(default_factory=list)
    budget_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


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

        The graph structure (nodes + edges) is static across all invocations.
        Node closures capture only ``self`` (the orchestrator instance); all
        per-call data (``bundle_json``, ``valid_metrics``) is read from state
        so the compiled graph can be reused safely.
        """
        graph = StateGraph(PanelState)

        # ── Node: analyst_node ──────────────────────────────────────────
        async def analyst_node(state: PanelState) -> dict[str, Any]:
            """Round 0: call the data analyst."""
            logger.info("Panel round 0: Analyst")
            raw = await self._call_with_budget(state["runtime"], ANALYST, state["bundle_json"])
            analyst = _parse_analyst_opinion(raw, ANALYST)
            # Analyst citation validation (prevent hallucinated refs reaching moderator)
            bogus = validate_citations(analyst, state["valid_metrics"])
            if bogus:
                logger.warning("Hallucinated citations {} in analyst — marking", bogus)
                # Re-parse as skipped to keep graph flowing
                analyst = ExpertOpinion(
                    role=ANALYST.role,
                    perspective=ANALYST.perspective,
                    attribution_types=(),
                    confidence={},
                    evidence_citations=(),
                    argument="",
                    raw_json=raw,
                    skipped=True,
                )
            entry = TranscriptEntry(role=ANALYST.role, content=_opinion_summary(analyst), round=0)
            state["runtime"].transcript.append(entry)
            return {
                "analyst_opinion": analyst,
                "transcript": list(state["runtime"].transcript),
                "call_count": state["runtime"].call_count,
            }

        # ── Node: attribution_node ──────────────────────────────────────
        async def attribution_node(state: PanelState) -> dict[str, Any]:
            """Round 1: call all three attribution experts in parallel.

            Supports 1 retry per expert on forbidden-word rejection.
            """
            logger.info("Panel round 1: Attribution experts (parallel)")

            async def _call_and_parse(exp: ExpertDef) -> ExpertOpinion:
                raw = await self._safe_call_with_budget(state["runtime"], exp, state["bundle_json"])
                op = _parse_expert_opinion(raw, exp, valid_metrics=state["valid_metrics"])

                # Retry once if skipped due to forbidden words
                if op.skipped and _contains_forbidden_words(raw):
                    logger.warning("{} triggered forbidden words, retrying once", exp.role)
                    retry_msg = (
                        "你的上一条回复包含禁用词汇（诊断、治疗、患者、处方）。"
                        "请用中文重新输出，严格遵守禁用词规则。"
                    )
                    raw2 = await self._safe_call_with_budget(state["runtime"], exp, retry_msg)
                    op2 = _parse_expert_opinion(raw2, exp, valid_metrics=state["valid_metrics"])
                    if not op2.skipped:
                        return op2
                    logger.warning("{} retry still failed, using original", exp.role)

                return op

            results = await asyncio.gather(*[
                _call_and_parse(exp) for exp in ATTRIBUTION_EXPERTS
            ])
            opinions = list(results)
            for op in opinions:
                state["runtime"].transcript.append(
                    TranscriptEntry(role=op.role, content=_opinion_summary(op), round=1),
                )

            non_skipped = [o for o in opinions if not o.skipped]
            if len(non_skipped) < 2:
                raise PanelUnavailableError(
                    reason=f"仅{len(non_skipped)}份归因意见有效，需至少2份",
                    call_count=state["runtime"].call_count,
                )

            return {
                "attribution_opinions": opinions,
                "transcript": list(state["runtime"].transcript),
                "call_count": state["runtime"].call_count,
            }

        # ── Node: conflict_detection_node ───────────────────────────────
        async def conflict_detection_node(state: PanelState) -> dict[str, Any]:
            """Pure-function conflict detection + disagreement analytics — no LLM call."""
            logger.info("Conflict detection")
            conflict = detect_conflict(state["attribution_opinions"])
            escalated = conflict.has_conflict
            if escalated:
                logger.info("Conflict detected: {}", conflict.details)
            else:
                logger.info("No conflict among attribution experts")

            # Compute structured disagreement analytics
            ds = analyze_disagreement(
                state["attribution_opinions"],
                conflict.details,
                conflict.max_confidence_gap,
                rebuttal_delta=None,
            )
            logger.info(
                "Disagreement analytics: agreement={:.3f}, stability={}",
                ds.agreement_strength,
                ds.stability,
            )

            return {
                "conflict_report": conflict,
                "escalated": escalated,
                "disagreement_summary": ds,
            }

        # ── Node: rebuttal_node ─────────────────────────────────────────
        async def rebuttal_node(state: PanelState) -> dict[str, Any]:
            """Round 2a: attribution experts rebut each other (parallel)."""
            logger.info("Panel round 2a: Attribution rebuttal (parallel)")
            opinions = state["attribution_opinions"]
            prompts = [
                (ATTRIBUTION_EXPERTS[i], _build_rebuttal_prompt(state["bundle_json"], opinions, i))
                for i in range(len(ATTRIBUTION_EXPERTS))
            ]
            responses = await asyncio.gather(*[
                self._safe_call_with_budget(state["runtime"], exp, msg) for exp, msg in prompts
            ])
            new_opinions = []
            for raw, exp in zip(responses, ATTRIBUTION_EXPERTS, strict=True):
                op = _parse_expert_opinion(raw, exp, valid_metrics=state["valid_metrics"])
                # Retry once on forbidden words
                if op.skipped and _contains_forbidden_words(raw):
                    logger.warning("{} rebuttal triggered forbidden words, retrying", exp.role)
                    retry_msg = (
                        "你的上一条回复包含禁用词汇（诊断、治疗、患者、处方）。"
                        "请用中文重新输出，严格遵守禁用词规则并回到推理内容。"
                    )
                    raw2 = await self._safe_call_with_budget(state["runtime"], exp, retry_msg)
                    op2 = _parse_expert_opinion(raw2, exp, valid_metrics=state["valid_metrics"])
                    if not op2.skipped:
                        op = op2
                new_opinions.append(op)
            for op in new_opinions:
                state["runtime"].transcript.append(
                    TranscriptEntry(role=op.role, content=_opinion_summary(op), round=2),
                )

            non_skipped = [o for o in new_opinions if not o.skipped]
            if len(non_skipped) < 2:
                raise PanelUnavailableError(
                    reason=f"辩论后仅{len(non_skipped)}份归因意见有效",
                    call_count=state["runtime"].call_count,
                )

            # Compute rebuttal delta: pre-debate vs post-debate convergence
            delta = compute_rebuttal_delta(opinions, new_opinions)
            logger.info(
                "Rebuttal delta: agreement {:.3f}→{:.3f}, delta={:+.3f}, converged={}",
                delta.before_agreement,
                delta.after_agreement,
                delta.agreement_delta,
                delta.converged,
            )

            return {
                "attribution_opinions": new_opinions,
                "transcript": list(state["runtime"].transcript),
                "call_count": state["runtime"].call_count,
                "rebuttal_delta": delta,
            }

        # ── Node: moderator_node ────────────────────────────────────────
        async def moderator_node(state: PanelState) -> dict[str, Any]:
            """Round 2b/3/4: moderator synthesises the verdict.

            Supports both first-pass and redo (when critic rejected).
            """
            is_redo = state["critic_retries"] > 0
            analyst = state["analyst_opinion"]
            conflict = state["conflict_report"]
            # These are guaranteed non-None by graph execution order
            assert analyst is not None
            assert conflict is not None

            if is_redo:
                round_num = 4
                prompt = _build_moderator_redo_prompt(
                    state["bundle_json"],
                    analyst,
                    state["attribution_opinions"],
                    conflict,
                    cast(CriticResult, state["critic_result"]).issues,
                )
            else:
                round_num = 2 if not state["escalated"] else 3
                prompt = _build_moderator_user_prompt(
                    state["bundle_json"],
                    analyst,
                    state["attribution_opinions"],
                    conflict,
                )

            logger.info("Panel round {}: Moderator", round_num)
            raw = await self._call_with_budget(state["runtime"], MODERATOR, prompt)
            verdict = _parse_verdict(raw)
            if verdict is None:
                raise PanelUnavailableError(
                    reason="主持人输出解析失败",
                    call_count=state["runtime"].call_count,
                )

            state["runtime"].transcript.append(
                TranscriptEntry(
                    role=MODERATOR.role,
                    content=_verdict_summary(verdict),
                    round=round_num,
                ),
            )

            return {
                "moderator_verdict": verdict,
                "transcript": list(state["runtime"].transcript),
                "call_count": state["runtime"].call_count,
            }

        # ── Node: critic_node ───────────────────────────────────────────
        async def critic_node(state: PanelState) -> dict[str, Any]:
            """Round 3/4/5: critic validates the moderator's verdict."""
            # Determine round number based on escalation + retries
            base_round = 2 if not state["escalated"] else 3
            round_num = base_round + 1 + state["critic_retries"]

            logger.info("Panel round {}: Critic", round_num)

            pending_verdict = _verdict_dict_to_panel_verdict(
                cast(dict[str, Any], state["moderator_verdict"]),
                state["escalated"],
                tuple(state["runtime"].transcript),
                state["runtime"].call_count,
            )

            all_opinions: list[ExpertOpinion] = [
                cast(ExpertOpinion, state["analyst_opinion"]),
                *state["attribution_opinions"],
            ]
            prompt = _build_critic_user_prompt(
                state["bundle_json"],
                pending_verdict,
                all_opinions,
                state["valid_metrics"],
            )
            raw = await self._call_with_budget(state["runtime"], CRITIC, prompt)
            result = _parse_critic(raw)
            state["runtime"].transcript.append(
                TranscriptEntry(role=CRITIC.role, content=_critic_summary(result), round=round_num),
            )

            updates: dict[str, Any] = {
                "critic_result": result,
                "transcript": list(state["runtime"].transcript),
                "call_count": state["runtime"].call_count,
            }
            if not result.approved:
                updates["critic_retries"] = state["critic_retries"] + 1
            return updates

        # ── Conditional route helpers ───────────────────────────────────

        def should_escalate(state: PanelState) -> str:
            """Route: conflict detected → rebuttal, else → moderator."""
            return "rebuttal" if state["escalated"] else "moderator"

        def critic_verdict(state: PanelState) -> str:
            """Route: approved→END, rejected+retries<2→redo, else→END."""
            if cast(CriticResult, state["critic_result"]).approved:
                return "approved"
            # Maximum 1 retry: critic_retries is incremented by moderator_node
            # on redo. After the re-pass through critic, if it still rejects,
            # critic_retries will be >= 2 → exhausted.
            if state["critic_retries"] < 2:
                return "rejected_retry"
            return "rejected_exhausted"

        # ── Wire graph ──────────────────────────────────────────────────
        graph.add_node("analyst", analyst_node)
        graph.add_node("attribution", attribution_node)
        graph.add_node("conflict_detection", conflict_detection_node)
        graph.add_node("rebuttal", rebuttal_node)
        graph.add_node("moderator", moderator_node)
        graph.add_node("critic", critic_node)

        graph.set_entry_point("analyst")
        graph.add_edge("analyst", "attribution")
        graph.add_edge("attribution", "conflict_detection")
        graph.add_conditional_edges(
            "conflict_detection",
            should_escalate,
            {"rebuttal": "rebuttal", "moderator": "moderator"},
        )
        graph.add_edge("rebuttal", "moderator")
        graph.add_edge("moderator", "critic")
        graph.add_conditional_edges(
            "critic",
            critic_verdict,
            {
                "approved": END,
                "rejected_retry": "moderator",
                "rejected_exhausted": END,
            },
        )

        return graph.compile()

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
            "attribution_opinions": [],
            "conflict_report": None,
            "escalated": False,
            "moderator_verdict": None,
            "critic_result": None,
            "critic_retries": 0,
            "call_count": 0,
            "transcript": [],
            "disagreement_summary": None,
            "rebuttal_delta": None,
            "runtime": runtime,
        }

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
