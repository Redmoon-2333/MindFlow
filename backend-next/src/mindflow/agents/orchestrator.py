"""Expert panel deliberation helpers for the v2 PanelGraph.

This module holds the parsing, citation-validation, prompt-building, and
per-invocation budget/transcript helpers used by the v2 ``PanelGraph``
(``mindflow.graph.panel_graph``).  The legacy ``PanelOrchestrator`` class was
removed when PanelGraph became the only active panel path; these module-level
helpers are imported lazily by PanelGraph nodes (and their tests) to avoid
circular imports with ``panel_service``.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from loguru import logger
from pydantic import BaseModel

from mindflow.agents.claims import (
    ClaimLedger,
    claim_ledger_for,
    ledger_from_claims_payload,
    render_claims_argument,
    render_conflict_summary,
    render_ledger_table,
    summarize_conflicts,
    validate_ledger,
)
from mindflow.agents.conflict import ConflictReport
from mindflow.agents.experts import ExpertDef
from mindflow.agents.schemas import (
    TYPE_ALIASES,
    VALID_TECHNIQUES,
    VALID_TYPES,
    AnalystOutput,
    AttributionOutput,
    CriticOutput,
    ModeratorOutput,
)
from mindflow.agents.types import (
    CriticResult,
    ExpertOpinion,
    PanelVerdict,
    TranscriptEntry,
    _contains_forbidden_words,
)

# Canonical procrastination types / CBT techniques are validated through
# ``agents.schemas`` (VALID_TYPES / VALID_TECHNIQUES) so the schema layer stays
# the single source of truth for enum vocabularies.

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


_SchemaT = TypeVar("_SchemaT", bound=BaseModel)


def _parse_with_pydantic(
    raw: str,
    schema_class: type[_SchemaT],
    context: str,
) -> _SchemaT | None:
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
    valid_metrics: Collection[str],
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
        elif (canonical := bare_to_canonical.get(cite)) is not None:
            resolved.add(canonical)
        else:
            unresolved.add(cite)

    return tuple(sorted(unresolved))


def validate_verdict_schema(verdict: dict[str, Any]) -> list[str]:
    """Deterministically validate a moderator verdict before the critic call."""
    issues: list[str] = []
    verdict = normalize_verdict_types(verdict)

    types_raw = verdict.get("types")
    if not isinstance(types_raw, list):
        issues.append("types 必须为数组")
    else:
        if len(types_raw) > 3:
            issues.append("types 最多 3 个")
        for t in types_raw:
            if str(t) not in VALID_TYPES:
                issues.append(f"未知拖延类型: {t}")

    confidence = verdict.get("confidence")
    if not isinstance(confidence, dict):
        issues.append("confidence 必须为对象")
    else:
        for key, value in confidence.items():
            if str(key) not in VALID_TYPES:
                issues.append(f"置信度键不是合法类型: {key}")
            try:
                number = float(value)
                if not 0.0 <= number <= 1.0:
                    issues.append(f"置信度越界: {key}={value}")
            except (TypeError, ValueError):
                issues.append(f"置信度不是数字: {key}={value}")

    technique = verdict.get("recommended_technique")
    if technique is not None and str(technique) not in VALID_TECHNIQUES:
        issues.append(f"未知 CBT 技术: {technique}")
    return issues


# ``TYPE_ALIASES`` now lives in ``agents/schemas.py`` (the schema layer owns the
# canonical vocabularies); it stays imported/re-exported here because the project
# rule is "new types must be added to TYPE_ALIASES and to experts.py's enum list".


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
    valid_metrics: Collection[str] | None = None,
) -> ExpertOpinion:
    """Parse an expert's raw LLM response into ``ExpertOpinion``.

    Uses ``AttributionOutput.model_validate_json`` for type-safe parsing, which
    also runs the Phase 1.3 semantic gate (non-empty argument, at least one
    legal procrastination type, confidence per type, and at least one evidence
    citation).  A response that fails the gate is not an opinion and degrades
    to a skipped expert.

    If JSON parsing fails, returns a skipped opinion (graceful degradation).
    Forbidden-word and hallucinated-citation checks are still enforced at
    the code level (semantic validation that Pydantic cannot express).
    """
    if skipped:
        return _make_skipped_opinion(expert, raw)

    parsed = _parse_with_pydantic(raw, AttributionOutput, expert.role)
    if parsed is None:
        return _make_skipped_opinion(expert, raw)

    # ── Claim Ledger contract (phase 2.3) ──────────────────────────────
    if parsed.claims:
        ledger = ledger_from_claims_payload([c.model_dump() for c in parsed.claims])
        if valid_metrics is not None:
            ledger_issues = validate_ledger(ledger, valid_metrics)
            if ledger_issues:
                logger.warning(
                    "Invalid claim ledger for {}: {}", expert.role, ledger_issues,
                )
                return _make_skipped_opinion(expert, raw)
        argument = render_claims_argument(ledger)
        if _contains_forbidden_words(argument):
            logger.warning("Forbidden word in {} claim ledger — skipping", expert.role)
            return _make_skipped_opinion(expert, raw)
        types: list[str] = []
        confidence: dict[str, float] = {}
        citations: list[str] = []
        for claim in ledger.claims:
            if claim.type not in types:
                types.append(claim.type)
            confidence[claim.type] = max(
                confidence.get(claim.type, 0.0), float(claim.confidence),
            )
            for eid in claim.evidence_ids:
                if eid not in citations:
                    citations.append(eid)
        return ExpertOpinion(
            role=expert.role,
            perspective=expert.perspective,
            attribution_types=tuple(types),
            confidence=confidence,
            evidence_citations=tuple(citations),
            argument=argument,
            raw_json=raw,
            claims=ledger.claims,
        )

    if parsed.insufficient_data:
        # Explicit expert abstention: contributes no claim, so the panel
        # treats it exactly like a skipped expert (fail-safe, not fail-open).
        logger.info(
            "{} abstained (insufficient_data): {}", expert.role, parsed.evidence_gaps,
        )
        return _make_skipped_opinion(expert, raw)

    attribution_types = tuple(parsed.attribution_types)
    confidence = {k: float(v) for k, v in parsed.confidence.items()}
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
    into the generic ``ExpertOpinion`` shape.  A report with no pattern and
    no anomaly is rejected by the schema itself (Phase 1.3: an empty report
    is not a valid opinion).
    """
    parsed = _parse_with_pydantic(raw, AnalystOutput, expert.role)
    if parsed is None:
        return _make_skipped_opinion(expert, raw)

    evidence_citations = tuple(parsed.evidence_citations)

    # Build argument text from patterns + anomalies + declared top concerns
    parts: list[str] = []
    for p in parsed.patterns:
        if isinstance(p, dict):
            parts.append(f"[{p.get('severity', 'info')}] {p.get('description', '')}")
    for a in parsed.anomalies:
        if isinstance(a, dict):
            parts.append(f"异常-{a.get('metric', '')}: {a.get('detail', '')}")
    concerns = [str(c).strip() for c in parsed.top_concerns if str(c).strip()]
    if concerns:
        parts.append("重点关注：" + "；".join(concerns))
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

    A rejection without concrete issues is not actionable, so a placeholder
    issue is added rather than letting the moderator redo blind.

    Returns a safe default (not approved, with an explanation) on failure.
    """
    text = _strip_markdown_fences(raw)

    try:
        parsed = CriticOutput.model_validate_json(text)
    except Exception:
        logger.warning("Critic JSON parse failed")
        return CriticResult(approved=False, issues=("批评家输出解析失败",))

    issues = tuple(parsed.issues) if parsed.issues else ()
    if not parsed.approved and not issues:
        detail = parsed.critique_detail.strip()
        issues = (detail or "批评家驳回但未给出具体问题",)

    return CriticResult(approved=parsed.approved, issues=issues)


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
            (
                f"agreement_strength={disagreement_summary.agreement_strength:.3f}, "
                f"stability={disagreement_summary.stability}"
            ),
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
    valid_metrics: Collection[str],
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


def _collect_ledgers(
    analyst: ExpertOpinion | None,
    attribution_opinions: Sequence[ExpertOpinion],
) -> dict[str, ClaimLedger]:
    """Collect the validated ledgers of every participating attribution expert."""
    ledgers: dict[str, ClaimLedger] = {}
    for opinion in attribution_opinions:
        if opinion.skipped:
            continue
        ledgers[opinion.role] = claim_ledger_for(opinion)
    return ledgers


def _build_moderator_claims_prompt(
    bundle_json: str,
    analyst: ExpertOpinion | None,
    attribution_opinions: Sequence[ExpertOpinion],
    conflict: ConflictReport,
    disagreement_summary: Any | None = None,
    agreement_strength: float = 1.0,
) -> str:
    """Build the moderator prompt from *validated claims only* (phase 2.3).

    The moderator no longer receives several pages of expert prose: it receives
    the analyst digest, the validated claim table, the multi-expert conflict
    summary, and the instruction to abstain when the table cannot support a
    conclusion.
    """
    ledgers = _collect_ledgers(analyst, attribution_opinions)
    summary = summarize_conflicts(ledgers, agreement_strength)

    parts: list[str] = [
        "## 用户行为数据",
        bundle_json,
        "",
    ]

    if analyst is not None and not analyst.skipped and analyst.argument.strip():
        parts.extend([
            "## 数据分析师要点（已压缩）",
            analyst.argument.strip()[:800],
            "",
        ])

    parts.extend([
        "## 已校验的 Claim Ledger（唯一可信的专家意见来源）",
        render_ledger_table(ledgers) if ledgers else "（无有效 claim）",
        "",
        "## 跨专家冲突摘要",
        render_conflict_summary(summary),
        "",
    ])

    if disagreement_summary is not None:
        parts.extend([
            "## 共识强度",
            (
                f"agreement_strength={disagreement_summary.agreement_strength:.3f}, "
                f"stability={disagreement_summary.stability}"
            ),
            "",
        ])

    if conflict.has_conflict:
        parts.extend(["## 冲突检测报告", conflict.details, ""])

    parts.append(
        "## 你的任务\n"
        "只依据上表中的 claim 与证据裁决：\n"
        "1. 综合一致 claim 形成结论，冲突时按证据强度裁决并把被否决方写入 dissent；\n"
        "2. 若表中没有任何 claim 能支撑结论，输出 insufficient_data=true 并填写 evidence_gaps；\n"
        "3. 不要引入表中不存在的新证据、新类型或未经校验的论断。"
    )
    return "\n".join(parts)


def _build_moderator_claims_redo_prompt(
    bundle_json: str,
    analyst: ExpertOpinion | None,
    attribution_opinions: Sequence[ExpertOpinion],
    conflict: ConflictReport,
    critic_issues: tuple[str, ...],
    disagreement_summary: Any | None = None,
    agreement_strength: float = 1.0,
) -> str:
    """Redo variant of :func:`_build_moderator_claims_prompt`."""
    base = _build_moderator_claims_prompt(
        bundle_json,
        analyst,
        attribution_opinions,
        conflict,
        disagreement_summary,
        agreement_strength,
    )
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


# ── Per-invocation runtime (budget + transcript) ─────────────────────────────


@dataclass
class _PanelRunContext:
    """Mutable state owned by exactly one panel invocation."""

    call_count: int = 0
    transcript: list[TranscriptEntry] = field(default_factory=list)
    budget_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Per-role LLM-call usage for phase budgets (architecture plan G/1.3).
    phase_usage: dict[str, int] = field(default_factory=dict)


# Context variable to carry the mutable per-invocation runtime through the
# LangGraph StateGraph without including it in the checkpointable state.
# Set by ``PanelGraph.ainvoke`` before ``ainvoke`` and read by all graph nodes.
_PANEL_RUNTIME: contextvars.ContextVar[_PanelRunContext] = contextvars.ContextVar(
    "_PANEL_RUNTIME",
)
