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
from typing import Any, TypedDict, cast

from langgraph.graph import END, StateGraph
from loguru import logger

from mindflow.agents.conflict import ConflictReport, detect_conflict
from mindflow.agents.experts import (
    ANALYST,
    ATTRIBUTION_EXPERTS,
    CRITIC,
    MODERATOR,
    ExpertDef,
)
from mindflow.agents.llm_gateway import PanelLLMGateway
from mindflow.agents.types import (
    FORBIDDEN_WORDS,
    CriticResult,
    ExpertOpinion,
    PanelBudgetExceededError,
    PanelUnavailableError,
    PanelVerdict,
    TranscriptEntry,
)
from mindflow.domain.evidence import EvidenceBundle, metric_names, to_prompt_json
from mindflow.domain.procrastination import CBTTechnique, ProcrastinationType

# ── Parsing helpers ────────────────────────────────────────────────────────────


def _contains_forbidden_words(text: str) -> str | None:
    """Return the first forbidden word found in *text*, or None."""
    for word in FORBIDDEN_WORDS:
        if word in text:
            return word
    return None


def _safe_parse_json(raw: str, context: str) -> dict[str, Any] | None:
    """Parse *raw* as JSON, returning None on failure.

    Strips Markdown fence markers if present.
    """
    text = raw.strip()
    # Strip ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[: -3].strip()

    try:
        result: dict[str, Any] = json.loads(text)
        return result
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse failed for {}: {}", context, exc)
        return None


_CITATION_PATTERN = re.compile(r"\[证据[:：]\s*([A-Za-z0-9_]+)\s*\]")


def validate_citations(
    opinion: ExpertOpinion,
    valid_metrics: frozenset[str],
) -> tuple[str, ...]:
    """Code-level citation validation — never trust the LLM critic alone.

    Extracts every ``[证据: metric]`` reference from the argument plus the
    structured ``evidence_citations`` field, and returns the subset that does
    NOT exist in the bundle's metric_names (design §3: hallucinated citations
    must be caught mechanically; review P1 fix).
    """
    cited: set[str] = set(opinion.evidence_citations)
    cited.update(_CITATION_PATTERN.findall(opinion.argument))
    return tuple(sorted(cited - valid_metrics))


def _parse_expert_opinion(
    raw: str,
    expert: ExpertDef,
    skipped: bool = False,
    valid_metrics: frozenset[str] | None = None,
) -> ExpertOpinion:
    """Parse an expert's raw LLM response into ``ExpertOpinion``.

    If JSON parsing fails, returns a skipped opinion (graceful degradation).
    If forbidden words are found, also skips. If *valid_metrics* is given,
    hallucinated citations mark the opinion skipped as well (review P1).
    """
    if skipped:
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

    data = _safe_parse_json(raw, expert.role)
    if data is None:
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

    # Extract fields with safe defaults
    attribution_types = tuple(data.get("attribution_types", []))
    confidence_raw = data.get("confidence", {})
    if not isinstance(confidence_raw, dict):
        confidence_raw = {}
    confidence: dict[str, float] = {}
    for k, v in confidence_raw.items():
        if isinstance(v, (int, float)):
            confidence[str(k)] = float(v)

    evidence_citations = tuple(data.get("evidence_citations", []))
    argument = str(data.get("argument", ""))

    # Check forbidden words
    forbidden = _contains_forbidden_words(argument)
    if forbidden:
        logger.warning("Forbidden word {!r} in {} opinion — skipping", forbidden, expert.role)
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

    return opinion


def _parse_analyst_opinion(
    raw: str,
    expert: ExpertDef,
) -> ExpertOpinion:
    """Parse analyst output, which has a different JSON shape.

    The analyst outputs ``patterns`` / ``anomalies`` / ``top_concerns``
    rather than ``attribution_types`` / ``confidence``. We map those
    into the generic ``ExpertOpinion`` shape.
    """
    data = _safe_parse_json(raw, expert.role)
    if data is None:
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

    evidence_citations = tuple(data.get("evidence_citations", []))

    # Build argument text from patterns + anomalies
    patterns = data.get("patterns", [])
    anomalies = data.get("anomalies", [])

    parts: list[str] = []
    for p in patterns:
        if isinstance(p, dict):
            parts.append(f"[{p.get('severity', 'info')}] {p.get('description', '')}")
    for a in anomalies:
        if isinstance(a, dict):
            parts.append(f"异常-{a.get('metric', '')}: {a.get('detail', '')}")
    argument = "\n".join(parts) if parts else data.get("argument", "") or ""

    # Check forbidden words
    forbidden = _contains_forbidden_words(argument)
    if forbidden:
        logger.warning("Forbidden word {!r} in analyst opinion — skipping", forbidden)
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

    Returns None on parse failure.
    """
    return _safe_parse_json(raw, "moderator")


def _parse_critic(raw: str) -> CriticResult:
    """Parse the critic's JSON output.

    Returns a safe default (not approved, with an explanation) on failure.
    """
    data = _safe_parse_json(raw, "critic")
    if data is None:
        return CriticResult(approved=False, issues=("批评家输出解析失败",))
    approved = bool(data.get("approved", False))
    issues_raw = data.get("issues", [])
    if not isinstance(issues_raw, list):
        issues: tuple[str, ...] = ("批评家输出格式异常：issues字段非数组",)
    else:
        issues = tuple(str(i) for i in issues_raw)
    return CriticResult(approved=approved, issues=issues)


def _verdict_dict_to_panel_verdict(
    data: dict[str, Any],
    escalated: bool,
    transcript: tuple[TranscriptEntry, ...],
    call_count: int,
) -> PanelVerdict:
    """Convert a moderator's JSON dict into a ``PanelVerdict``.

    This is a best-effort conversion: unknown types or techniques are
    silently mapped to safe defaults rather than crashing the panel.
    """
    # Parse types
    types_raw: list[str] = data.get("types", []) if isinstance(data.get("types"), list) else []
    parsed_types: list[ProcrastinationType] = []
    for t in types_raw:
        try:
            parsed_types.append(ProcrastinationType(t))
        except ValueError:
            logger.warning("Unknown procrastination type in verdict: {!r}", t)

    if not parsed_types:
        # Fallback — should not happen with a well-behaved moderator
        parsed_types = [ProcrastinationType.TASK_AVERSION]

    # Parse confidence
    conf_raw: dict[str, object] = (
        data.get("confidence", {}) if isinstance(data.get("confidence"), dict) else {}
    )
    confidence: dict[ProcrastinationType, float] = {}
    for k, v in conf_raw.items():
        try:
            pt = ProcrastinationType(k)
            if isinstance(v, (int, float)):
                confidence[pt] = float(v)
        except ValueError:
            pass

    # Fill in any missing types with a default confidence
    for pt in parsed_types:
        if pt not in confidence:
            confidence[pt] = 0.5

    # Parse technique
    technique_raw = data.get("recommended_technique")
    technique: CBTTechnique | None = None
    if technique_raw is not None:
        try:
            technique = CBTTechnique(str(technique_raw))
        except ValueError:
            logger.warning("Unknown CBT technique in verdict: {!r}", technique_raw)

    # Parse rationale and dissent
    rationale = str(data.get("rationale", ""))
    dissent_raw: list[str] = data.get("dissent", [])
    if not isinstance(dissent_raw, list):
        dissent_raw = []
    dissent = tuple(str(d) for d in dissent_raw)

    return PanelVerdict(
        types=tuple(parsed_types),
        confidence=confidence,
        recommended_technique=technique,
        rationale=rationale,
        dissent=dissent,
        transcript=transcript,
        escalated=escalated,
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
        self._call_count: int = 0
        self._transcript: list[TranscriptEntry] = []
        # Serializes budget check-and-increment across parallel batches (P2).
        self._budget_lock = asyncio.Lock()
        self._compiled_graph = None

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
        self._call_count = 0
        self._transcript = []
        self._budget_lock = asyncio.Lock()

        try:
            return await self._run_graph(bundle)
        except (PanelBudgetExceededError, PanelUnavailableError):
            raise
        except Exception as exc:
            logger.error("Panel orchestrator unexpected error: {}", exc)
            raise PanelUnavailableError(
                reason=f"编排器异常：{exc}",
                call_count=self._call_count,
            ) from exc

    # ── LangGraph orchestration ───────────────────────────────────────────

    def _build_compiled_graph(self):  # noqa: ANN202 — return type is langgraph CompiledStateGraph
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
            raw = await self._call_with_budget(ANALYST, state["bundle_json"])
            analyst = _parse_analyst_opinion(raw, ANALYST)
            entry = TranscriptEntry(role=ANALYST.role, content=_opinion_summary(analyst), round=0)
            self._transcript.append(entry)
            return {
                "analyst_opinion": analyst,
                "transcript": list(self._transcript),
                "call_count": self._call_count,
            }

        # ── Node: attribution_node ──────────────────────────────────────
        async def attribution_node(state: PanelState) -> dict[str, Any]:
            """Round 1: call all three attribution experts in parallel."""
            logger.info("Panel round 1: Attribution experts (parallel)")
            responses = await asyncio.gather(*[
                self._safe_call_with_budget(exp, state["bundle_json"])
                for exp in ATTRIBUTION_EXPERTS
            ])
            opinions = [
                _parse_expert_opinion(raw, exp, valid_metrics=state["valid_metrics"])
                for raw, exp in zip(responses, ATTRIBUTION_EXPERTS, strict=True)
            ]
            for op in opinions:
                self._transcript.append(
                    TranscriptEntry(role=op.role, content=_opinion_summary(op), round=1),
                )

            non_skipped = [o for o in opinions if not o.skipped]
            if len(non_skipped) < 2:
                raise PanelUnavailableError(
                    reason=f"仅{len(non_skipped)}份归因意见有效，需至少2份",
                    call_count=self._call_count,
                )

            return {
                "attribution_opinions": opinions,
                "transcript": list(self._transcript),
                "call_count": self._call_count,
            }

        # ── Node: conflict_detection_node ───────────────────────────────
        async def conflict_detection_node(state: PanelState) -> dict[str, Any]:
            """Pure-function conflict detection — no LLM call."""
            logger.info("Conflict detection")
            conflict = detect_conflict(state["attribution_opinions"])
            escalated = conflict.has_conflict
            if escalated:
                logger.info("Conflict detected: {}", conflict.details)
            else:
                logger.info("No conflict among attribution experts")
            return {"conflict_report": conflict, "escalated": escalated}

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
                self._safe_call_with_budget(exp, msg) for exp, msg in prompts
            ])
            new_opinions = [
                _parse_expert_opinion(raw, exp, valid_metrics=state["valid_metrics"])
                for raw, exp in zip(responses, ATTRIBUTION_EXPERTS, strict=True)
            ]
            for op in new_opinions:
                self._transcript.append(
                    TranscriptEntry(role=op.role, content=_opinion_summary(op), round=2),
                )

            non_skipped = [o for o in new_opinions if not o.skipped]
            if len(non_skipped) < 2:
                raise PanelUnavailableError(
                    reason=f"辩论后仅{len(non_skipped)}份归因意见有效",
                    call_count=self._call_count,
                )

            return {
                "attribution_opinions": new_opinions,
                "transcript": list(self._transcript),
                "call_count": self._call_count,
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
            raw = await self._call_with_budget(MODERATOR, prompt)
            verdict = _parse_verdict(raw)
            if verdict is None:
                raise PanelUnavailableError(
                    reason="主持人输出解析失败",
                    call_count=self._call_count,
                )

            self._transcript.append(
                TranscriptEntry(
                    role=MODERATOR.role,
                    content=_verdict_summary(verdict),
                    round=round_num,
                ),
            )

            updates: dict[str, Any] = {
                "moderator_verdict": verdict,
                "transcript": list(self._transcript),
                "call_count": self._call_count,
            }
            if is_redo:
                # Increment critic_retries to track the redo cycle.
                # The critic_verdict edge uses this to stop after 1 retry.
                updates["critic_retries"] = state["critic_retries"] + 1

            return updates

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
                tuple(self._transcript),
                self._call_count,
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
            raw = await self._call_with_budget(CRITIC, prompt)
            result = _parse_critic(raw)
            self._transcript.append(
                TranscriptEntry(role=CRITIC.role, content=_critic_summary(result), round=round_num),
            )

            return {
                "critic_result": result,
                "transcript": list(self._transcript),
                "call_count": self._call_count,
            }

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

    def _get_compiled_graph(self):  # noqa: ANN202 — return type is langgraph CompiledStateGraph
        """Return the compiled graph, building it on first access (lazy)."""
        if self._compiled_graph is None:
            self._compiled_graph = self._build_compiled_graph()
        return self._compiled_graph

    async def _run_graph(self, bundle: EvidenceBundle) -> PanelVerdict:
        """Run the compiled LangGraph StateGraph for this session."""
        bundle_json = to_prompt_json(bundle)
        valid_metrics = metric_names(bundle)

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
        }

        final = await compiled.ainvoke(initial)

        return _verdict_dict_to_panel_verdict(
            cast(dict[str, Any], final["moderator_verdict"]),
            final["escalated"],
            tuple(self._transcript),
            self._call_count,
        )

    # ── Gateway helpers ───────────────────────────────────────────────────

    async def _call_with_budget(
        self,
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
        async with self._budget_lock:
            self._call_count += 1
            if self._call_count > 12:
                raise PanelBudgetExceededError(call_count=self._call_count)
        return await self._gateway.complete(
            system=expert.system_prompt,
            user=user_message,
            model=expert.model,
        )

    async def _safe_call_with_budget(
        self,
        expert: ExpertDef,
        user_message: str,
    ) -> str:
        """Like ``_call_with_budget`` but returns empty string on failure.

        Used in parallel batches so a single failed call doesn't abort the group.
        """
        try:
            return await self._call_with_budget(expert, user_message)
        except Exception as exc:
            logger.error("Parallel call to {} failed: {}", expert.role, exc)
            return ""
