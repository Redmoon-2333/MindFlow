"""Role-level completion policies for the expert panel (optimisation plan 2.1).

Every LLM call the panel makes is shaped by one row of the plan's table:

===============  ================  ==================  ==========================
role             reasoning effort  max output tokens   intent
===============  ================  ==================  ==========================
analyst          low               1200                fact extraction + anomaly
                                                       ranking only
attribution      low               1000                structured claims only
rebuttal         low               800                 answer the conflicting
                                                       fields only
moderator        high              1600                synthesis + abstention
                                                       judgement only
critic           low               500                 approved / issues / detail
                                                       only
chat             provider default  2048                independent of panel policy
===============  ================  ==================  ==========================

``CompletionPolicy`` is a *request* policy, not a model policy: the gateway
applies it as per-request overrides on the already-constructed model, so no role
gets its own HTTP client or its own model instance.

The lookup is keyed by the expert **role** string (``ExpertDef.role``, i.e. the
Chinese role names in :mod:`mindflow.agents.experts`), by the plan's canonical
role keys, and by the ``ExpertDef`` itself. An unknown role resolves to ``None``,
which means "send no overrides" — exactly today's behaviour.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from mindflow.agents.experts import (
    ANALYST,
    ATTRIBUTION_EXPERTS,
    CRITIC,
    MODERATOR,
    ExpertDef,
)

# ── Canonical role keys (the plan's table rows) ────────────────────────────────

ROLE_ANALYST = "analyst"
ROLE_ATTRIBUTION = "attribution"
ROLE_REBUTTAL = "rebuttal"
ROLE_MODERATOR = "moderator"
ROLE_CRITIC = "critic"
ROLE_CHAT = "chat"

#: Minimum ``max_tokens`` for a DeepSeek request that enables thinking mode.
#: Thinking tokens count against the cap, so a cap sized for the answer alone
#: truncates the response (real-request observation, 2026-09-25:
#: ``LengthFinishReasonError`` on every 1000-token attribution call). 4096
#: leaves room for a bounded reasoning trace plus the structured answer while
#: staying far below DeepSeek's own 64K thinking default.
THINKING_TOKEN_FLOOR: int = 4096


@dataclass(frozen=True, slots=True)
class CompletionPolicy:
    """Per-request generation policy for one panel role.

    ``None`` means "leave the provider default in place" for that field — the
    gateway then sends no override at all, so a policy can never silently
    degrade a provider that does not support a parameter.

    Attributes:
        role: Canonical role key (one of the ``ROLE_*`` constants).
        reasoning_effort: Thinking tier to request, or ``None`` for the
            provider default (the chat policy).
        max_output_tokens: Output ceiling for the request, or ``None``.
        temperature: Sampling temperature, or ``None`` to keep the provider's.
        intent: One-line statement of what the role is allowed to do. Prompt
            engineering lives in ``experts.py``; this is the contract stated in
            the plan's table, kept next to the numbers so they cannot drift.
        node: Panel-graph node label used for observability grouping.
    """

    role: str
    reasoning_effort: str | None
    max_output_tokens: int | None
    temperature: float | None = None
    intent: str = ""
    node: str = ""

    def request_overrides(
        self, *, provider: str = "deepseek",
    ) -> dict[str, Any]:
        """Translate the policy into per-request kwargs for the chat model.

        Only fields the policy actually sets are returned.

        Args:
            provider: ``"deepseek"`` (production L1 — DeepSeek documents
                ``reasoning_effort``, ``thinking`` and ``max_tokens``),
                ``"ecnu"`` (legacy campus gateway spelling
                ``max_completion_tokens``), or ``"generic"`` (plain
                OpenAI-compatible body: token cap only, no thinking fields).
        """
        overrides: dict[str, Any] = {}
        if provider == "ecnu":
            if self.reasoning_effort is not None:
                overrides["reasoning_effort"] = self.reasoning_effort
            if self.max_output_tokens is not None:
                overrides["max_completion_tokens"] = self.max_output_tokens
            if self.temperature is not None:
                overrides["temperature"] = self.temperature
            return overrides

        if provider == "deepseek":
            # DeepSeek's OpenAI-compatible endpoint: ``max_tokens`` is the
            # documented output cap (``langchain-openai`` would rewrite a
            # top-level ``max_tokens`` into ``max_completion_tokens``, which
            # DeepSeek does not document, so it rides in ``extra_body``).
            # Thinking mode is requested per call and its intensity is the
            # policy's reasoning effort; ``temperature`` is NOT effective in
            # thinking mode, so it is only sent when thinking is off.
            extra_body: dict[str, Any] = {}
            thinking = self.reasoning_effort is not None
            if self.reasoning_effort is not None:
                overrides["reasoning_effort"] = self.reasoning_effort
                extra_body["thinking"] = {"type": "enabled"}
            if self.max_output_tokens is not None:
                # CRITICAL: thinking tokens are billed *against* ``max_tokens``.
                # A cap sized for the JSON answer alone (e.g. 1000) is consumed
                # by the reasoning trace before the answer starts, and the
                # provider then truncates with ``finish_reason=length`` —
                # measured against real DeepSeek on 2026-09-25: every
                # attribution call failed with ``LengthFinishReasonError``. The
                # floor keeps the answer budget intact while still bounding the
                # request; the policy's own cap remains the effective value when
                # thinking is off.
                extra_body["max_tokens"] = (
                    max(self.max_output_tokens, THINKING_TOKEN_FLOOR)
                    if thinking else self.max_output_tokens
                )
            if extra_body:
                overrides["extra_body"] = extra_body
            if self.temperature is not None and not thinking:
                overrides["temperature"] = self.temperature
            return overrides

        if self.max_output_tokens is not None:
            overrides["extra_body"] = {"max_tokens": self.max_output_tokens}
        if self.temperature is not None:
            overrides["temperature"] = self.temperature
        return overrides

    def as_dict(self) -> dict[str, Any]:
        """Scalar view for evidence trails and observability (no prompt text)."""
        return {
            "role": self.role,
            "reasoning_effort": self.reasoning_effort,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "intent": self.intent,
            "node": self.node,
        }


# ── The single role → policy table ────────────────────────────────────────────


def _attribution_policy(role: str, node: str) -> CompletionPolicy:
    """One shared attribution row for CBT / TMT / emotion-regulation experts."""
    return CompletionPolicy(
        role=role,
        reasoning_effort="low",
        max_output_tokens=1000,
        intent="只输出结构化归因主张（类型、置信度、证据引用），不做自由论证",
        node=node,
    )


ANALYST_POLICY: CompletionPolicy = CompletionPolicy(
    role=ROLE_ANALYST,
    reasoning_effort="low",
    max_output_tokens=1200,
    intent="只做事实提取与异常排序，不解释成因",
    node="analyst",
)

#: The plan's "cbt/tmt/emotion (attribution)" row: identical numbers, one per
#: expert so per-role aggregates stay distinguishable.
ATTRIBUTION_POLICY: CompletionPolicy = _attribution_policy(ROLE_ATTRIBUTION, "attribution")
CBT_POLICY: CompletionPolicy = replace(ATTRIBUTION_POLICY, role="cbt")
TMT_POLICY: CompletionPolicy = replace(ATTRIBUTION_POLICY, role="tmt")
EMOTION_POLICY: CompletionPolicy = replace(ATTRIBUTION_POLICY, role="emotion")

REBUTTAL_POLICY: CompletionPolicy = CompletionPolicy(
    role=ROLE_REBUTTAL,
    reasoning_effort="low",
    max_output_tokens=800,
    intent="只回答冲突字段（对方的归因类型与置信度是否成立），不重写全文",
    node="rebuttal",
)

MODERATOR_POLICY: CompletionPolicy = CompletionPolicy(
    role=ROLE_MODERATOR,
    reasoning_effort="high",
    max_output_tokens=1600,
    intent="只做综合裁决与数据不足判断（insufficient_data / evidence_gaps）",
    node="moderator",
)

CRITIC_POLICY: CompletionPolicy = CompletionPolicy(
    role=ROLE_CRITIC,
    reasoning_effort="low",
    max_output_tokens=500,
    intent="只输出 approved / issues / critique_detail，不新增分析",
    node="critic",
)

#: Chat is independent of the panel: provider-default reasoning effort (the
#: campus gateway's own default), its own output ceiling.
CHAT_POLICY: CompletionPolicy = CompletionPolicy(
    role=ROLE_CHAT,
    reasoning_effort=None,
    max_output_tokens=2048,
    intent="对话回复；沿用 provider 默认推理强度，与面板策略无关",
    node="chat",
)

#: The plan's table, keyed by canonical role.
POLICIES: Mapping[str, CompletionPolicy] = {
    ROLE_ANALYST: ANALYST_POLICY,
    ROLE_ATTRIBUTION: ATTRIBUTION_POLICY,
    ROLE_REBUTTAL: REBUTTAL_POLICY,
    ROLE_MODERATOR: MODERATOR_POLICY,
    ROLE_CRITIC: CRITIC_POLICY,
    ROLE_CHAT: CHAT_POLICY,
}


def _normalise(role: str) -> str:
    return " ".join(role.strip().lower().split())


#: Every panel expert paired with its policy (identity match first, role string
#: as the fallback for reconstructed defs).
_EXPERT_ROLE_POLICIES: tuple[tuple[ExpertDef, CompletionPolicy], ...] = (
    (ANALYST, ANALYST_POLICY),
    (ATTRIBUTION_EXPERTS[0], CBT_POLICY),
    (ATTRIBUTION_EXPERTS[1], TMT_POLICY),
    (ATTRIBUTION_EXPERTS[2], EMOTION_POLICY),
    (CRITIC, CRITIC_POLICY),
    (MODERATOR, MODERATOR_POLICY),
)


def _build_role_index() -> dict[str, CompletionPolicy]:
    """Alias index: canonical keys, English expert keys, Chinese role names."""
    index: dict[str, CompletionPolicy] = dict(POLICIES)
    index.update({
        "cbt": CBT_POLICY,
        "tmt": TMT_POLICY,
        "emotion": EMOTION_POLICY,
        "emotion_regulation": EMOTION_POLICY,
    })
    for expert, policy in _EXPERT_ROLE_POLICIES:
        index[_normalise(expert.role)] = policy
    return index


#: Built once at import: the table is static, and a lazily built index would
#: have to survive ``policy_for_expert`` re-entering the lookup.
_ROLE_INDEX: dict[str, CompletionPolicy] = _build_role_index()


def policy_for_expert(expert: ExpertDef) -> CompletionPolicy | None:
    """Return the policy for an :class:`ExpertDef`, or ``None`` when unknown.

    The attribution experts stay distinguishable (cbt / tmt / emotion) so
    observability can aggregate them separately even though the plan gives all
    three one shared row of numbers.
    """
    for candidate, policy in _EXPERT_ROLE_POLICIES:
        if expert is candidate:
            return policy
    return _ROLE_INDEX.get(_normalise(expert.role))


def policy_for_role(role: str | ExpertDef | None) -> CompletionPolicy | None:
    """Look up the completion policy for *role*.

    Accepts the expert role string (the Chinese names in ``experts.py``), the
    plan's canonical role key (``"analyst"``/``"attribution"``/…), or an
    ``ExpertDef``. Unknown roles — and ``None`` — return ``None``, which the
    gateway reads as "no overrides", preserving the pre-policy behaviour.
    """
    if role is None:
        return None
    if isinstance(role, ExpertDef):
        return policy_for_expert(role)
    return _ROLE_INDEX.get(_normalise(role))


__all__ = [
    "ANALYST_POLICY",
    "ATTRIBUTION_POLICY",
    "CBT_POLICY",
    "CHAT_POLICY",
    "CRITIC_POLICY",
    "CompletionPolicy",
    "EMOTION_POLICY",
    "MODERATOR_POLICY",
    "POLICIES",
    "REBUTTAL_POLICY",
    "ROLE_ANALYST",
    "ROLE_ATTRIBUTION",
    "ROLE_CHAT",
    "ROLE_CRITIC",
    "ROLE_MODERATOR",
    "ROLE_REBUTTAL",
    "TMT_POLICY",
    "policy_for_expert",
    "policy_for_role",
]
