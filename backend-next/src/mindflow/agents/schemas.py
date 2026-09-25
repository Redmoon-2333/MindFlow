"""Pydantic v2 structured output schemas for the expert panel pipeline.

Replaces the ~200 lines of manual JSON parsing in ``orchestrator.py``
(``_parse_expert_opinion``, ``_parse_analyst_opinion``, ``_parse_verdict``,
``_parse_critic``) with Pydantic ``model_validate_json()`` calls.

Each schema matches the JSON contract that the corresponding expert LLM is
prompted to produce (``agents/experts.py``). CRITICAL: ``CriticOutput.approved``
is a strict boolean — Pydantic correctly parses JSON ``true``/``false``, fixing
the bug where ``bool("false") == True`` in the manual parsing code.

Phase 1.3 contract rules — every production schema is a ``_StrictModel``:

  1. ``extra="forbid"``: field drift is a hard, logged error instead of being
     silently dropped.  Every field a prompt asks for exists in its schema
     (``top_concerns``, ``cognitive_distortions``, ``tmt_factors``,
     ``emotion_pattern``, ``is_emotion_driven``, ``critique_detail``).
  2. Deterministic semantic validation (``validate_opinion_semantics``):
     an opinion with an empty argument, no legal procrastination type, no
     evidence citation, or a confidence/citation mismatch is **not** a valid
     expert opinion.
  3. Citation, confidence, type-enum and technique-enum checks are performed
     in code, never delegated to the LLM critic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Self

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mindflow.domain.procrastination import CBTTechnique, ProcrastinationType

# ── Canonical vocabularies (single source of truth) ─────────────────────────

VALID_TYPES: frozenset[str] = frozenset(t.value for t in ProcrastinationType)
VALID_TECHNIQUES: frozenset[str] = frozenset(t.value for t in CBTTechnique)

_PATTERN_SEVERITIES: frozenset[str] = frozenset({"info", "mild", "moderate", "severe"})

#: Prompt-level labels that map back to canonical enum values.  New types must
#: be added here *and* to ``experts.py``'s enum explanation.
TYPE_ALIASES: dict[str, str] = {
    "决策性拖延": "decisional",
    "任务价值感知不足型拖延": "task_aversion",
    "冲动型拖延": "impulsivity",
    "完美主义拖延": "perfectionism",
    "情绪调节型拖延": "emotional_regulation",
    "冲动分心": "impulsivity",
    "任务畏惧": "task_aversion",
    "决策困难": "decisional",
    "完美主义": "perfectionism",
    "情绪调节": "emotional_regulation",
}

#: A claim at or above this confidence must cite at least one real metric.
HIGH_CONFIDENCE_THRESHOLD: float = 0.7


def canonical_type(raw: object) -> str | None:
    """Return the canonical procrastination type for *raw*, or None."""
    text = str(raw).strip()
    if not text:
        return None
    if text in VALID_TYPES:
        return text
    return TYPE_ALIASES.get(text)


def canonical_types(values: Sequence[object]) -> tuple[list[str], list[str]]:
    """Normalise a type list, returning ``(kept, dropped)``."""
    kept: list[str] = []
    dropped: list[str] = []
    for raw in values:
        canonical = canonical_type(raw)
        if canonical is None:
            dropped.append(str(raw))
        elif canonical not in kept:
            kept.append(canonical)
    return kept, dropped


# ── Base model ──────────────────────────────────────────────────────────────


class _StrictModel(BaseModel):
    """Base class for every production panel schema.

    ``extra="forbid"`` makes prompt/schema drift loud: a field the prompt asks
    for but the schema lacks (or vice versa) surfaces as a parse failure with a
    logged reason instead of a silently discarded value.
    """

    model_config = ConfigDict(extra="forbid")


# ── Schemas ─────────────────────────────────────────────────────────────────


class AnalystOutput(_StrictModel):
    """数据分析师结构化输出。

    The analyst discovers behavior patterns and anomalies from sensor data.
    """

    patterns: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Discovered behavior patterns with severity and description",
    )
    anomalies: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Anomalies with metric and detail",
    )
    top_concerns: list[str] = Field(
        default_factory=list,
        description="最值得关注的 1-3 个问题（prompt 要求，schema 必须承接）",
    )
    evidence_citations: list[str] = Field(
        default_factory=list,
        description="Evidence metric IDs cited",
    )

    @field_validator("patterns")
    @classmethod
    def _clean_patterns(cls, value: list[Any]) -> list[dict[str, Any]]:
        """Drop substance-free patterns and normalise the severity label."""
        cleaned: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            description = str(item.get("description", "")).strip()
            if not description:
                continue
            severity = str(item.get("severity", "info")).strip().lower()
            if severity not in _PATTERN_SEVERITIES:
                severity = "info"
            cleaned.append({**item, "severity": severity, "description": description})
        return cleaned

    @field_validator("anomalies")
    @classmethod
    def _clean_anomalies(cls, value: list[Any]) -> list[dict[str, Any]]:
        """An anomaly without a metric or a detail is not an anomaly."""
        cleaned: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            detail = str(item.get("detail", "")).strip()
            metric = str(item.get("metric", "")).strip()
            if not detail and not metric:
                continue
            cleaned.append({**item, "metric": metric, "detail": detail})
        return cleaned

    @model_validator(mode="after")
    def _require_substance(self) -> Self:
        """An empty analyst report is not a valid opinion (Phase 1.3)."""
        if not self.patterns and not self.anomalies:
            raise ValueError("分析师输出必须至少包含一个模式或异常（空论据不算有效意见）")
        return self


class ClaimModel(_StrictModel):
    """One row of an expert's Claim Ledger (phase 2.3).

    The ledger is the validated unit of reasoning: free-text argumentation is
    replaced by claims whose evidence ids, confidence, support and competing
    explanation are all machine-checkable.
    """

    type: str = Field(description="Canonical procrastination type")
    confidence: float = Field(ge=0, le=1, description="Confidence in [0, 1]")
    evidence_ids: list[str] = Field(
        default_factory=list, description="Catalog ids supporting this claim",
    )
    support: str = Field(default="", description="Chinese explanation tied to the evidence")
    alternative: str = Field(
        default="", description="The main competing explanation considered",
    )

    @model_validator(mode="after")
    def _check_claim(self) -> Self:
        canonical = canonical_type(self.type)
        if canonical is None:
            raise ValueError(f"未知 claim 类型: {self.type}")
        self.type = canonical
        if not [e for e in self.evidence_ids if str(e).strip()]:
            raise ValueError(f"claim 必须至少引用一条证据: {canonical}")
        if not self.support.strip():
            raise ValueError(f"claim 缺少论据 support: {canonical}")
        if not self.alternative.strip():
            raise ValueError(f"claim 缺少替代解释 alternative: {canonical}")
        return self


class AttributionOutput(_StrictModel):
    """归因专家结构化输出（CBT/TMT/情绪）。

    Two contracts are accepted, in this order of preference:

    1. **Claim Ledger** (``claims`` + ``insufficient_data`` + ``evidence_gaps``)
       — the phase 2.3 contract; every claim is validated in code.
    2. The legacy free-text contract (``attribution_types`` + ``confidence`` +
       ``argument`` + ``evidence_citations``) — still parsed so older prompts,
       mock gateways and recorded fixtures keep working.  It is converted into
       an equivalent ledger downstream.

    The prompt-aligned optional fields (``cognitive_distortions`` /
    ``tmt_factors`` / ``emotion_pattern`` / ``is_emotion_driven``) are carried
    per expert rather than dropped at the schema boundary.
    """

    attribution_types: list[str] = Field(
        default_factory=list,
        description="1-3 procrastination types identified (legacy contract)",
    )
    confidence: dict[str, float] = Field(
        default_factory=dict,
        description="Per-type confidence scores 0-1 (legacy contract)",
    )
    argument: str = Field(
        default="",
        description="Full reasoning in Chinese with [证据: metric] citations (legacy contract)",
    )
    evidence_citations: list[str] = Field(
        default_factory=list,
        description="Evidence metric IDs cited in argument (legacy contract)",
    )
    claims: list[ClaimModel] = Field(
        default_factory=list,
        description="Claim Ledger rows (preferred contract)",
    )
    insufficient_data: bool = Field(
        default=False,
        description="True when this expert cannot support any claim",
    )
    evidence_gaps: list[str] = Field(
        default_factory=list,
        description="What evidence is missing when insufficient_data=true",
    )
    cognitive_distortions: list[str] = Field(
        default_factory=list,
        description="CBT prompt: identified cognitive distortions",
    )
    tmt_factors: dict[str, str] = Field(
        default_factory=dict,
        description="TMT prompt: Expectancy/Value/Impulsiveness/Delay levels",
    )
    emotion_pattern: str | None = Field(
        default=None,
        description="Emotion prompt: detected emotion-regulation pattern",
    )
    is_emotion_driven: bool | None = Field(
        default=None,
        description="Emotion prompt: whether emotion regulation dominates",
    )

    @field_validator("attribution_types", mode="before")
    @classmethod
    def _coerce_types(cls, value: object) -> object:
        """Accept a bare string (small models often emit one)."""
        if isinstance(value, str):
            return [value]
        return value

    @model_validator(mode="after")
    def _normalise_and_check(self) -> Self:
        kept, dropped = canonical_types(self.attribution_types)
        if dropped:
            logger.warning(
                "Attribution output contained unknown types {} — dropped", dropped,
            )
        self.attribution_types = kept

        confidence: dict[str, float] = {}
        for key, value in self.confidence.items():
            canonical = canonical_type(key)
            if canonical is None or canonical not in kept:
                logger.warning(
                    "Confidence key {!r} does not match a listed type — dropped", key,
                )
                continue
            number = float(value)
            if not 0.0 <= number <= 1.0:
                raise ValueError(f"置信度越界: {canonical}={number}")
            confidence[canonical] = number
        self.confidence = confidence

        if self.insufficient_data and not [
            gap for gap in self.evidence_gaps if str(gap).strip()
        ]:
            raise ValueError("insufficient_data=true 时必须提供 evidence_gaps")

        if self.claims:
            # Ledger contract: the schema already validated every claim row.
            return self

        if self.insufficient_data:
            # An explicit abstention needs no legacy fields.
            return self

        issues = validate_opinion_semantics(
            attribution_types=tuple(self.attribution_types),
            confidence=self.confidence,
            evidence_citations=tuple(self.evidence_citations),
            argument=self.argument,
        )
        if issues:
            raise ValueError("；".join(issues))
        return self


class ModeratorOutput(_StrictModel):
    """主持人/综合者结构化输出。

    The moderator synthesizes all expert opinions into a final verdict, or
    explicitly abstains with ``insufficient_data=True`` plus ``evidence_gaps``.
    """

    types: list[str] = Field(
        default_factory=list,
        description="Final procrastination type verdict",
    )
    confidence: dict[str, float] = Field(
        default_factory=dict,
        description="Per-type confidence scores",
    )
    recommended_technique: str | None = Field(
        default=None,
        description="Recommended CBT technique",
    )
    rationale: str = Field(
        default="",
        description="Chinese explanation",
    )
    dissent: list[str] = Field(
        default_factory=list,
        description="Recorded dissenting opinions",
    )
    insufficient_data: bool = Field(
        default=False,
        description="True when evidence is insufficient for a confident verdict",
    )
    uncertainty: float | None = Field(
        default=None, ge=0, le=1, description="Overall verdict uncertainty in [0, 1]",
    )
    evidence_gaps: list[str] = Field(
        default_factory=list,
        description="Missing evidence categories that prevent a stronger conclusion",
    )

    @field_validator("types", mode="before")
    @classmethod
    def _coerce_types(cls, value: object) -> object:
        if isinstance(value, str):
            return [value]
        return value

    @model_validator(mode="after")
    def _normalise_and_check(self) -> Self:
        kept, dropped = canonical_types(self.types)
        if dropped:
            logger.warning("Moderator verdict contained unknown types {} — dropped", dropped)
        self.types = kept

        confidence: dict[str, float] = {}
        for key, value in self.confidence.items():
            canonical = canonical_type(key)
            if canonical is None or canonical not in kept:
                continue
            number = float(value)
            if not 0.0 <= number <= 1.0:
                raise ValueError(f"置信度越界: {canonical}={number}")
            confidence[canonical] = number
        self.confidence = confidence

        technique = self.recommended_technique
        if technique is not None:
            technique = str(technique).strip()
            if not technique or technique.lower() in {"none", "null", "无", "无推荐"}:
                self.recommended_technique = None
            elif technique not in VALID_TECHNIQUES:
                raise ValueError(f"未知 CBT 技术: {technique}")
            else:
                self.recommended_technique = technique

        # An abstention must say what is missing; a real verdict must have a
        # type and a rationale.  Empty objects are not valid verdicts.
        if self.insufficient_data:
            if not [gap for gap in self.evidence_gaps if str(gap).strip()]:
                raise ValueError("insufficient_data=true 时必须提供 evidence_gaps")
        elif not self.types:
            raise ValueError("裁决必须至少给出一个合法拖延类型（或设置 insufficient_data=true）")
        if not self.rationale.strip():
            raise ValueError("裁决理由 rationale 不能为空")
        return self


class CriticOutput(_StrictModel):
    """批评家结构化输出。

    CRITICAL: ``approved`` 是严格布尔值 — Pydantic 正确解析 JSON
    ``true``/``false``，修复了 ``bool("false") == True`` 的 bug。
    """

    approved: bool = Field(
        default=False,
        description="Whether the verdict passes review.  A field_validator "
        "normalises int 0/1 (common LLM output) but rejects string coercions.",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Issues found (empty if approved)",
    )
    critique_detail: str = Field(
        default="",
        description="批评家审查说明（prompt 要求，≤300 字）",
    )

    @field_validator("approved", mode="before")
    @classmethod
    def _normalise_approved(cls, v: object) -> bool:
        """Normalise int 0/1 → False/True (common LLM output), reject strings.

        Without this, StrictBool would reject ``{"approved": 1}`` —
        a legitimate output from many LLMs, especially smaller/local models.
        Python's ``bool("false") == True`` means we can't accept strings,
        but rejecting ints is overly strict and causes silent critic failures.
        """
        if isinstance(v, bool):
            return v
        if isinstance(v, int):
            return bool(v)
        raise ValueError(f"approved must be bool or int, got {type(v).__name__}: {v!r}")


# ── Deterministic semantic validation ───────────────────────────────────────


def validate_opinion_semantics(
    *,
    attribution_types: Sequence[str],
    confidence: Mapping[str, float],
    evidence_citations: Sequence[str],
    argument: str,
) -> tuple[str, ...]:
    """Return the deterministic issues that disqualify an expert opinion.

    Pydantic can express shape but not meaning.  Phase 1.3 requires that a
    "valid expert opinion" has a non-empty argument, at least one legal
    procrastination type, a confidence for each type, and at least one evidence
    citation — with citations mandatory once a claim is confident.

    Returns:
        A tuple of Chinese issue strings; empty means the opinion is valid.
    """
    issues: list[str] = []

    if not argument.strip():
        issues.append("论据为空")

    if not attribution_types:
        issues.append("未给出合法拖延类型")

    missing_confidence = [t for t in attribution_types if t not in confidence]
    if missing_confidence:
        issues.append(f"缺少置信度: {', '.join(missing_confidence)}")

    for key, value in confidence.items():
        if not 0.0 <= float(value) <= 1.0:
            issues.append(f"置信度越界: {key}={value}")

    citations = [c for c in evidence_citations if str(c).strip()]
    if not citations:
        issues.append("缺少证据引用")

    return tuple(issues)


# ── Fence stripping & graceful parsing ──────────────────────────────────────


def _strip_markdown_fence(text: str) -> str:
    """Strip ```json ... ``` or ``` ... ``` Markdown fences from *text*."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[: -3].strip()
    return text


def parse_or_skip(
    raw: str,
    schema_class: type[BaseModel],
    context: str = "",
) -> BaseModel | None:
    """Parse *raw* JSON into *schema_class*, returning ``None`` on failure.

    Handles Markdown fence wrapping (`````json...`````) that LLMs sometimes
    add. Logs a warning on parse failure for observability.

    This is the primary entry point for replacing manual JSON parsing in
    ``orchestrator.py`` — call it instead of ``_safe_parse_json`` + manual
    field extraction.

    Args:
        raw: Raw LLM output string.
        schema_class: Pydantic model class to parse into.
        context: Context label for log messages (e.g., expert role name).

    Returns:
        An instance of *schema_class* on success, or ``None`` on failure.
    """
    text = _strip_markdown_fence(raw)

    try:
        return schema_class.model_validate_json(text)
    except Exception as exc:
        logger.warning(
            "Pydantic parse failed for {}: {}",
            context or schema_class.__name__,
            exc,
        )
        return None
