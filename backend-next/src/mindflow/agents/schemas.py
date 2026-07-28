"""Pydantic v2 structured output schemas for the expert panel pipeline.

Replaces the ~200 lines of manual JSON parsing in ``orchestrator.py``
(``_parse_expert_opinion``, ``_parse_analyst_opinion``, ``_parse_verdict``,
``_parse_critic``) with Pydantic ``model_validate_json()`` calls.

Each schema matches the JSON contract that the corresponding expert LLM is
prompted to produce. CRITICAL: ``CriticOutput.approved`` is a strict boolean
— Pydantic correctly parses JSON ``true``/``false``, fixing the bug where
``bool("false") == True`` in the current manual parsing code.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import BaseModel, Field, field_validator


class AnalystOutput(BaseModel):
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
    evidence_citations: list[str] = Field(
        default_factory=list,
        description="Evidence metric IDs cited",
    )


class AttributionOutput(BaseModel):
    """归因专家结构化输出（CBT/TMT/情绪）。

    Each attribution expert identifies 1-3 procrastination types and provides
    reasoning with evidence citations.
    """

    attribution_types: list[str] = Field(
        default_factory=list,
        description="1-3 procrastination types identified",
    )
    confidence: dict[str, float] = Field(
        default_factory=dict,
        description="Per-type confidence scores 0-1",
    )
    argument: str = Field(
        default="",
        description="Full reasoning in Chinese with [证据: metric] citations",
    )
    evidence_citations: list[str] = Field(
        default_factory=list,
        description="Evidence metric IDs cited in argument",
    )


class ModeratorOutput(BaseModel):
    """主持人/综合者结构化输出。

    The moderator synthesizes all expert opinions into a final verdict.
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


class CriticOutput(BaseModel):
    """批评家结构化输出。

    CRITICAL: ``approved`` 是严格布尔值 — Pydantic 正确解析 JSON
    ``true``/``false``，修复了当前代码中 ``bool("false") == True``
    的 bug。
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
