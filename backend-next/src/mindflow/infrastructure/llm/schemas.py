"""Pydantic v2 schemas for LLM attribution pipeline.

Defines the strict output contract for DeepSeek / Ollama responses and
the internal domain conversion helper.

NF-S7 double-check: the output forbidden-word validator ensures that
medical terminology ("诊断", "治疗", "患者", "处方") never leaks into
response_text — this is a second layer of defence after the system
prompt instructs the model to avoid them.

All schemas use Pydantic v2's ``model_validate_json`` strict mode for
maximum type safety (see llm-cbt.md §3 — format constraint alone is not
enough; semantic validation is required).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ── Literal type aliases ──────────────────────────────────────────────────────

PROCRASTINATION_TYPES = Literal[
    "task_aversion",
    "impulsivity",
    "decisional",
    "perfectionism",
    "emotional_regulation",
]

CBT_TECHNIQUES = Literal[
    "behavioral_experiment",
    "cognitive_restructuring",
    "stimulus_control",
    "goal_setting",
    "graded_exposure",
    "mindfulness",
]

# ── Forbidden words (NF-S7) ────────────────────────────────────────────────────

_FORBIDDEN_WORDS: frozenset[str] = frozenset({
    "诊断",
    "治疗",
    "患者",
    "处方",
})


# ── Output contract ────────────────────────────────────────────────────────────


class LLMAttributionResult(BaseModel):
    """Structured output from the LLM attribution pipeline.

    Parsed from DeepSeek / Ollama JSON response via ``model_validate_json``.
    All fields are validated at runtime — type errors, missing keys, and
    forbidden words raise ``ValidationError``.

    Fields follow the design in llm-cbt.md §3 (Pydantic output contract)
    with the addition of ``cognitive_distortions`` for richer insight.
    """

    procrastination_types: list[PROCRASTINATION_TYPES] = Field(
        ...,
        min_length=1,
        max_length=3,
        description="Detected procrastination types, sorted by confidence descending",
    )
    type_confidence: dict[str, float] = Field(
        ...,
        description="Per-type confidence scores in [0, 1]",
    )
    cognitive_distortions: list[str] = Field(
        default_factory=list,
        description="Identified cognitive distortions (e.g. all-or-nothing thinking)",
    )
    cbt_technique: CBT_TECHNIQUES = Field(
        ...,
        description="Recommended CBT intervention technique",
    )
    response_text: str = Field(
        ...,
        max_length=500,
        description="User-facing intervention text in Chinese, ≤500 characters",
    )
    next_action: str = Field(
        ...,
        description="Suggested next micro-action for the user",
    )

    # ── Validators ────────────────────────────────────────────────────

    @field_validator("response_text")
    @classmethod
    def _no_forbidden_words(cls, v: str) -> str:
        """Reject response_text containing forbidden medical terminology.

        Raises:
            ValueError: If any forbidden word appears in the text.
        """
        for word in _FORBIDDEN_WORDS:
            if word in v:
                msg = f"response_text contains forbidden word: {word!r} (NF-S7)"
                raise ValueError(msg)
        return v

    @field_validator("type_confidence")
    @classmethod
    def _confidence_keys_match_types(cls, v: dict[str, float], info: Any) -> dict[str, float]:
        """Ensure type_confidence keys are a superset of procrastination_types."""
        # Access other fields from validation context
        if isinstance(info.data, dict):
            types = info.data.get("procrastination_types", [])
            missing = [t for t in types if t not in v]
            if missing:
                msg = f"type_confidence missing keys for types: {missing}"
                raise ValueError(msg)
        return v

    @field_validator("type_confidence")
    @classmethod
    def _confidence_in_range(cls, v: dict[str, float]) -> dict[str, float]:
        """Ensure all confidence values are in [0, 1]."""
        for key, val in v.items():
            if not 0.0 <= val <= 1.0:
                msg = f"confidence for {key!r} must be in [0, 1], got {val}"
                raise ValueError(msg)
        return v

    model_config = {
        "extra": "forbid",
        "strict": True,
    }
