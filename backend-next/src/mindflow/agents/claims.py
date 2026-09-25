"""Claim Ledger — the validated unit of expert reasoning (phase 2.3).

Free-text expert arguments are hard to validate, hard to compress, and easy for
an LLM to pad.  The ledger replaces them with an explicit, code-checked
structure:

.. code-block:: json

    {
      "claims": [
        {
          "type": "impulsivity",
          "confidence": 0.68,
          "evidence_ids": ["summary.context_switches_per_hour"],
          "support": "频繁切换且最长专注块较短",
          "alternative": "可能是任务需要多应用协作"
        }
      ],
      "insufficient_data": false,
      "evidence_gaps": []
    }

Everything downstream (moderator prompt, conflict summary, critic) consumes
*validated* claims only.  The checks are deterministic and live in this module:

  1. every ``evidence_ids`` entry exists in the bundle catalog,
  2. every claim carries at least one piece of evidence,
  3. ``confidence`` is inside [0, 1],
  4. ``alternative`` states the main counter-explanation,
  5. multi-expert claims are compared for conflicts,
  6. high disagreement lowers confidence or forces abstention.

Design constraint: pure stdlib (mirrors ``agents/types.py``) so the ledger can
be used by the schema layer, the graph layer, and the offline evaluator without
pulling in pydantic.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from mindflow.agents.types import Claim, ClaimLedger, ExpertOpinion

__all__ = [
    "ABSTENTION_AGREEMENT_THRESHOLD",
    "DISAGREEMENT_CONFIDENCE_PENALTY",
    "HIGH_DISAGREEMENT_THRESHOLD",
    "Claim",
    "ClaimConflict",
    "ClaimLedger",
    "LedgerConflictSummary",
    "apply_disagreement_penalty",
    "claim_ledger_for",
    "confidence_ceiling",
    "guard_verdict_confidence",
    "ledger_from_claims_payload",
    "ledger_from_opinion",
    "render_claims_argument",
    "render_conflict_summary",
    "render_ledger_table",
    "summarize_conflicts",
    "validate_claim",
    "validate_ledger",
]

#: Agreement below this level counts as high disagreement.
HIGH_DISAGREEMENT_THRESHOLD: float = 0.5

#: Bounded digest length for a claim synthesised from a legacy free-text opinion.
#: The moderator table is a summary by design; the full prose stays in the
#: transcript (and in the critic's prompt), so truncating here keeps the promise
#: "the moderator receives validated claims, not several pages of prose".
LEGACY_SUPPORT_MAX_CHARS: int = 300

#: Confidence multiplier applied to claims when experts disagree strongly.
DISAGREEMENT_CONFIDENCE_PENALTY: float = 0.7

#: Below this agreement the panel should prefer abstention over a strong claim.
ABSTENTION_AGREEMENT_THRESHOLD: float = 0.34


@dataclass(frozen=True)
class ClaimConflict:
    """Conflict summary for one procrastination type across experts."""

    type: str
    supporters: tuple[str, ...]
    mean_confidence: float
    max_confidence: float
    min_confidence: float
    contested: bool


@dataclass(frozen=True)
class LedgerConflictSummary:
    """Aggregate conflict picture used by the moderator prompt."""

    conflicts: tuple[ClaimConflict, ...]
    agreement_strength: float
    abstained_roles: tuple[str, ...]

    @property
    def has_conflict(self) -> bool:
        return any(conflict.contested for conflict in self.conflicts)

    @property
    def high_disagreement(self) -> bool:
        return self.agreement_strength < HIGH_DISAGREEMENT_THRESHOLD


# ── Validation ──────────────────────────────────────────────────────────────


def validate_claim(
    claim: Claim,
    valid_evidence_ids: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Return the deterministic issues that disqualify *claim*.

    Args:
        claim: The claim to check.
        valid_evidence_ids: Catalog ids from the evidence bundle.  When None the
            existence check is skipped (schema-level validation only).

    Returns:
        Chinese issue strings; empty means the claim is valid.
    """
    issues: list[str] = []

    if not str(claim.type).strip():
        issues.append("claim 缺少类型")
    if not 0.0 <= float(claim.confidence) <= 1.0:
        issues.append(f"claim 置信度越界: {claim.type}={claim.confidence}")

    evidence_ids = [str(e).strip() for e in claim.evidence_ids if str(e).strip()]
    if not evidence_ids:
        issues.append(f"claim 缺少证据: {claim.type}")

    if valid_evidence_ids is not None:
        known = set(valid_evidence_ids)
        unknown = [e for e in evidence_ids if e not in known]
        if unknown:
            issues.append(f"claim 引用了不存在的证据: {', '.join(unknown)}")

    if not str(claim.support).strip():
        issues.append(f"claim 缺少论据: {claim.type}")
    if not str(claim.alternative).strip():
        issues.append(f"claim 缺少替代解释: {claim.type}")

    return tuple(issues)


def validate_ledger(
    ledger: ClaimLedger,
    valid_evidence_ids: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Return the deterministic issues that disqualify a whole ledger."""
    issues: list[str] = []

    for claim in ledger.claims:
        issues.extend(validate_claim(claim, valid_evidence_ids))

    if ledger.insufficient_data:
        if not [gap for gap in ledger.evidence_gaps if str(gap).strip()]:
            issues.append("insufficient_data=true 时必须提供 evidence_gaps")
    elif not ledger.claims:
        issues.append("未提供任何有效 claim")

    return tuple(issues)


# ── Construction ────────────────────────────────────────────────────────────


def _coerce_id_sequence(value: object) -> tuple[str, ...]:
    """Coerce a raw ``evidence_ids`` payload into a tuple of strings."""
    if value is None or isinstance(value, (str, bytes)):
        return (str(value),) if value else ()
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value)
    return (str(value),)


def ledger_from_claims_payload(payload: Sequence[Mapping[str, object]]) -> ClaimLedger:
    """Build a ledger from raw claim dicts (already schema-validated shape)."""
    claims: list[Claim] = []
    for item in payload:
        claims.append(
            Claim(
                type=str(item.get("type", "")).strip(),
                confidence=float(item.get("confidence", 0.0)),  # type: ignore[arg-type]
                evidence_ids=_coerce_id_sequence(item.get("evidence_ids")),
                support=str(item.get("support", "")),
                alternative=str(item.get("alternative", "")),
            )
        )
    return ClaimLedger(claims=tuple(claims))


def ledger_from_opinion(opinion: ExpertOpinion) -> ClaimLedger:
    """Synthesise an equivalent ledger from legacy opinion fields.

    Expert responses that still use the pre-ledger contract
    (``attribution_types`` + ``confidence`` + ``evidence_citations``) are mapped
    onto claims so every downstream consumer sees one representation.  The
    synthesised claim inherits the expert's argument as ``support`` and states
    the missing counter-explanation honestly instead of inventing one.
    """
    if opinion.claims:
        return ClaimLedger(claims=tuple(opinion.claims))

    claims: list[Claim] = []
    for ptype in opinion.attribution_types:
        confidence = float(opinion.confidence.get(ptype, 0.0))
        support = opinion.argument.strip() or f"{ptype}（未提供论据）"
        if len(support) > LEGACY_SUPPORT_MAX_CHARS:
            support = support[:LEGACY_SUPPORT_MAX_CHARS].rstrip() + "…"
        claims.append(
            Claim(
                type=ptype,
                confidence=confidence,
                evidence_ids=tuple(opinion.evidence_citations),
                support=support,
                alternative="未提供替代解释（旧格式输出）",
            )
        )
    return ClaimLedger(claims=tuple(claims))


def claim_ledger_for(opinion: ExpertOpinion) -> ClaimLedger:
    """Return the opinion's ledger, synthesising one when the expert omitted it."""
    return ledger_from_opinion(opinion)


# ── Conflict detection ──────────────────────────────────────────────────────


def summarize_conflicts(
    ledgers: Mapping[str, ClaimLedger],
    agreement_strength: float,
) -> LedgerConflictSummary:
    """Compare validated claims across experts.

    A type is *contested* when at least two experts claim it with materially
    different confidence (spread ≥ 0.2).  ``agreement_strength`` comes from the
    existing disagreement analysis so this module never re-invents that metric.
    """
    by_type: dict[str, list[tuple[str, float]]] = {}
    abstained: list[str] = []
    for role, ledger in ledgers.items():
        if ledger.insufficient_data or not ledger.claims:
            abstained.append(role)
        for claim in ledger.claims:
            by_type.setdefault(claim.type, []).append((role, float(claim.confidence)))

    conflicts: list[ClaimConflict] = []
    for ptype, entries in sorted(by_type.items()):
        confidences = [value for _, value in entries]
        supporters = tuple(role for role, _ in entries)
        spread = max(confidences) - min(confidences) if confidences else 0.0
        conflicts.append(
            ClaimConflict(
                type=ptype,
                supporters=supporters,
                mean_confidence=round(sum(confidences) / len(confidences), 4),
                max_confidence=max(confidences),
                min_confidence=min(confidences),
                contested=len(entries) > 1 and spread >= 0.2,
            )
        )

    return LedgerConflictSummary(
        conflicts=tuple(conflicts),
        agreement_strength=float(agreement_strength),
        abstained_roles=tuple(abstained),
    )


# ── Disagreement handling ───────────────────────────────────────────────────


def apply_disagreement_penalty(
    ledger: ClaimLedger,
    agreement_strength: float,
) -> ClaimLedger:
    """Lower claim confidence (or abstain) when experts disagree strongly.

    Mirrors the moderator instruction "共识强度低时请降低置信度，或设置
    insufficient_data=true", but applies it in code so a persuasive-looking LLM
    cannot keep a high confidence the panel did not earn.
    """
    if not ledger.claims:
        return ledger

    if agreement_strength <= ABSTENTION_AGREEMENT_THRESHOLD:
        gaps = tuple(ledger.evidence_gaps) or ("专家共识强度过低，无法形成稳定结论",)
        return ClaimLedger(claims=(), insufficient_data=True, evidence_gaps=gaps)

    if agreement_strength >= HIGH_DISAGREEMENT_THRESHOLD:
        return ledger

    penalised = tuple(
        replace(claim, confidence=round(claim.confidence * DISAGREEMENT_CONFIDENCE_PENALTY, 4))
        for claim in ledger.claims
    )
    return ClaimLedger(
        claims=penalised,
        insufficient_data=ledger.insufficient_data,
        evidence_gaps=ledger.evidence_gaps,
    )


def confidence_ceiling(agreement_strength: float) -> float:
    """Highest confidence a verdict may claim at this agreement level.

    Full agreement (``agreement_strength >= HIGH_DISAGREEMENT_THRESHOLD``) puts
    no ceiling on the moderator; weaker agreement caps the claimable confidence
    at the panel's actual agreement, so a low-consensus verdict cannot be
    reported with high confidence.
    """
    if agreement_strength >= HIGH_DISAGREEMENT_THRESHOLD:
        return 1.0
    return max(0.0, min(1.0, float(agreement_strength)))


def should_abstain(agreement_strength: float) -> bool:
    """True when the panel's agreement is too low to stand behind any verdict."""
    return agreement_strength <= ABSTENTION_AGREEMENT_THRESHOLD


def guard_verdict_confidence(
    verdict: dict[str, Any],
    agreement_strength: float,
) -> dict[str, Any]:
    """Apply the panel's actual agreement to a moderator verdict, in code.

    The moderator prompt tells the model to lower confidence (or abstain) when
    consensus is weak; a persuasive-looking answer must not be able to ignore
    that.  This mirrors :func:`apply_disagreement_penalty` at the verdict level:

      * agreement ≥ ``HIGH_DISAGREEMENT_THRESHOLD`` → unchanged;
      * ``ABSTENTION_AGREEMENT_THRESHOLD`` < agreement < threshold → confidences
        are multiplied by ``DISAGREEMENT_CONFIDENCE_PENALTY`` and the reason is
        appended to ``dissent``;
      * agreement ≤ ``ABSTENTION_AGREEMENT_THRESHOLD`` → the verdict is marked
        ``insufficient_data`` (with the reason in ``evidence_gaps``) and every
        confidence is capped at the agreement level.

    Returns a new dict; the input is not mutated.
    """
    guarded = dict(verdict)
    confidence = {
        str(k): float(v)
        for k, v in dict(guarded.get("confidence") or {}).items()
    }

    if agreement_strength >= HIGH_DISAGREEMENT_THRESHOLD:
        return guarded

    if agreement_strength <= ABSTENTION_AGREEMENT_THRESHOLD:
        guarded["insufficient_data"] = True
        gaps = [str(gap) for gap in (guarded.get("evidence_gaps") or [])]
        note = (
            f"专家共识强度过低（agreement={agreement_strength:.2f}），"
            "自动弃权，不给出高置信度结论"
        )
        if note not in gaps:
            gaps.append(note)
        guarded["evidence_gaps"] = gaps
        ceiling = confidence_ceiling(agreement_strength)
        guarded["confidence"] = {
            key: round(min(value, ceiling), 4) for key, value in confidence.items()
        }
        return guarded

    penalised = {
        key: round(value * DISAGREEMENT_CONFIDENCE_PENALTY, 4)
        for key, value in confidence.items()
    }
    guarded["confidence"] = penalised
    dissent = [str(item) for item in (guarded.get("dissent") or [])]
    note = f"共识强度偏低（agreement={agreement_strength:.2f}），置信度已按代码规则下调"
    if note not in dissent:
        dissent.append(note)
    guarded["dissent"] = dissent
    return guarded


# ── Rendering for the moderator prompt ──────────────────────────────────────


def render_claims_argument(ledger: ClaimLedger) -> str:
    """Render a ledger as the legacy argument text (with ``[证据: id]`` markers).

    Used when a claim-based response has to populate the free-text
    ``ExpertOpinion.argument`` field that the transcript, critic and moderator
    summaries still display.
    """
    lines: list[str] = []
    for claim in ledger.claims:
        cites = "".join(f" [证据: {eid}]" for eid in claim.evidence_ids)
        lines.append(
            f"- {claim.type}（置信度 {claim.confidence:.2f}）：{claim.support}"
            f"；替代解释：{claim.alternative}{cites}"
        )
    if ledger.insufficient_data:
        gaps = "；".join(ledger.evidence_gaps) or "未说明"
        lines.append(f"证据不足，弃权：{gaps}")
    return "\n".join(lines)


def render_ledger_table(ledgers: Mapping[str, ClaimLedger]) -> str:
    """Render validated claims as a compact table for the moderator prompt.

    The moderator receives *only* this table (plus the evidence table and the
    conflict summary) instead of every expert's uncompressed prose.
    """
    lines: list[str] = ["| 专家 | 类型 | 置信度 | 证据 | 论据 | 替代解释 |"]
    lines.append("|---|---|---|---|---|---|")
    for role, ledger in ledgers.items():
        if ledger.insufficient_data:
            gaps = "；".join(ledger.evidence_gaps) or "未说明"
            lines.append(f"| {role} | （弃权） | - | - | 证据不足 | {gaps} |")
            continue
        for claim in ledger.claims:
            evidence = ", ".join(claim.evidence_ids)
            support = claim.support.replace("\n", " ").strip()
            alternative = claim.alternative.replace("\n", " ").strip()
            lines.append(
                f"| {role} | {claim.type} | {claim.confidence:.2f} | {evidence} "
                f"| {support} | {alternative} |"
            )
    return "\n".join(lines)


def render_conflict_summary(summary: LedgerConflictSummary) -> str:
    """Render the multi-expert conflict picture for the moderator prompt."""
    lines = [f"agreement_strength={summary.agreement_strength:.3f}"]
    if summary.abstained_roles:
        lines.append("弃权专家：" + "、".join(summary.abstained_roles))
    if not summary.conflicts:
        lines.append("无跨专家 claim")
    for conflict in summary.conflicts:
        marker = "冲突" if conflict.contested else "一致"
        lines.append(
            f"- {conflict.type}: {marker}（支持者 {len(conflict.supporters)}，"
            f"均值 {conflict.mean_confidence:.2f}，区间 "
            f"{conflict.min_confidence:.2f}-{conflict.max_confidence:.2f}）"
        )
    if summary.high_disagreement:
        lines.append("分歧较大：请降低置信度，或输出 insufficient_data=true 并说明证据缺口。")
    return "\n".join(lines)
