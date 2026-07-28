"""Canonical forbidden-word and crisis-keyword constants.

Single source of truth for NF-S7 compliance terms and crisis detection keywords.
Import this module from schemas, agents, safety_guard, and crisis_detector.

Design (P1-2):
  - ``FORBIDDEN_MEDICAL_TERMS``: the canonical 4 medical terms (NF-S7 contract).
    Safety guard adds 8 more locally for a total effective set of 12.
  - ``CRISIS_KEYWORDS``: 28-term centralized crisis set (15 legacy union + 13
    P2-3 suggestions). Whitespace normalization applied at scan time.
"""

from __future__ import annotations

# ── NF-S7 Medical Terminology (canonical 4) ──────────────────────────────────────

FORBIDDEN_MEDICAL_TERMS: frozenset[str] = frozenset({
    "诊断",
    "治疗",
    "患者",
    "处方",
})
"""The canonical 4-term set.

Schemas and agents import exactly these 4 terms. Safety guard extends this
locally with 8 more terms for a total of 12::

    _FORBIDDEN_MEDICAL = FORBIDDEN_MEDICAL_TERMS | frozenset({
        "药物", "剂量", "复诊", "挂号", "住院", "手术", "服药", "副作用",
    })
"""

# ── Crisis Keywords (union of safety_guard + CrisisDetector + P2-3 suggestions) ──

CRISIS_KEYWORDS: frozenset[str] = frozenset({
    # From safety_guard._FORBIDDEN_CRISIS (8 terms)
    "自杀",
    "自残",
    "伤害自己",
    "不想活",
    "活不下去",
    "结束生命",
    "一了百了",
    "轻生",
    # From CrisisDetector._CRISIS_KEYWORDS — unique additions (7 terms)
    "结束自己的生命",
    "自伤",
    "撑不下去",
    "不想活了",
    "没有意义",
    "死了算了",
    "想死",
    # P2-3: genuinely new suggested terms (13 terms, 4 of 17 already above)
    "活够了",
    "活腻了",
    "割腕",
    "割脉",
    "跳楼",
    "服毒",
    "上吊",
    "遗书",
    "救救我",
    "我受不了了",
    "太累了",
    "走投无路",
    "安眠药",
})
"""Union of safety_guard, CrisisDetector crisis keywords, and P2-3 suggestions.

28 unique terms total: 15 legacy union + 13 genuinely new from 17 plan suggestions
(4 of the 17 — 自残, 一了百了, 自伤, 撑不下去 — already existed in the legacy set).

Both safety_guard and CrisisDetector import this set as their crisis detection
vocabulary. Whitespace normalization is applied at scan time (not keyword level).
"""
