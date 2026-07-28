"""Independent crisis keyword detector — runs before any LLM call.

Pure rule-based, zero LLM, zero network. Scans input text for Chinese crisis
keywords and returns a CrisisLevel + CrisisResponse when a match is found.

This is the safety gate required by NF-S7b (crisis detection independent of LLM)
and California SB 243 / Illinois HB 1806 compliance.

Design:
  - Keyword set is a frozen set of Chinese crisis phrases compiled at import
    time for O(1) membership check per word.
  - Only whole-word substring matching (no regex — simpler, cheaper).
  - HIGH triggers a hard stop: LLM call is skipped entirely, a fixed crisis
    response with national hotline info is returned, and the incident is logged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from mindflow.domain.forbidden_words import CRISIS_KEYWORDS


def _normalize_whitespace(text: str) -> str:
    """Remove all whitespace from *text* for bypass-resistant scanning.

    P2-3: normalizing whitespace prevents simple spacing/line-break evasion
    (e.g. ``"割 腕"`` → ``"割腕"``, ``"自\\n杀"`` → ``"自杀"``).
    """
    return re.sub(r"\s+", "", text)


class CrisisLevel(StrEnum):
    """Severity level of detected crisis signal."""

    NONE = "none"
    HIGH = "high"


_CRISIS_KEYWORDS: frozenset[str] = CRISIS_KEYWORDS
"""Core Chinese crisis keywords — canonical import from domain/forbidden_words.

Extend via :meth:`CrisisDetector.add_keywords`.
"""


@dataclass(frozen=True)
class CrisisResponse:
    """Response payload when a HIGH crisis signal is detected.

    Attributes:
        message: Crisis hotline information in Chinese.
        stop_llm: If True, the LLM call must be skipped entirely.
    """

    message: str = field(
        default=(
            "看到你正在经历困难时刻。请记住你并不孤单——"
            "全国24小时心理援助热线：400-161-9995 或 010-82951332"
            "（北京心理危机研究与干预中心）。"
            "请立即寻求专业帮助。"
        )
    )
    stop_llm: bool = True


class CrisisDetector:
    """Rule-based crisis keyword scanner.

    Thread-safe (immutable state after construction). Keyword additions via
    *add_keywords* create a new frozen set and are not thread-safe — intended
    for single-thread configuration at startup.

    Usage::

        detector = CrisisDetector()
        result = detector.scan("我感觉撑不下去了")
        # → CrisisLevel.HIGH, CrisisResponse

        result = detector.scan("今天有点累")
        # → CrisisLevel.NONE, None
    """

    def __init__(self, extra_keywords: frozenset[str] | None = None) -> None:
        """Initialise detector with optional additional keywords.

        Args:
            extra_keywords: Additional crisis keywords to merge into the
                built-in set.  Each keyword is a Chinese string matched as
                a substring against input text.
        """
        all_kw = _CRISIS_KEYWORDS
        if extra_keywords:
            all_kw = all_kw | extra_keywords
        self._keywords: frozenset[str] = all_kw

    @property
    def keywords(self) -> frozenset[str]:
        """Return the current keyword set (immutable)."""
        return self._keywords

    def add_keywords(self, extra: frozenset[str]) -> None:
        """Extend the keyword set at runtime (startup-only usage).

        Creates a new frozen set — not thread-safe but suitable for
        one-time configuration during service startup.
        """
        self._keywords = self._keywords | extra

    def scan(self, text: str) -> tuple[CrisisLevel, CrisisResponse | None]:
        """Scan *text* for crisis keywords.

        P2-3: whitespace is normalised from *text* before scanning so
        spacing/line-break evasion (``"割 腕"``, ``"跳   楼"``,
        ``"自\\n杀"``) is detected.  Empty or whitespace-only text
        returns NONE immediately.

        Args:
            text: The input text to scan (e.g. manual_tag content,
                  intended_task description). Empty or whitespace-only
                  text returns NONE immediately.

        Returns:
            A tuple of (CrisisLevel, CrisisResponse | None).
            CrisisLevel.HIGH implies a non-None CrisisResponse.
        """
        if not text or not text.strip():
            return CrisisLevel.NONE, None

        normalised = _normalize_whitespace(text)

        for keyword in self._keywords:
            if keyword in normalised:
                return CrisisLevel.HIGH, CrisisResponse()

        return CrisisLevel.NONE, None

    def scan_texts(self, texts: list[str]) -> tuple[CrisisLevel, CrisisResponse | None]:
        """Scan multiple text fields, short-circuiting on first match.

        Args:
            texts: List of text strings to scan. Empty list returns NONE.

        Returns:
            Same shape as :meth:`scan`.
        """
        for t in texts:
            level, response = self.scan(t)
            if level == CrisisLevel.HIGH:
                return level, response
        return CrisisLevel.NONE, None
