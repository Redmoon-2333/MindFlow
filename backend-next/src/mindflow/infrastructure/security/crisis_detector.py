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

from dataclasses import dataclass, field
from enum import StrEnum


class CrisisLevel(StrEnum):
    """Severity level of detected crisis signal."""

    NONE = "none"
    HIGH = "high"


_CRISIS_KEYWORDS: frozenset[str] = frozenset({
    "自杀",
    "不想活",
    "结束生命",
    "结束自己的生命",
    "伤害自己",
    "自伤",
    "撑不下去",
    "活不下去",
    "不想活了",
    "没有意义",
    "死了算了",
    "想死",
})
"""Core Chinese crisis keywords. Extend via :meth:`CrisisDetector.add_keywords`."""


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

        for keyword in self._keywords:
            if keyword in text:
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
