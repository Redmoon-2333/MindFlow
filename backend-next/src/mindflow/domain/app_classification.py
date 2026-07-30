"""User-customizable app classification with context awareness.

``UserAppClassifier`` wraps the built-in ``AppClassifier`` with two
additional tiers:

  Tier 0 — User-defined rules from ``app_classification_rules`` table
           (checked first, priority-ordered, per-process-name + optional
           window-title pattern matching).

  Bilibili special case — If the window title matches productive learning
           patterns (e.g., "高等数学第3讲"), classify as ``browser_work``
           even though bilibili would otherwise be ``entertainment``.

  Tiers 1–5 — Fall through to the standard ``AppClassifier``.

User rules always take precedence over all heuristics — if a user
explicitly classifies an app, that classification is final.

Design: domain layer.  No framework or I/O dependencies beyond the async
repository interface.
"""

from __future__ import annotations

import re
from typing import Any, Protocol, cast

from mindflow.domain.classifier import AppClassifier
from mindflow.domain.features import title_features

# ── Repository protocol (avoids circular imports) ────────────────────────────


class ClassificationRulesProtocol(Protocol):
    """Protocol for ``AppClassificationRulesRepository`` — async rule lookup.

    Only the methods this classifier actually needs are declared here.
    """

    async def get_all(self, user_id: int) -> list[dict[str, Any]]: ...


# ── Public API ────────────────────────────────────────────────────────────────

_VALID_CATEGORIES: frozenset[str] = frozenset({
    "code",
    "document",
    "browser_work",
    "communication",
    "entertainment",
    "social",
    "other",
})

_PRODUCTIVITY_SCORES: dict[str, float] = {
    "code": 1.0,
    "document": 1.0,
    "browser_work": 1.0,
    "communication": 0.5,
    "entertainment": 0.0,
    "social": 0.0,
    "other": 0.3,
}


class UserAppClassifier:
    """User-customizable app classifier with rule- and context-awareness.

    Args:
        rules_repo: Repository for loading user classification rules.
            Must expose ``get_all(user_id) -> list[dict]``.
        base: Fallback classifier.  Defaults to ``AppClassifier()``.
    """

    def __init__(
        self,
        rules_repo: ClassificationRulesProtocol | None = None,
        base: AppClassifier | None = None,
    ) -> None:
        self._rules_repo = rules_repo
        self._base = base or AppClassifier()

    # ── classify ─────────────────────────────────────────────────────────

    async def classify(
        self,
        process_name: str,
        window_title: str,
        *,
        user_id: int = 1,
    ) -> str:
        """Classify an app by process name and window title.

        Resolution order:
          1. User-defined rules (exact process name + optional title pattern).
          2. Bilibili productive-learning heuristic (window title).
          3. Built-in ``AppClassifier``.

        Args:
            process_name: Executable / process name (e.g. ``"bilibili.exe"``).
            window_title: Active window title text.
            user_id: User identifier for rule lookup.

        Returns:
            One of the seven valid category strings.
        """
        # ── Tier 0: User rules (first match wins) ──────────────────────
        if self._rules_repo is not None:
            rules = await self._rules_repo.get_all(user_id)
            for rule in rules:
                if self._rule_matches(rule, process_name, window_title):
                    return cast(str, rule["category"])

        # ── Bilibili special case ──────────────────────────────────────
        tf = title_features(window_title)
        if tf.is_likely_productive_learning:
            return "browser_work"

        # ── Tier 1-5: Built-in classifier ──────────────────────────────
        return self._base.classify(process_name, window_title)

    # ── classify_sync (for non-async callers / training pipeline) ──────

    def classify_sync(
        self,
        process_name: str,
        window_title: str,
    ) -> str:
        """Synchronous classification — skips user rules, uses only
        productive-learning heuristic + built-in classifier.

        Intended for offline / training-pipeline use where async I/O is
        unavailable (e.g., ``BehaviorFeatureExtractor``).
        """
        tf = title_features(window_title)
        if tf.is_likely_productive_learning:
            return "browser_work"
        return self._base.classify(process_name, window_title)

    # ── Rule matching ──────────────────────────────────────────────────

    @staticmethod
    def _rule_matches(
        rule: dict[str, Any],
        process_name: str,
        window_title: str,
    ) -> bool:
        """Return True if *rule* matches the given process and title.

        Rules match on **exact** process name (case-insensitive).  An
        optional ``window_title_pattern`` supports SQL ``LIKE``-style
        wildcards (``%``) for partial title matching.
        """
        pname = process_name.lower().strip()
        rule_proc = str(rule.get("process_name", "")).lower().strip()
        if pname != rule_proc:
            return False

        pattern = rule.get("window_title_pattern")
        if pattern is None or pattern == "":
            return True  # rule matches any title for this process

        # SQL LIKE-style wildcard: % → .*
        # re.escape does not escape % in Python ≥ 3.7, so replace % directly.
        regex = re.escape(pattern).replace("%", ".*")
        return bool(re.search(regex, window_title, re.IGNORECASE))

    # ── Productivity score ─────────────────────────────────────────────

    @staticmethod
    def get_productivity_score(category: str) -> float:
        """Return 0.0–1.0 productivity score for a given category.

        Mirrors ``AppClassifier.get_productivity_score()``.
        """
        return _PRODUCTIVITY_SCORES.get(category, 0.3)
