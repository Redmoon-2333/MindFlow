"""Observed task context derived from app/domain classification (schema v4).

The v4 feature window carries a *task context* dimension: which kind of work the
user was doing during a window, inferred from what was in the foreground.  The
resolution chain is deliberately three-layered so no app-name list is ever
hard-coded in the feature builder:

1. ``process/domain`` → the existing user-editable classification rules
   (``app_classification_rules`` via :class:`UserAppClassifier`, plus the
   built-in :class:`AppClassifier` heuristics) → one of the seven *activity*
   categories (``code``, ``document``, ``browser_work``, ``communication``,
   ``entertainment``, ``social``, ``other``).
2. activity category / browser domain → **this module's** small user-editable
   mapping layer (``task_context_rules``) → one of the seven *task contexts*.
3. the resolved per-context durations → the v4 feature columns
   (``task_type_code``, ``task_type_entropy``, ``task_type_dominant_ratio``,
   ``task_context_transition``, ``task_unknown_ratio``).

Why a second mapping layer instead of more categories in
``app_classification_rules``: the activity vocabulary cannot express the
reading/writing split the task context needs, and classifying an app is a
different question from knowing which task it was serving.  The mapping layer
is persisted exactly like the app rules (same table shape, same
priority/ordering semantics, same REST surface).

Privacy: only the *host* of a domain is ever stored or matched — a pasted
``https://docs.python.org/3/library/asyncio.html`` is reduced to
``docs.python.org`` by :func:`normalize_domain_rule`, so no URL path or query
can enter the rule store or a feature window.  Pure stdlib: no framework, no
I/O, no numpy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

# ── Vocabulary ──────────────────────────────────────────────────────────────

TASK_CONTEXT_UNKNOWN = "unknown"

#: The observed task-context vocabulary.  ``unknown`` is a first-class value:
#: an unclassified window reports it instead of a fabricated category.
TASK_CONTEXT_CATEGORIES: tuple[str, ...] = (
    "coding",
    "writing",
    "reading",
    "meeting",
    "entertainment",
    "social",
    "unknown",
)

#: Categorical codes for the ``task_type_code`` feature column.  Codes start at
#: 1 so ``0.0`` stays reserved for "no value at all" (a legacy payload without
#: the column); ``unknown`` is 7.0, never 0.0, so an unclassified window cannot
#: be mistaken for a missing column.
TASK_TYPE_CODES: Mapping[str, float] = {
    name: float(index) for index, name in enumerate(TASK_CONTEXT_CATEGORIES, start=1)
}

#: Reverse map used when reading a stored window back (transition tracking).
TASK_TYPE_CODE_TO_CONTEXT: Mapping[float, str] = {
    code: name for name, code in TASK_TYPE_CODES.items()
}

#: Default activity-category → task-context mapping.  These are *category*
#: semantics (not app names), and every entry is overridable by a user rule.
DEFAULT_CONTEXT_BY_CATEGORY: Mapping[str, str] = {
    "code": "coding",
    "document": "writing",
    "browser_work": "reading",
    "communication": "meeting",
    "entertainment": "entertainment",
    "social": "social",
    "other": TASK_CONTEXT_UNKNOWN,
}

MATCH_TYPE_CATEGORY = "category"
MATCH_TYPE_DOMAIN = "domain"
MATCH_TYPES: tuple[str, ...] = (MATCH_TYPE_CATEGORY, MATCH_TYPE_DOMAIN)

#: Activity categories a ``category`` rule may target (kept in sync with
#: ``app_classification._VALID_CATEGORIES`` / the app-classification API).
CLASSIFICATION_CATEGORIES: tuple[str, ...] = (
    "code",
    "document",
    "browser_work",
    "communication",
    "entertainment",
    "social",
    "other",
)

_CONTEXT_ALIASES: Mapping[str, str] = {
    "code": "coding",
    "coding": "coding",
    "dev": "coding",
    "development": "coding",
    "programming": "coding",
    "doc": "writing",
    "document": "writing",
    "documents": "writing",
    "write": "writing",
    "writing": "writing",
    "read": "reading",
    "reading": "reading",
    "research": "reading",
    "study": "reading",
    "studying": "reading",
    "call": "meeting",
    "communication": "meeting",
    "meet": "meeting",
    "meeting": "meeting",
    "meetings": "meeting",
    "entertainment": "entertainment",
    "fun": "entertainment",
    "game": "entertainment",
    "gaming": "entertainment",
    "media": "entertainment",
    "chat": "social",
    "social": "social",
    "socializing": "social",
    "": TASK_CONTEXT_UNKNOWN,
    "none": TASK_CONTEXT_UNKNOWN,
    "other": TASK_CONTEXT_UNKNOWN,
    "unknown": TASK_CONTEXT_UNKNOWN,
}


class TaskContextRulesProtocol(Protocol):
    """Async lookup for the user's task-context mapping rules.

    Mirrors ``app_classification.ClassificationRulesProtocol`` so both rule
    stores are consumed the same way by the telemetry rollup.
    """

    async def get_all(self, user_id: int) -> list[dict[str, Any]]: ...


# ── Normalisation ───────────────────────────────────────────────────────────


def normalize_task_context(value: object) -> str:
    """Coerce *value* to a known task context.

    Anything unrecognised — including ``None`` and empty strings — becomes
    ``unknown``.  A value is never invented, and an unrecognised value is never
    silently mapped to a real category.
    """
    if value is None:
        return TASK_CONTEXT_UNKNOWN
    text = str(value).strip().lower()
    return _CONTEXT_ALIASES.get(text, TASK_CONTEXT_UNKNOWN)


def is_task_context(value: object) -> bool:
    """True when *value* names a task context (no alias guessing)."""
    return isinstance(value, str) and str(value).strip().lower() in TASK_CONTEXT_CATEGORIES


def normalize_domain_rule(value: object) -> str:
    """Return the bare host for a domain rule value.

    Accepts a bare host, a wildcard-prefixed host or a full URL, and always
    returns host-only text: ``https://docs.python.org/3/library/x.html`` →
    ``docs.python.org``.  Schemas and paths are dropped rather than stored, so
    a pasted URL cannot smuggle private path data into the rule table or into
    a feature window.  Returns ``""`` for anything without a usable host.
    """
    raw = "" if value is None else str(value).strip().lower()
    if not raw:
        return ""
    if raw.startswith("*."):
        raw = raw[2:]
    parsed = urlsplit(raw if "://" in raw else f"https://{raw}")
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return ""
    return host.removeprefix("www.")[:253]


# ── Rule resolution ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TaskContextRule:
    """One normalised mapping rule (``category`` or ``domain`` → context)."""

    match_type: str
    match_value: str
    task_context: str
    priority: int = 0

    @classmethod
    def from_mapping(cls, rule: Mapping[str, Any]) -> TaskContextRule | None:
        """Build a rule from a stored row; ``None`` when the row is unusable.

        Stored rows are data, not code: an invalid row is skipped instead of
        raising, so one bad edit cannot make the rollup fail.
        """
        match_type = str(rule.get("match_type", "")).strip().lower()
        if match_type == MATCH_TYPE_DOMAIN:
            value = normalize_domain_rule(rule.get("match_value"))
            if not value:
                return None
        elif match_type == MATCH_TYPE_CATEGORY:
            value = str(rule.get("match_value", "")).strip().lower()
            if value not in CLASSIFICATION_CATEGORIES:
                return None
        else:
            return None
        try:
            priority = int(rule.get("priority", 0) or 0)
        except (TypeError, ValueError):
            priority = 0
        return cls(
            match_type=match_type,
            match_value=value,
            task_context=normalize_task_context(rule.get("task_context")),
            priority=priority,
        )


class TaskContextMapper:
    """Resolve activity category / browser domain → observed task context.

    Args:
        rules: User rules in *priority order* (highest first — the repository
            already returns them that way).  Invalid rows are ignored.
    """

    def __init__(self, rules: Sequence[Mapping[str, Any]] | None = None) -> None:
        parsed: list[TaskContextRule] = []
        for rule in rules or []:
            normalised = TaskContextRule.from_mapping(rule)
            if normalised is not None:
                parsed.append(normalised)
        self._rules: tuple[TaskContextRule, ...] = tuple(parsed)

    @property
    def rules(self) -> tuple[TaskContextRule, ...]:
        """The normalised user rules, in resolution order."""
        return self._rules

    def has_rules(self) -> bool:
        """True when at least one usable user rule is configured."""
        return bool(self._rules)

    def context_for_category(self, category: object) -> str:
        """Map an activity category through user rules, then defaults."""
        key = str(category or "").strip().lower()
        for rule in self._rules:
            if rule.match_type == MATCH_TYPE_CATEGORY and rule.match_value == key:
                return rule.task_context
        return DEFAULT_CONTEXT_BY_CATEGORY.get(key, TASK_CONTEXT_UNKNOWN)

    def context_for_domain(self, domain: object) -> str | None:
        """Map a browser domain through user rules; ``None`` when unmatched.

        Domain rules match the host itself and its subdomains: a rule for
        ``example.com`` matches ``example.com`` and ``docs.example.com``.
        """
        host = normalize_domain_rule(domain)
        if not host:
            return None
        for rule in self._rules:
            if rule.match_type != MATCH_TYPE_DOMAIN:
                continue
            if host == rule.match_value or host.endswith(f".{rule.match_value}"):
                return rule.task_context
        return None

    def resolve(self, category: object, domain: object = "") -> str:
        """Resolve one observation: domain rule first, then category mapping.

        Args:
            category: Activity category of the foreground process (or of the
                browser process when only a domain is known).
            domain: Browser domain, when the observation came from a browser
                heartbeat.
        """
        domain_context = self.context_for_domain(domain)
        if domain_context is not None:
            return domain_context
        return self.context_for_category(category)


# ── Per-window summary ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TaskContextSummary:
    """Task-context distribution of one window.

    Attributes:
        dominant: Context with the largest share (``unknown`` when nothing was
            observed or when every observation failed to classify).
        entropy: Shannon entropy of the observed distribution normalised by
            ``log(len(TASK_CONTEXT_CATEGORIES))`` → ``0.0`` (one context) to
            ``1.0`` (perfectly even spread across all seven).  The ``unknown``
            bucket participates: a window whose observations are half
            unclassifiable is genuinely less certain about its task context.
        dominant_ratio: Share of observed seconds held by *dominant*
            (``0.0`` when nothing was observed).
        unknown_ratio: Share of observed seconds that could not be classified
            — the unknown/missing indicator.  ``1.0`` when nothing was observed
            at all (missing is not the same as classified-as-safe), ``0.0``
            when every observed second was classified.
        total_seconds: Observed seconds the shares are computed over.
    """

    dominant: str
    entropy: float
    dominant_ratio: float
    unknown_ratio: float
    total_seconds: float


def summarize_task_context(durations: Mapping[str, float]) -> TaskContextSummary:
    """Summarise a ``{task_context: seconds}`` distribution.

    Non-positive and unrecognised entries are folded into ``unknown``; missing
    contexts simply have a zero share.  Ties for the dominant context are
    broken by the vocabulary order, which puts ``unknown`` last, so a real
    category wins any tie it is part of.
    """
    totals: dict[str, float] = dict.fromkeys(TASK_CONTEXT_CATEGORIES, 0.0)
    for name, raw_seconds in durations.items():
        try:
            seconds = float(raw_seconds)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(seconds) or seconds <= 0.0:
            continue
        totals[normalize_task_context(name)] += seconds

    total = math.fsum(totals.values())
    if total <= 0.0:
        return TaskContextSummary(
            dominant=TASK_CONTEXT_UNKNOWN,
            entropy=0.0,
            dominant_ratio=0.0,
            unknown_ratio=1.0,
            total_seconds=0.0,
        )

    entropy = -math.fsum(
        (share := seconds / total) * math.log(share)
        for seconds in totals.values()
        if seconds > 0.0
    )
    normaliser = math.log(len(TASK_CONTEXT_CATEGORIES))
    dominant = max(TASK_CONTEXT_CATEGORIES, key=lambda name: totals[name])
    return TaskContextSummary(
        dominant=dominant,
        entropy=max(0.0, min(1.0, entropy / normaliser)),
        dominant_ratio=totals[dominant] / total,
        unknown_ratio=totals[TASK_CONTEXT_UNKNOWN] / total,
        total_seconds=total,
    )
