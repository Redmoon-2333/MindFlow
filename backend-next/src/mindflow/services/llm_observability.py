"""Aggregated, PII-free LLM request observability (optimisation plan 2.5).

Every LLM request — panel expert, moderator, critic, chat, attribution, Ollama
fallback — lands here as one :class:`LLMRequestRecord`, and the aggregator turns
those records into per-role and per-graph p50 / p95 / failure-rate / cost
statistics.

Privacy contract (ADR-003), enforced by construction:

* ``LLMRequestRecord`` has **only** scalar metadata fields — there is no field
  that could hold a prompt, a provider body, an API key, a completion, or user
  text. Constructing one with an unknown field is a ``TypeError``.
* Label fields are whitespace-collapsed and truncated (``_TEXT_LIMIT``), so even
  a careless caller cannot park free text in ``node`` or ``final_source``.
* Non-scalar values (dict / list / bytes) are rejected outright.

The module owns a process-wide default aggregator so the gateway can emit
without plumbing; services that want isolation (tests, eval harnesses, an
in-app diagnostics view) construct their own :class:`LLMObservabilityAggregator`
and inject it.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, replace
from typing import Any, TypedDict

from mindflow.infrastructure.llm.http_client import HTTPAttemptMetrics

#: Maximum length of a label field. Long enough for role/node/source names,
#: short enough that no prompt or body text can ride along.
_TEXT_LIMIT = 64

#: Cost is expressed in *relative* units: 1 unit = 1e6 input tokens, with output
#: and reasoning tokens weighted 4x (they are the expensive direction on every
#: provider MindFlow talks to). Model-agnostic by design — the plan asks for a
#: comparable cost signal, not a billing figure.
_COST_PER_MILLION_INPUT = 1.0
_COST_PER_MILLION_OUTPUT = 4.0


# ═══════════════════════════════════════════════════════════════════════════════
# Field coercion — the allow-list boundary
# ═══════════════════════════════════════════════════════════════════════════════


def _text(value: object, *, name: str) -> str:
    """Return *value* as a short label; reject anything that is not a string."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string label, got {type(value).__name__}")
    return " ".join(value.split())[:_TEXT_LIMIT]


def _count(value: object, *, name: str) -> int:
    """Return a non-negative integer token/count field."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    return max(0, int(value))


def _milliseconds(value: object, *, name: str) -> float:
    """Return a non-negative, finite millisecond value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return 0.0
    return max(0.0, number)


def _epoch_seconds(value: object, *, name: str) -> float:
    """Return a non-negative, finite Unix timestamp in seconds."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return 0.0
    return max(0.0, number)


def _flag(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _optional_flag(value: object, *, name: str) -> bool | None:
    if value is None:
        return None
    return _flag(value, name=name)


# ═══════════════════════════════════════════════════════════════════════════════
# Record
# ═══════════════════════════════════════════════════════════════════════════════

_TEXT_FIELDS = frozenset({
    "graph", "node", "role", "provider", "model",
    "http_status_class", "fallback_reason", "final_source", "error_category",
})
_MILLISECOND_FIELDS = frozenset({
    "queue_latency_ms", "http_latency_ms", "total_latency_ms",
})
_COUNT_FIELDS = frozenset({
    "input_tokens", "output_tokens", "reasoning_tokens", "retry_count",
})
_FLAG_FIELDS = frozenset({
    "parse_failure", "forbidden_word_failure", "cache_hit", "ok",
})


@dataclass(frozen=True, slots=True)
class LLMRequestRecord:
    """One LLM request's aggregated metadata.

    Args:
        graph: Graph that issued the call (``panel`` / ``analysis`` / ``chat``…).
        node: Graph node that issued the call.
        role: Expert/consumer role the request was shaped for.
        provider: ``ecnu`` / ``deepseek`` / ``ollama`` / ``rule_engine``.
        model: Model id sent on the wire.
        queue_latency_ms: Time spent waiting for the shared concurrency gate.
        http_latency_ms: Time spent inside HTTP (summed over attempts).
        total_latency_ms: Wall time of the whole gateway call. ``0`` marks an
            *event* record (batch retry, degradation outcome) with no measured
            request of its own — event records are excluded from latency
            percentiles and cost means.
        input_tokens: Prompt tokens reported by the provider (0 when unknown).
        output_tokens: Completion tokens reported by the provider.
        reasoning_tokens: Thinking tokens reported by the provider.
        retry_count: Retries consumed by this request (0 = first attempt won).
        http_status_class: ``2xx`` / ``4xx`` / ``5xx`` / ``""`` (transport error).
        parse_failure: The caller could not parse the returned payload.
        forbidden_word_failure: The output tripped the forbidden-word gate.
        fallback_reason: Why the request fell through (sanitized category).
        cache_hit: The result came from cache.
        critic_approved: Critic verdict, when the request produced one.
        final_source: Source label the workflow finally published.
        ok: Transport-level success of the request.
        error_category: Sanitized transport error category (``timeout``/…).
        recorded_at: Unix timestamp (seconds) of the record.
    """

    graph: str = ""
    node: str = ""
    role: str = ""
    provider: str = ""
    model: str = ""
    queue_latency_ms: float = 0.0
    http_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    retry_count: int = 0
    http_status_class: str = ""
    parse_failure: bool = False
    forbidden_word_failure: bool = False
    fallback_reason: str = ""
    cache_hit: bool = False
    critic_approved: bool | None = None
    final_source: str = ""
    ok: bool = True
    error_category: str = ""
    recorded_at: float = 0.0

    def __post_init__(self) -> None:
        # Normalise every field through the allow-list boundary above. Frozen
        # dataclass: write through ``object.__setattr__``.
        for field in fields(self):
            name = field.name
            raw = getattr(self, name)
            if name in _TEXT_FIELDS:
                value: object = _text(raw, name=name)
            elif name in _MILLISECOND_FIELDS:
                value = _milliseconds(raw, name=name)
            elif name in _COUNT_FIELDS:
                value = _count(raw, name=name)
            elif name in _FLAG_FIELDS:
                value = _flag(raw, name=name)
            elif name == "critic_approved":
                value = _optional_flag(raw, name=name)
            elif name == "recorded_at":
                value = _epoch_seconds(raw, name=name)
            else:  # pragma: no cover - guards a new field added without a rule
                raise TypeError(f"{name} has no coercion rule")
            object.__setattr__(self, name, value)

    @property
    def cost_units(self) -> float:
        """Relative cost of the request (see ``_COST_PER_MILLION_*``)."""
        return (
            self.input_tokens * _COST_PER_MILLION_INPUT
            + (self.output_tokens + self.reasoning_tokens) * _COST_PER_MILLION_OUTPUT
        ) / 1_000_000.0

    @property
    def failed(self) -> bool:
        """True when the request failed at transport, parse, or safety level."""
        return (not self.ok) or self.parse_failure or self.forbidden_word_failure

    @property
    def is_event(self) -> bool:
        """True for annotation-only records (no measured request of their own)."""
        return self.total_latency_ms <= 0.0

    def as_dict(self) -> dict[str, object]:
        """Flat scalar view — the only shape that ever leaves this module."""
        return {field.name: getattr(self, field.name) for field in fields(self)}


#: Everything a record may carry. Anything else is rejected before storage.
ALLOWED_FIELDS: frozenset[str] = frozenset(
    field.name for field in fields(LLMRequestRecord)
)


def summarise_attempts(
    attempts: Sequence[HTTPAttemptMetrics],
) -> tuple[float, float, str, str]:
    """Fold per-attempt HTTP metrics into one request-level summary.

    Returns ``(queue_latency_ms, http_latency_ms, status_class, error_category)``.
    Latencies are summed (a retried request spent that time), while the status
    class and error category come from the last attempt — the one that decided
    the outcome.
    """
    queue_ms = sum(a.queue_latency_ms for a in attempts)
    http_ms = sum(a.http_latency_ms for a in attempts)
    if not attempts:
        return queue_ms, http_ms, "", ""
    last = attempts[-1]
    return queue_ms, http_ms, last.status_class, last.error_category


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════════════════════


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of *values* (0.0 for an empty sample).

    Args:
        values: Sample (unsorted is fine).
        q: Quantile in ``[0, 1]`` — ``0.5`` = p50, ``0.95`` = p95.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * min(max(q, 0.0), 1.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


class GroupStats(TypedDict):
    """Aggregates for one group of records (per role, per graph, or overall)."""

    count: int
    requests: int
    failure_rate: float
    parse_failure_rate: float
    forbidden_word_failure_rate: float
    p50_latency_ms: float
    p95_latency_ms: float
    mean_latency_ms: float
    mean_cost_units: float
    total_cost_units: float
    mean_retries: float
    retry_rate: float
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int


class ObservabilitySnapshot(TypedDict):
    """Snapshot shape: overall totals plus per-role and per-graph breakdowns."""

    total: GroupStats
    by_role: dict[str, GroupStats]
    by_graph: dict[str, GroupStats]


def _group_stats(records: Sequence[LLMRequestRecord]) -> GroupStats:
    """Aggregate one group of records (per role, per graph, or overall)."""
    count = len(records)

    # Latency and cost are only meaningful for records that measured a request:
    # event records (batch retries, degradation outcomes) carry no latency.
    measured = [r for r in records if not r.is_event]
    latencies = [r.total_latency_ms for r in measured]
    costed = [r for r in measured if (r.input_tokens or r.output_tokens)]
    total_cost = sum(r.cost_units for r in costed)

    return GroupStats(
        count=count,
        requests=len(measured),
        failure_rate=(sum(1 for r in records if r.failed) / count) if count else 0.0,
        parse_failure_rate=(
            sum(1 for r in records if r.parse_failure) / count
        ) if count else 0.0,
        forbidden_word_failure_rate=(
            sum(1 for r in records if r.forbidden_word_failure) / count
        ) if count else 0.0,
        p50_latency_ms=percentile(latencies, 0.50),
        p95_latency_ms=percentile(latencies, 0.95),
        mean_latency_ms=(sum(latencies) / len(latencies)) if latencies else 0.0,
        mean_cost_units=(total_cost / len(costed)) if costed else 0.0,
        total_cost_units=total_cost,
        mean_retries=(sum(r.retry_count for r in records) / count) if count else 0.0,
        retry_rate=(sum(1 for r in records if r.retry_count > 0) / count) if count else 0.0,
        input_tokens=sum(r.input_tokens for r in records),
        output_tokens=sum(r.output_tokens for r in records),
        reasoning_tokens=sum(r.reasoning_tokens for r in records),
    )


class LLMObservabilityAggregator:
    """Bounded, in-memory store of :class:`LLMRequestRecord` plus aggregation.

    Args:
        max_records: Ring size. The oldest record is dropped past the cap, so a
            long-running desktop process cannot grow without bound.
    """

    def __init__(self, *, max_records: int = 5000) -> None:
        self._max_records = max(1, int(max_records))
        self._records: list[LLMRequestRecord] = []

    # ── Recording ─────────────────────────────────────────────────────────

    def record(self, record: LLMRequestRecord) -> LLMRequestRecord:
        """Store *record* (filling in the timestamp when the caller left it 0)."""
        stored = record if record.recorded_at > 0 else replace(record, recorded_at=time.time())
        self._records.append(stored)
        if len(self._records) > self._max_records:
            del self._records[: len(self._records) - self._max_records]
        return stored

    def records(self) -> tuple[LLMRequestRecord, ...]:
        """Snapshot of stored records, oldest first."""
        return tuple(self._records)

    def reset(self) -> None:
        """Drop every stored record."""
        self._records.clear()

    # ── Reporting ─────────────────────────────────────────────────────────

    def snapshot(self) -> ObservabilitySnapshot:
        """Per-role and per-graph aggregates over the stored window.

        Records without a role/graph label are counted in ``total`` only, so an
        unlabelled caller cannot invent a phantom group.
        """
        records = self._records
        roles = sorted({r.role for r in records if r.role})
        graphs = sorted({r.graph for r in records if r.graph})
        return ObservabilitySnapshot(
            total=_group_stats(records),
            by_role={role: _group_stats([r for r in records if r.role == role]) for role in roles},
            by_graph={
                graph: _group_stats([r for r in records if r.graph == graph])
                for graph in graphs
            },
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Process-wide default aggregator + ambient call labels
# ═══════════════════════════════════════════════════════════════════════════════

_DEFAULT_AGGREGATOR = LLMObservabilityAggregator()


def default_aggregator() -> LLMObservabilityAggregator:
    """Return the process-wide aggregator used when none is injected."""
    return _DEFAULT_AGGREGATOR


def record_llm_request(record: LLMRequestRecord) -> LLMRequestRecord:
    """Record *record* on the default aggregator."""
    return _DEFAULT_AGGREGATOR.record(record)


def record_llm_outcome(**fields: Any) -> LLMRequestRecord:
    """Record a decision-level outcome (parse, safety, fallback, final source).

    Graph nodes know things the transport layer cannot (a parse failure, a
    forbidden-word rejection, the source that was finally published, the critic
    verdict). This is the single entry point for those facts.

    Args:
        **fields: Any subset of :data:`ALLOWED_FIELDS`.

    Raises:
        TypeError: On an unknown field name — the allow-list is enforced here,
            not only in the dataclass, so a typo cannot smuggle text into the
            store.
    """
    unknown = set(fields) - ALLOWED_FIELDS
    if unknown:
        raise TypeError(f"unknown observability field(s): {sorted(unknown)}")
    return _DEFAULT_AGGREGATOR.record(LLMRequestRecord(**fields))


def llm_observability_snapshot() -> ObservabilitySnapshot:
    """Aggregates over the default aggregator's current window."""
    return _DEFAULT_AGGREGATOR.snapshot()


def reset_llm_observability() -> None:
    """Drop every record held by the default aggregator (test/diagnostics hook)."""
    _DEFAULT_AGGREGATOR.reset()


@dataclass(frozen=True, slots=True)
class _CallLabels:
    graph: str = ""
    node: str = ""
    role: str = ""


_labels: ContextVar[_CallLabels | None] = ContextVar("mindflow_llm_call_labels", default=None)


def _active_labels() -> _CallLabels:
    """Current labels, materialised on demand (no shared mutable default)."""
    return _labels.get() or _CallLabels()


@contextmanager
def llm_call_context(
    *, graph: str = "", node: str = "", role: str = "",
) -> Iterator[None]:
    """Label every LLM request emitted inside this context.

    Inner contexts override outer ones field by field, so a graph node can add a
    node name without losing the graph the caller already set.
    """
    current = _active_labels()
    merged = _CallLabels(
        graph=graph or current.graph,
        node=node or current.node,
        role=role or current.role,
    )
    token = _labels.set(merged)
    try:
        yield
    finally:
        _labels.reset(token)


def current_llm_labels() -> tuple[str, str, str]:
    """Return ``(graph, node, role)`` for the request being served."""
    labels = _active_labels()
    return labels.graph, labels.node, labels.role


__all__ = [
    "ALLOWED_FIELDS",
    "GroupStats",
    "LLMObservabilityAggregator",
    "LLMRequestRecord",
    "ObservabilitySnapshot",
    "current_llm_labels",
    "default_aggregator",
    "llm_call_context",
    "llm_observability_snapshot",
    "percentile",
    "record_llm_outcome",
    "record_llm_request",
    "reset_llm_observability",
    "summarise_attempts",
]
