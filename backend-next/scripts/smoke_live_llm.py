"""Budgeted live smoke test for the four LLM paths (optimisation plan item 4).

This is a **maintenance script**, not a test: it never runs automatically, it is
excluded from ``pytest`` collection by living outside ``tests/``, and it refuses
to do anything without an explicit ``--yes``.

What it does
------------
It spends a hard-capped number of *real* provider requests on up to four
independently selectable smoke targets, and reports facts rather than
impressions:

============  ==========================================================
target        what it exercises
============  ==========================================================
``--panel``   the structured expert panel (analyst + 3 attribution +
              moderator + critic) on a synthetic evidence bundle
``--chat``    one plain chat generation through ``ChatGraph``
``--tools``   one chat turn that must actually call a tool and then answer
              (DeepSeek's thinking-mode contract — the assistant turn's
              ``reasoning_content`` is echoed on the request carrying tools —
              lives in ``infrastructure/llm/thinking.py``; this target proves it
              end to end by requiring the tool call to succeed and the follow-up
              request to be answered rather than 400-rejected)
``--attribution``  the single-expert L1 attribution path
============  ==========================================================

Per target it records: request count, schema pass rate, citation validity,
the degradation path and the marker that selected it, **provider-reported**
token usage, failure rate and p95 latency.

Cost honesty
------------
Character counts are never reported as token savings. Token numbers come from
the provider's own ``usage`` field; when a provider omits usage the report says
``usage_reported: false`` instead of inventing an estimate.

Thresholds (the PASS/FAIL contract)
-----------------------------------
Scenario gates — a target FAILS when any of these does not hold:

* ``schema_passes``  ≥ 1 for panel/attribution (a run with zero validated
  structured outputs proves nothing);
* ``panel``: ``schema_pass_rate`` = 1.0, ``citation_validity`` = 1.0,
  ``degradation`` = ``panel`` with an empty degradation path;
* ``chat``/``tools``: a non-empty answer, ``failure_rate`` ≤ 0.5;
* ``tools``: ``tool_call_rate`` ≥ 1.0 (the premise of the target is that the
  model *actually* calls a tool);
* ``attribution``: ``degradation`` = ``single_expert`` (L1), not a fallback.

Soundness gates — every target FAILS when any of these does not hold:

* ``failure_rate`` ≤ 0.5;
* latency is sound: ``total_latency_ms`` > 0 and ≥ ``http_latency_ms`` on every
  measured call;
* ``usage_reported`` is true, or the report explicitly says usage was missing;
* no credential material appears anywhere in the artifacts;
* a degraded path is *reported*, never used to claim the smoke test passed.

Exit codes: ``0`` all selected targets PASS, ``1`` at least one FAIL,
``2`` configuration/usage error (missing credential, budget refusal, bad flag).

Safety
------
* ``--data-dir`` defaults to a fresh temporary directory; the user's real
  database is never opened by this script (chat persistence goes to an
  in-memory repository).
* The API key is never printed, logged, or written: artifacts record only the
  provider label, the model id, and the base-URL **host**.
* ``--dry-run`` exercises the entire pipeline with fake transports and no
  network access, so the plumbing (budgets, aggregation, artifacts, verdicts)
  is verifiable offline.

Usage
-----
    # offline plumbing check, no credential and no network needed
    uv run python scripts/smoke_live_llm.py --dry-run --yes --panel --chat --tools --attribution

    # live, budgeted, artifacts under data/experiments/<run-id>/
    uv run python scripts/smoke_live_llm.py --yes --max-requests 12 \\
        --panel --chat --tools --attribution
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import tempfile
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = PROJECT_ROOT / "data" / "experiments"

# ── Budget ────────────────────────────────────────────────────────────────────

#: Default per-run request budget. The script refuses to exceed it.
DEFAULT_MAX_REQUESTS = 12
#: Hard ceiling: --max-requests above this is refused, not clamped silently.
HARD_MAX_REQUESTS = 60
#: Wiring-only requests, so a budget cannot be spent before the first target.
MIN_REQUESTS = 1

#: Worst-case real provider requests per selected target (panel = analyst + 3
#: attribution + moderator + critic; tools = tool-proposing turn + answer turn).
TARGET_COST: dict[str, int] = {
    "panel": 6,
    "chat": 1,
    "tools": 2,
    "attribution": 1,
}
ALL_TARGETS: tuple[str, ...] = ("panel", "chat", "tools", "attribution")

# ── Thresholds (see the module docstring) ────────────────────────────────────

SCHEMA_PASS_RATE_PANEL = 1.0
CITATION_VALIDITY_PANEL = 1.0
MAX_FAILURE_RATE = 0.5
MIN_TOOL_CALL_RATE = 1.0
MIN_SCHEMA_PASSES = 1

TABLE_KEYS: tuple[str, ...] = (
    "input_tokens", "output_tokens", "reasoning_tokens", "retry_count",
)

#: Credential env var fallbacks, in priority order *after* the settings object.
#: Production L1 is DeepSeek direct; its documented spellings are both accepted.
_API_KEY_ENV_FALLBACKS: tuple[str, ...] = (
    "DEEPSEEK_API_KEY",
    "MINDFLOW_LLM__DEEPSEEK_API_KEY",
)

#: ``fallback_reason`` marking an attempt the budget refused before the transport.
_BUDGET_REFUSED_REASON = "request_budget_refused"

_DEGRADED_MARKERS: dict[str, str] = {
    "panel": "panel",
    "single_expert": "single_expert",
    "ollama": "ollama",
    "rule_engine": "rule_engine",
    "insufficient_data": "panel_fast_path_insufficient_data",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Ambient request accounting (PII-free: allow-listed fields only)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RequestAccounting:
    """Per-window view over the process-wide observability aggregator.

    The panel path records through the gateway and the chat path through
    ``ChatGraph``; both land on ``default_aggregator()``, so one window between
    two ``reset()`` calls is exactly "the requests this target made".
    """

    records: list[Any] = field(default_factory=list)

    def reset(self) -> None:
        """Drop the aggregator's window (the smoke run owns the process)."""
        _observability_module().reset_llm_observability()
        self.records = []

    def collect(self) -> list[Any]:
        """Snapshot the window as PII-free dicts (allow-listed fields only)."""
        self.records = [r.as_dict() for r in _observability_module().default_aggregator().records()]
        return self.records

    @property
    def refused(self) -> list[Any]:
        """Attempts the budget stopped before the transport (events, not calls)."""
        return [r for r in self.records if r["fallback_reason"] == _BUDGET_REFUSED_REASON]

    @property
    def measured(self) -> list[Any]:
        """Records that measured a real request (they carry a latency)."""
        return [r for r in self.records if float(r["total_latency_ms"]) > 0.0]

    @property
    def request_count(self) -> int:
        return len(self.measured)

    @property
    def failures(self) -> int:
        return sum(
            1 for r in self.measured
            if (not r["ok"]) or r["parse_failure"] or r["forbidden_word_failure"]
        )

    @property
    def failure_rate(self) -> float:
        total = len(self.measured)
        return (self.failures / total) if total else 0.0

    @property
    def latencies(self) -> list[float]:
        return [float(r["total_latency_ms"]) for r in self.measured]

    @property
    def p95_latency_ms(self) -> float:
        samples = self.latencies
        if not samples:
            return 0.0
        if len(samples) == 1:
            return samples[0]
        return statistics.quantiles(samples, n=20)[-1]

    @property
    def token_totals(self) -> dict[str, int]:
        return {
            key: sum(int(r[key]) for r in self.records)
            for key in ("input_tokens", "output_tokens", "reasoning_tokens")
        }

    @property
    def usage_reported(self) -> bool:
        """True only when the provider actually reported usage for a call."""
        return any(
            int(r["input_tokens"]) or int(r["output_tokens"]) or int(r["reasoning_tokens"])
            for r in self.records
        )

    @property
    def latency_sound(self) -> tuple[bool, str]:
        """``(sound, reason)`` — total must be positive and cover the HTTP span."""
        for r in self.measured:
            total = float(r["total_latency_ms"])
            http = float(r["http_latency_ms"])
            if total < 0.0 or http < 0.0:
                return (False, "negative latency recorded")
            if total < http:
                return (False, f"total {total:.1f}ms < http {http:.1f}ms")
        if not self.measured:
            return (False, "no measured request latency")
        return (True, "")

    @property
    def nodes(self) -> list[str]:
        seen: list[str] = []
        for r in self.records:
            node = str(r["node"])
            if node and node not in seen:
                seen.append(node)
        return seen

    @property
    def provider(self) -> str:
        for r in self.records:
            if r["provider"]:
                return str(r["provider"])
        return ""

    @property
    def model(self) -> str:
        for r in self.records:
            if r["model"]:
                return str(r["model"])
        return ""

    @property
    def error_categories(self) -> list[str]:
        seen: list[str] = []
        for r in self.records:
            category = str(r["error_category"])
            if category and category not in seen:
                seen.append(category)
        return seen

    @property
    def status_classes(self) -> list[str]:
        seen: list[str] = []
        for r in self.records:
            status = str(r["http_status_class"])
            if status and status not in seen:
                seen.append(status)
        return seen


def _observability_module() -> Any:
    """Import the observability module (deliberately not at module import time)."""
    return import_module("mindflow.services.llm_observability")


# ═══════════════════════════════════════════════════════════════════════════════
# Budget
# ═══════════════════════════════════════════════════════════════════════════════


class BudgetRefusedError(RuntimeError):
    """Raised by a budget guard when the next request would exceed the cap."""


class RunBudget:
    """Hard request budget, enforced *before* a request reaches a transport."""

    def __init__(self, max_requests: int) -> None:
        self.max_requests = max_requests
        self.attempted = 0

    def reserve(self) -> int:
        """Reserve one request, or refuse when the budget is exhausted."""
        if self.attempted >= self.max_requests:
            msg = (
                f"request budget exhausted: {self.attempted}/{self.max_requests} "
                f"used — refusing to send another real request"
            )
            raise BudgetRefusedError(msg)
        self.attempted += 1
        return self.attempted

    @property
    def remaining(self) -> int:
        return max(0, self.max_requests - self.attempted)


class BudgetedGateway:
    """``PanelLLMGateway`` proxy that charges the budget before every call."""

    def __init__(self, inner: Any, budget: RunBudget) -> None:
        self._inner = inner
        self._budget = budget

    async def complete(self, system: str, user: str, **kwargs: Any) -> str:
        self._budget.reserve()
        return str(await self._inner.complete(system, user, **kwargs))

    async def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if close is not None:
            await close()


class BudgetedChatModel:
    """Chat-model proxy that charges the budget before every generation."""

    def __init__(self, inner: Any, budget: RunBudget) -> None:
        self._inner = inner
        self._budget = budget

    def bind_tools(self, tools: Any, **kwargs: Any) -> BudgetedChatModel:
        bound = self._inner.bind_tools(tools, **kwargs) if hasattr(
            self._inner, "bind_tools"
        ) else self._inner
        return BudgetedChatModel(bound, self._budget)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        self._budget.reserve()
        return await self._inner.ainvoke(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class BudgetedAttributionClient:
    """``DeepSeekClient`` proxy that charges the budget before every analyze()."""

    def __init__(self, inner: Any, budget: RunBudget) -> None:
        self._inner = inner
        self._budget = budget

    async def analyze(self, summary_json: str) -> Any:
        self._budget.reserve()
        return await self._inner.analyze(summary_json)

    @property
    def model(self) -> str:
        return str(getattr(self._inner, "model", ""))

    @property
    def last_usage(self) -> tuple[int, int, int] | None:
        """Forward the inner client's provider-reported usage (or None)."""
        usage = getattr(self._inner, "last_usage", None)
        return usage if isinstance(usage, tuple) else None


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic inputs (deterministic — a smoke run must be comparable run to run)
# ═══════════════════════════════════════════════════════════════════════════════


def build_synthetic_bundle() -> Any:
    """One synthetic but schema-valid evidence bundle for the panel target."""
    from datetime import datetime as dt

    from mindflow.domain.evidence import EvidenceBundle, EvidenceItem
    from mindflow.domain.procrastination import BehaviorSummary

    def item(
        metric: str, value: float, baseline: float | None, severity: str,
        confidence: float, source: str, human_readable: str,
    ) -> EvidenceItem:
        return EvidenceItem(
            metric=metric, value=value, baseline=baseline, severity=severity,  # type: ignore[arg-type]
            confidence=confidence, source=source, human_readable=human_readable,
        )

    window = (
        dt(2026, 7, 18, 9, 0, tzinfo=UTC),
        dt(2026, 7, 18, 11, 0, tzinfo=UTC),
    )
    return EvidenceBundle(
        user_id=1,
        window=window,
        items=(
            item("focus_score", 0.31, 0.72, "moderate", 0.85, "feature_computation",
                 "专注度 31%（基线 72%）"),
            item("switch_rate", 21.0, 8.0, "severe", 0.80, "feature_computation",
                 "切换频率 21.0 次/小时（基线 8.0）"),
            item("longest_focus_block_s", 96.0, 600.0, "severe", 0.90, "feature_computation",
                 "最长专注块 96s（基线 600s）"),
            item("social_media_ratio", 0.58, 0.20, "moderate", 0.75, "feature_computation",
                 "娱乐占比 58%（基线 20%）"),
            item("start_delay_min", 34.0, 10.0, "moderate", 0.80, "feature_computation",
                 "启动延迟 34min（基线 10min）"),
            item("behavior_deviation", 1.8, 0.0, "severe", 0.85, "welford_baseline",
                 "行为偏差 +1.8σ"),
        ),
        behavior_summary=BehaviorSummary(
            intended_task="复习线性代数",
            duration_min=120.0,
            actual_focus_min=37.0,
            context_switches_per_hour=21.0,
            longest_focus_block_s=96.0,
            social_media_ratio=0.58,
            start_delay_min=34.0,
            keyword_flags=frozenset(),
            baseline_deviation=1.8,
        ),
        intervention_history=(),
        novelty_flags=(),
    )


def build_synthetic_events() -> list[Any]:
    """Activity events that produce a realistic attribution behavior summary.

    Deliberately shaped so the derived summary is the "high-switch, low-focus"
    case: an entertainment-first start (start delay), a run of short alternating
    blocks (context switches), and a long social tail (social-media ratio).
    Browser windows carry an explicit entertainment domain because the
    classifier reads the visible URL, not the process name.
    """
    from datetime import datetime as dt
    from datetime import timedelta

    from mindflow.domain.events import ActivityEvent, WindowSnapshot

    start = dt(2026, 7, 18, 9, 0, tzinfo=UTC)
    social_title = "bilibili.com/video/BV1smoke - Google Chrome"
    plan: list[tuple[str, str, float]] = [
        ("chrome.exe", social_title, 900.0),          # entertainment-first start
        ("Code.exe", "线性代数笔记.md - VS Code", 600.0),   # the delayed start
    ]
    # Twenty short alternating blocks: the derived context-switch count.
    for index in range(20):
        productive = index % 2 == 0
        plan.append((
            "Code.exe" if productive else "chrome.exe",
            "线性代数笔记.md - VS Code" if productive else social_title,
            90.0,
        ))
    # A long social tail: the derived social-media ratio.
    plan.append(("chrome.exe", social_title, 1500.0))

    events: list[Any] = []
    cursor = start
    for index, (process, title, duration) in enumerate(plan):
        events.append(ActivityEvent(
            id=f"smoke-{index}",
            user_id=1,
            timestamp_utc=cursor,
            duration_s=duration,
            event_type="window_snapshot",
            data=WindowSnapshot(
                app_name=process.split(".")[0],
                window_title=title,
                process_name=process,
                is_idle=False,
                timestamp_utc=cursor,
            ),
        ))
        cursor += timedelta(seconds=duration)
    return events


def build_synthetic_summary_json() -> str:
    """Serialize ``build_synthetic_events()`` for the single-expert L1 call."""
    from mindflow.graph.fallback_nodes import build_behavior_bundle

    _summary, summary_json = build_behavior_bundle(build_synthetic_events())
    return str(summary_json)


# ═══════════════════════════════════════════════════════════════════════════════
# Citation validation (independent of the graph — the script checks the output)
# ═══════════════════════════════════════════════════════════════════════════════

_CITATION_OPEN = "[证据"


def extract_citation_ids(text: str) -> list[str]:
    """Extract ids from every ``[证据: id]`` marker in *text*."""
    found: list[str] = []
    cursor = 0
    while True:
        start = text.find(_CITATION_OPEN, cursor)
        if start < 0:
            break
        colon = text.find(":", start)
        end = text.find("]", start)
        if colon < 0 or end < 0 or colon > end:
            cursor = start + len(_CITATION_OPEN)
            continue
        cited = text[colon + 1:end].strip()
        if cited:
            found.append(cited)
        cursor = end + 1
    return found


def citation_report(
    citations: Sequence[str], catalog: frozenset[str],
) -> dict[str, Any]:
    """Resolve every citation against the bundle catalog (validity contract).

    Resolution mirrors the graph's own ``validate_citations``: an exact catalog
    id wins, and a bare metric name that maps to exactly one canonical id is
    accepted as an alias. Everything else is a hallucinated citation.
    """
    bare_to_canonical: dict[str, str | None] = {}
    for catalog_id in catalog:
        if "." in catalog_id:
            bare = catalog_id.rsplit(".", 1)[-1]
            bare_to_canonical[bare] = (
                catalog_id if bare not in bare_to_canonical else None
            )

    unique: list[str] = []
    for cited in citations:
        if cited not in unique:
            unique.append(cited)

    valid: list[str] = []
    invalid: list[str] = []
    for cited in unique:
        if cited in catalog or bare_to_canonical.get(cited):
            valid.append(cited)
        else:
            invalid.append(cited)
    return {
        "total": len(unique),
        "valid": len(valid),
        "invalid": sorted(invalid),
        "validity": (len(valid) / len(unique)) if unique else 1.0,
        "catalog_size": len(catalog),
    }


def collect_verdict_citations(verdict: Any) -> list[str]:
    """Every citation surface the panel verdict exposes, scanned in one place.

    Scanned: ``rationale``, every ``dissent`` entry, and every transcript entry's
    content.  Expert opinion citations are not re-derived here — the panel graph
    already drops opinions whose ``evidence_citations`` do not resolve, so a
    surviving verdict is what the graph accepted.
    """
    texts: list[str] = [str(getattr(verdict, "rationale", ""))]
    texts.extend(str(entry) for entry in getattr(verdict, "dissent", ()) or ())
    for entry in getattr(verdict, "transcript", ()) or ():
        texts.append(str(getattr(entry, "content", "")))
    citations: list[str] = []
    for text in texts:
        citations.extend(extract_citation_ids(text))
    return citations


# ═══════════════════════════════════════════════════════════════════════════════
# Target result
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TargetResult:
    """Everything the report states about one smoke target.

    With ``--repeat``/``--scenario-count`` one target aggregates several runs:
    counters add up, ``runs`` counts the attempts, ``runs_with_tool`` feeds the
    tool-call rate, and notes/degradation paths are merged in arrival order.
    """

    target: str
    ok: bool = False
    verdict: str = "FAIL"
    reasons: list[str] = field(default_factory=list)
    degraded: bool = False
    degradation_path: list[str] = field(default_factory=list)
    degradation_marker: str = ""
    schema_attempts: int = 0
    schema_passes: int = 0
    citations: dict[str, Any] = field(default_factory=dict)
    tool_calls: int = 0
    tool_call_rate: float = 0.0
    note: str = ""
    runs: int = 0
    runs_with_tool: int = 0

    @property
    def schema_pass_rate(self) -> float:
        return (self.schema_passes / self.schema_attempts) if self.schema_attempts else 0.0

    def add_note(self, text: str) -> None:
        """Append one run's note, keeping single-run notes unchanged."""
        if not text:
            return
        self.note = f"{self.note} | {text}" if self.note else text

    def merge_degradation(
        self, degraded: bool, path: list[str], marker: str = "",
    ) -> None:
        """Fold one run's degradation facts into the aggregated view."""
        self.degraded = self.degraded or degraded
        for step in path:
            if step not in self.degradation_path:
                self.degradation_path.append(step)
        self.degradation_marker = self.degradation_marker or marker

    def merge_citations(self, report: dict[str, Any]) -> None:
        """Union one run's citation report and recompute the validity rate.

        ``citation_report`` stores ``valid`` as a *count* and ``invalid`` as a
        list of ids, so the merge adds the counts and unions the ids.
        """
        if not report:
            return
        if not self.citations:
            self.citations = dict(report)
            return

        def _count(value: Any) -> int:
            return int(value) if isinstance(value, int) else len(value or [])

        total = int(self.citations.get("total", 0)) + int(report.get("total", 0))
        valid = _count(self.citations.get("valid")) + _count(report.get("valid"))
        invalid = sorted({
            *(self.citations.get("invalid") or []),
            *(report.get("invalid") or []),
        })
        self.citations = {
            "total": total,
            "valid": valid,
            "invalid": invalid,
            "validity": (valid / total) if total else 1.0,
            "catalog_size": max(
                int(self.citations.get("catalog_size", 0)),
                int(report.get("catalog_size", 0)),
            ),
        }

    def as_dict(
        self, accounting: RequestAccounting, elapsed_s: float,
    ) -> dict[str, Any]:
        """Scalar report row — plus the allow-listed per-call records."""
        sound, unsound_reason = accounting.latency_sound
        return {
            "target": self.target,
            "verdict": self.verdict,
            "ok": self.ok,
            "reasons": self.reasons,
            "note": self.note,
            "requests": accounting.request_count,
            "records": len(accounting.records),
            "budget_refused": len(accounting.refused),
            "failure_rate": accounting.failure_rate,
            "p95_latency_ms": accounting.p95_latency_ms,
            "latency_sound": sound,
            "latency_note": unsound_reason,
            "schema_attempts": self.schema_attempts,
            "schema_passes": self.schema_passes,
            "schema_pass_rate": self.schema_pass_rate,
            "citations": self.citations,
            "degraded": self.degraded,
            "degradation_path": self.degradation_path,
            "degradation_marker": self.degradation_marker,
            "tool_calls": self.tool_calls,
            "tool_call_rate": self.tool_call_rate,
            "usage_reported": accounting.usage_reported,
            "usage_note": (
                "provider-reported usage" if accounting.usage_reported
                else "provider omitted usage — token totals are NOT estimated"
            ),
            "tokens": accounting.token_totals,
            "provider": accounting.provider,
            "model": accounting.model,
            "http_status_classes": accounting.status_classes,
            "error_categories": accounting.error_categories,
            "nodes": accounting.nodes,
            "elapsed_s": round(elapsed_s, 3),
            "calls": accounting.records,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Verdicts
# ═══════════════════════════════════════════════════════════════════════════════


def _soundness_reasons(
    accounting: RequestAccounting, credential: str, *, dry_run: bool,
) -> list[str]:
    """Gates every target must satisfy, whatever its scenario result was.

    A dry run has no provider request to measure by construction, so the
    request/latency/usage gates would only measure the fake; they are skipped
    and the run is labelled ``dry_run`` in the artifacts instead.
    """
    reasons: list[str] = []
    if dry_run:
        return reasons
    if accounting.failure_rate > MAX_FAILURE_RATE:
        reasons.append(
            f"failure_rate {accounting.failure_rate:.2f} > {MAX_FAILURE_RATE}"
        )
    if accounting.request_count == 0:
        reasons.append("no measured request reached the provider")
    sound, unsound_reason = accounting.latency_sound
    if not sound:
        reasons.append(f"latency unsound: {unsound_reason}")
    payload = json.dumps(accounting.records, ensure_ascii=False, default=str)
    if credential and credential in payload:
        reasons.append("credential material found in observability records")
    return reasons


def evaluate_panel(
    result: TargetResult, accounting: RequestAccounting, credential: str,
    *, dry_run: bool = False,
) -> TargetResult:
    """PASS only when the structured panel answered without degrading."""
    if result.schema_attempts and result.schema_pass_rate < SCHEMA_PASS_RATE_PANEL:
        result.reasons.append(
            f"schema_pass_rate {result.schema_pass_rate:.2f} < {SCHEMA_PASS_RATE_PANEL}"
        )
    if result.schema_passes < MIN_SCHEMA_PASSES:
        result.reasons.append("no structured output passed schema validation")
    # Citation validity only applies when the verdict actually cited evidence:
    # zero citations is a schema/coverage question (caught above), not a
    # hallucinated-citation failure.
    validity = float(result.citations.get("validity", 0.0))
    citation_total = int(result.citations.get("total", 0))
    if result.citations and citation_total and validity < CITATION_VALIDITY_PANEL:
        invalid = result.citations.get("invalid") or []
        result.reasons.append(
            f"citation_validity {validity:.2f} < {CITATION_VALIDITY_PANEL} "
            f"(invalid: {invalid})"
        )
    if result.degraded or result.degradation_path:
        result.reasons.append(
            f"degraded path {result.degradation_path or result.degradation_marker} "
            "cannot be reported as a passing panel run"
        )
    result.reasons.extend(_soundness_reasons(accounting, credential, dry_run=dry_run))
    result.ok = not result.reasons
    result.verdict = "PASS" if result.ok else "FAIL"
    return result


def evaluate_chat_like(
    result: TargetResult, accounting: RequestAccounting, credential: str,
    *, require_tool: bool, dry_run: bool = False,
) -> TargetResult:
    """PASS when the chat turn answered — and, for ``--tools``, called a tool."""
    placeholder_notes = {"(empty answer)", "(no answer)"}
    if not result.note or all(
        note.strip() in placeholder_notes
        for note in result.note.split(" | ")
        if note.strip()
    ):
        result.reasons.append("empty answer")
    if result.degraded or result.degradation_path:
        result.reasons.append(
            f"degraded path {result.degradation_path or result.degradation_marker} "
            "cannot be reported as a passing chat run"
        )
    if require_tool and result.tool_call_rate < MIN_TOOL_CALL_RATE:
        result.reasons.append(
            f"tool_call_rate {result.tool_call_rate:.2f} < {MIN_TOOL_CALL_RATE} "
            "(the model answered without calling a tool)"
        )
    result.reasons.extend(_soundness_reasons(accounting, credential, dry_run=dry_run))
    result.ok = not result.reasons
    result.verdict = "PASS" if result.ok else "FAIL"
    return result


def evaluate_attribution(
    result: TargetResult, accounting: RequestAccounting, credential: str,
    *, dry_run: bool = False,
) -> TargetResult:
    """PASS when the single-expert L1 path itself answered (not a fallback).

    ``single_expert`` is the *name of the tier*, not a degradation: L1 answering
    on its own is exactly what this target tests. A task-level failure that
    routes onward to Ollama/rule engine is what must fail here.
    """
    if result.schema_passes < MIN_SCHEMA_PASSES:
        result.reasons.append("L1 produced no schema-valid attribution result")
    if result.degraded:
        result.reasons.append(
            f"L1 did not answer on its own: path {result.degradation_path} "
            f"marker {result.degradation_marker}"
        )
    result.reasons.extend(_soundness_reasons(accounting, credential, dry_run=dry_run))
    result.ok = not result.reasons
    result.verdict = "PASS" if result.ok else "FAIL"
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Target executors
# ═══════════════════════════════════════════════════════════════════════════════


async def run_panel_target(
    gateway: Any, bundle: Any, result: TargetResult,
    fanout_gateway: Any | None = None,
) -> None:
    """Structured panel on one bundle (analyst + 3 + moderator + critic).

    Aggregating: with ``--repeat``/``--scenario-count`` the counters add up and
    notes/degradation facts merge, so a single run behaves exactly as before.
    """
    from mindflow.agents.orchestrator import validate_citations
    from mindflow.domain.evidence_facts import build_evidence_catalog, evidence_catalog_ids
    from mindflow.eval.adapters import panel_analyzer

    catalog = frozenset(evidence_catalog_ids(build_evidence_catalog(bundle)))
    analyzer = panel_analyzer(gateway, fanout_gateway)
    verdict = await analyzer(bundle)
    result.runs += 1
    if verdict is None:
        result.add_note("panel produced no verdict")
        result.schema_attempts += 1
        return

    # Schema signal: the graph only publishes a verdict built from opinions that
    # passed their ``extra="forbid"`` Pydantic schema, and it drops an opinion
    # whose citations do not resolve. Count what survived: the analyst + one
    # attribution expert per opinion plus the moderator verdict itself.
    transcript = tuple(getattr(verdict, "transcript", ()) or ())
    attempts = max(1, len(transcript))
    result.schema_attempts += attempts
    if getattr(verdict, "types", ()):
        result.schema_passes += attempts

    citations = collect_verdict_citations(verdict)
    report = citation_report(citations, catalog)
    bogus = validate_citations(_verdict_as_opinion(verdict), catalog)
    if bogus:
        report["invalid"] = sorted(
            set(report.get("invalid") or []) | set(bogus)
        )
        total = int(report["total"])
        report["validity"] = (
            ((total - len(report["invalid"])) / total) if total else 0.0
        )
    result.merge_citations(report)

    source = str(getattr(verdict, "source", ""))
    escalated = bool(getattr(verdict, "escalated", False))
    # Escalation is an *internal deliberation round* (the experts disagreed, so
    # the panel ran a rebuttal), not a fallback to a weaker tier: the verdict
    # still comes from `source == "panel"`. Only a non-panel source counts as
    # degradation; the escalation is reported in the note and the artifact.
    result.merge_degradation(
        source != "panel",
        [source] if source and source != "panel" else [],
        _DEGRADED_MARKERS.get(source, ""),
    )
    types = ",".join(str(getattr(t, "value", t)) for t in getattr(verdict, "types", ()))
    escalation_note = "escalated(rebuttal round)" if escalated else ""
    result.add_note("panel verdict: {} {}".format(
        types or "(no types)", escalation_note,
    ).strip())


def _verdict_as_opinion(verdict: Any) -> Any:
    """Adapt a ``PanelVerdict`` to the citation validator's opinion interface."""
    from mindflow.agents.types import ExpertOpinion

    return ExpertOpinion(
        role="panel_verdict",
        perspective="panel",
        attribution_types=tuple(str(getattr(t, "value", t)) for t in verdict.types),
        confidence={str(getattr(t, "value", t)): 1.0 for t in verdict.types},
        evidence_citations=tuple(extract_citation_ids(str(verdict.rationale))),
        argument=str(verdict.rationale),
    )


async def run_chat_target(model: Any, repo: Any, result: TargetResult) -> None:
    """One plain chat generation through the production ``ChatGraph``."""
    from mindflow.graph.chat_graph import ChatGraph
    from mindflow.infrastructure.security.crisis_detector import CrisisDetector

    chat_graph = ChatGraph(
        chat_repo=repo,
        crisis_detector=CrisisDetector(),
        model=model,
        tools=[],
        tool_adapters=[],
        provider=_ACTIVE_PROVIDER[0],
    )
    answer = await chat_graph.ask(
        user_id=1, session_id="smoke-chat", message="用一句话说明今天可以如何开始专注。",
    )
    result.runs += 1
    result.add_note(str(getattr(answer, "answer", "")))
    if getattr(answer, "degraded", False):
        result.merge_degradation(True, ["chat_degraded"], "chat_safe_reply")


def build_evidence_tool() -> tuple[Any, Any]:
    """A real LangChain ``query_evidence`` tool over synthetic evidence data."""
    from mindflow.agents.langchain_tools import make_query_evidence
    from mindflow.graph.tools import QueryEvidenceOutput, QueryEvidenceTool

    payload = {
        "evidence": [
            {
                "metric": "focus_score",
                "value": 0.31,
                "baseline": 0.72,
                "severity": "moderate",
                "human_readable": "专注度 31%（基线 72%）",
            },
            {
                "metric": "switch_rate",
                "value": 21.0,
                "baseline": 8.0,
                "severity": "severe",
                "human_readable": "切换频率 21.0 次/小时（基线 8.0）",
            },
        ],
        "behavior_summary": {"duration_min": 120.0, "actual_focus_min": 37.0},
    }

    class _SyntheticEvidenceTool(QueryEvidenceTool):
        """Returns a fixed evidence bundle — no database, no network."""

        def __init__(self) -> None:
            super().__init__(evidence_builder=None)

        async def execute(self, days: int = 7) -> QueryEvidenceOutput:
            return QueryEvidenceOutput(
                evidence_json=json.dumps(payload, ensure_ascii=False),
                total_days=min(days, 30),
            )

    adapter = _SyntheticEvidenceTool()
    return adapter, make_query_evidence(adapter)


async def run_tools_target(
    model: Any, repo: Any, tool: Any, adapter: Any, result: TargetResult,
) -> None:
    """One chat turn that must actually call a tool and then answer."""
    from mindflow.graph.chat_graph import ChatGraph
    from mindflow.infrastructure.security.crisis_detector import CrisisDetector

    chat_graph = ChatGraph(
        chat_repo=repo,
        crisis_detector=CrisisDetector(),
        model=model,
        tools=[tool],
        tool_adapters=[adapter],
        provider=_ACTIVE_PROVIDER[0],
    )
    answer = await chat_graph.ask(
        user_id=1,
        session_id="smoke-tools",
        message="请先调用 query_evidence 查询证据,然后根据证据用一句话给出建议。",
    )
    answer_text = str(getattr(answer, "answer", ""))
    used = tuple(getattr(answer, "tools_used", ()) or ())
    result.runs += 1
    result.tool_calls += len(used)
    if used:
        result.runs_with_tool += 1
    result.tool_call_rate = (
        result.runs_with_tool / result.runs if result.runs else 0.0
    )
    result.add_note(answer_text)
    # This target has no bundle catalog, so no citation can be resolved: report
    # the count (0 expected) rather than asserting a validity that cannot exist.
    result.merge_citations(citation_report(
        extract_citation_ids(answer_text), frozenset(),
    ))
    if getattr(answer, "degraded", False):
        result.merge_degradation(True, ["chat_degraded"], "chat_safe_reply")


async def run_attribution_target(
    client: Any, result: TargetResult,
) -> None:
    """The single-expert L1 attribution path, on synthetic events."""
    from mindflow.graph.fallback_nodes import (
        FallbackRunContext,
        FallbackState,
        single_expert_node,
    )

    state: FallbackState = {
        "user_id": 1,
        "summary_json": build_synthetic_summary_json(),
        "degradation_path": [],
        "runtime": FallbackRunContext(deepseek_client=client),
    }
    if client is None:
        # L1 not configured: report the degraded tier, never claim a pass.
        result.merge_degradation(True, ["deepseek"], "deepseek_not_configured")
        result.add_note("L1 credential missing — single_expert skipped")
        return

    update = await single_expert_node(state)
    result.runs += 1
    path = list(update.get("degradation_path", []))
    error = update.get("error")
    if error or update.get("current_result") is None:
        result.degraded = True
        error_text = str(error or "")
        if error_text.startswith("deepseek_transport"):
            result.degradation_marker = "single_expert_transport_failure"
        elif "schema" in error_text:
            result.degradation_marker = "single_expert_schema_failure"
        else:
            result.degradation_marker = "single_expert_unavailable"
        result.add_note(f"L1 failed: {error_text}")
        return

    assessment = update.get("current_result") or {}
    types = list(assessment.get("procrastination_types", []))
    # The node only publishes a result after ``LLMAttributionResult`` validated
    # it (strict, ``extra="forbid"``), so a returned assessment IS the schema pass.
    result.schema_attempts += 1
    if types:
        result.schema_passes += 1
    # ``degradation_path == ["deepseek"]`` is the tier's own name — L1 answered,
    # which is the pass condition for this target, not a degradation.
    result.merge_degradation(False, [], "single_expert")
    result.add_note(
        "L1 attribution types: " + ", ".join(str(t) for t in types)
        + f" (tier path: {path or ['deepseek']})"
    )


#: Provider label for the graph instances this run builds (set in ``main``).
_ACTIVE_PROVIDER: list[str] = [""]


# ═══════════════════════════════════════════════════════════════════════════════
# Ephemeral chat repository (the user's database is never opened)
# ═══════════════════════════════════════════════════════════════════════════════


class EphemeralChatRepo:
    """In-memory stand-in for ``ChatRepository`` (same minimal interface).

    The smoke script must exercise the real graph without ever opening the
    user's database, so chat persistence goes here and is discarded with the
    process.
    """

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    async def append(
        self, session_id: str, role: str, content: str, *,
        user_id: int = 1, message_id: str | None = None,
    ) -> dict[str, Any]:
        row = {
            "id": message_id or f"smoke-{uuid.uuid4()}",
            "user_id": user_id,
            "session_id": session_id,
            "role": role,
            "content": content,
        }
        self._rows.append(row)
        return dict(row)

    async def recent(
        self, session_id: str, *, limit: int = 20, user_id: int | None = None,
    ) -> list[dict[str, Any]]:
        rows = [
            row for row in self._rows
            if row["session_id"] == session_id
            and (user_id is None or row["user_id"] == user_id)
        ]
        return [dict(row) for row in rows[-limit:]]


# ═══════════════════════════════════════════════════════════════════════════════
# Environment / credentials
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Credential:
    """Where the L1 key came from — the key itself is never stored or printed.

    ``l1_configured`` is the authoritative answer: it is driven by the resolved
    ``l1_target()`` when the settings object exposes one. A raw ``api_key`` that
    the L1 resolution refused (a legacy ECNU key under the DeepSeek pin) must
    never make this script claim a configured endpoint.
    """

    value: str
    source: str
    l1_configured: bool = False
    l1_available: bool = False
    provenance: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.value) and (self.l1_configured or not self.l1_available)


def prepare_data_dir(data_dir: str | None) -> Path:
    """Point the process data dir at an isolated directory.

    Set *before* settings are first loaded, so nothing in this run resolves a
    path inside the user's real data directory. The credential is deliberately
    *not* isolated: it comes from the user-level settings/env, so a smoke run
    uses the same L1 configuration as production.
    """
    if data_dir:
        path = Path(data_dir).expanduser().resolve()
    else:
        path = Path(tempfile.mkdtemp(prefix="mindflow-smoke-")).resolve()
    path.mkdir(parents=True, exist_ok=True)
    os.environ["MINDFLOW_DATA_DIR"] = str(path)
    _drop_cached_settings()
    return path


def _drop_cached_settings() -> None:
    """Forget the process-cached settings so the new data dir takes effect."""
    config = import_module("mindflow.config")
    config.SETTINGS = None


def _l1_target(settings: Any) -> Any:
    """The production L1 target object, when the settings object exposes one.

    ``LLMSettings.l1_target()`` is where the DeepSeek-direct pin lives. It is
    read defensively so the script keeps working against an older settings
    object (ECNU-pinned diagnostics) instead of failing on the accessor.
    """
    accessor = getattr(settings.llm, "l1_target", None)
    if not callable(accessor):
        return None
    try:
        return accessor()
    except Exception:  # noqa: BLE001 - a diagnostics view must never raise here
        return None


def resolve_credential(settings: Any) -> Credential:
    """Resolve the L1 key: the L1 target first, then settings, then env.

    A missing credential must be *reported*: the caller refuses to run rather
    than letting the chain degrade to Ollama/rule engine and calling that a
    passing smoke test.
    """
    target = _l1_target(settings)
    available = target is not None
    target_key = str(getattr(target, "api_key", "") or "")
    provenance = str(getattr(target, "provenance", "") or "")
    if target_key:
        return Credential(
            target_key, "settings.llm.l1_target().api_key",
            l1_configured=True, l1_available=available, provenance=provenance,
        )
    configured = str(getattr(settings.llm, "api_key", "") or "")
    if configured and not available:
        # No resolved target to consult (legacy settings shape): the raw key is
        # the only signal there is.
        return Credential(configured, "settings.llm.api_key", l1_configured=True)
    for name in _API_KEY_ENV_FALLBACKS:
        value = os.environ.get(name, "")
        if value:
            return Credential(
                value, f"env:{name}", l1_configured=True,
                l1_available=available, provenance=provenance,
            )
    return Credential(
        "", "unconfigured", l1_configured=False,
        l1_available=available, provenance=provenance,
    )


def provider_snapshot(settings: Any, registry: Any | None) -> dict[str, Any]:
    """Provider facts for the report — no key, no full URL, host only.

    The L1 target wins on provider/model/host because it is the *resolved*
    endpoint the run will actually call; the registry supplies the effective
    thinking/concurrency budget when one was built (dry runs build none).
    """
    from urllib.parse import urlparse

    llm = settings.llm
    target = _l1_target(settings)
    describe: dict[str, Any] = {}
    if registry is not None:
        describe = dict(registry.describe())

    # Provider *identity* (the label chat/panel records carry), not the wire
    # protocol: "ecnu" for the compat campus gateway, "deepseek" for the pin.
    # Both record sources must agree or per-provider aggregation splits.
    provider = (
        "ecnu" if bool(getattr(target, "is_ecnu", False))
        else (
            "deepseek" if getattr(target, "provider", "")
            else str(describe.get("provider") or "")
        )
    )
    if not provider:
        provider = "ecnu" if llm.is_ecnu else "deepseek"
    model = str(getattr(target, "model", "") or describe.get("model") or llm.model or "")
    if not model and registry is not None:
        model = _lazy_model_label(registry)
    target_url = str(getattr(target, "base_url", "") or "")
    url = target_url or str(describe.get("base_url") or llm.base_url or "")
    return {
        "provider": provider,
        "model": model,
        "base_url_host": urlparse(url).hostname or "",
        "thinking_enabled": bool(describe.get("thinking_enabled", llm.thinking_enabled)),
        "max_output_tokens": int(describe.get("max_output_tokens", llm.max_output_tokens)),
        "concurrency_limit": int(describe.get("concurrency_limit", llm.max_concurrent_requests)),
        "timeout_s": int(llm.timeout_s),
        "max_retries": int(llm.max_retries),
        "l1_provenance": str(getattr(target, "provenance", "") or ""),
    }


def _lazy_model_label(registry: Any) -> str:
    """Best-effort model id from the registry's lazily built gateway models."""
    gateway = None
    try:
        gateway = registry.get_gateway()
    except Exception:  # noqa: BLE001 - diagnostics only
        return ""
    for tier in ("_chat_model", "_reasoner_model"):
        candidate = getattr(gateway, tier, None)
        for attribute in ("model_name", "model"):
            value = getattr(candidate, attribute, "")
            if isinstance(value, str) and value:
                return value
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# Dry-run fakes (no network, no credential — plumbing verification only)
# ═══════════════════════════════════════════════════════════════════════════════


def build_dry_run_chat_model() -> Any:
    """Scripted chat model for ``--chat``: one plain answer with usage metadata."""
    from langchain_core.messages import AIMessage

    return _ScriptedChatModel(responses=[
        AIMessage(
            content="根据你的行为数据，建议先关闭通知，做一次 25 分钟单任务冲刺。",
            usage_metadata={
                "input_tokens": 96,
                "output_tokens": 28,
                "total_tokens": 124,
                "output_token_details": {"reasoning": 6},
            },
        ),
    ])


def build_dry_run_tool_model() -> Any:
    """Scripted chat model for ``--tools``: propose a tool call, then answer."""
    from langchain_core.messages import AIMessage

    return _ScriptedChatModel(responses=[
        AIMessage(
            content="",
            tool_calls=[{
                "id": "call_smoke_1",
                "name": "query_evidence",
                "args": {"days_back": 7},
            }],
            usage_metadata={
                "input_tokens": 128,
                "output_tokens": 22,
                "total_tokens": 150,
                "output_token_details": {"reasoning": 4},
            },
        ),
        AIMessage(
            content="根据查询到的证据，专注度 31%、切换 21 次/小时，建议先做 25 分钟单任务冲刺。",
            usage_metadata={
                "input_tokens": 210,
                "output_tokens": 41,
                "total_tokens": 251,
                "output_token_details": {"reasoning": 9},
            },
        ),
    ])


class _ScriptedChatModel:
    """Minimal scripted ``BaseChatModel`` stand-in for ``--dry-run``.

    ChatGraph only needs ``bind_tools`` + ``ainvoke``; responses carry
    ``usage_metadata`` so the token-accounting path is exercised offline.
    """

    def __init__(self, responses: Sequence[Any]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.model_name = "dry-run-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> _ScriptedChatModel:
        _ = (tools, kwargs)
        return self

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        _ = (args, kwargs)
        if not self._responses:
            from langchain_core.messages import AIMessage

            return AIMessage(content="（dry-run 无脚本响应）")
        response = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return response


def build_dry_run_attribution_client() -> Any:
    """Stub L1 client returning a schema-valid ``LLMAttributionResult``."""
    from mindflow.infrastructure.llm.schemas import LLMAttributionResult

    class _DryRunAttributionClient:
        """Deterministic stand-in for ``DeepSeekClient`` (no HTTP at all)."""

        model = "dry-run-model"

        async def analyze(self, summary_json: str) -> LLMAttributionResult:
            _ = summary_json
            return LLMAttributionResult.model_validate({
                "procrastination_types": ["impulsivity"],
                "type_confidence": {"impulsivity": 0.72},
                "cognitive_distortions": ["all_or_nothing"],
                "cbt_technique": "stimulus_control",
                "response_text": "根据行为数据，建议先关闭通知，做一次 25 分钟单任务冲刺。",
                "next_action": "关闭浏览器娱乐标签页",
            })

    return _DryRunAttributionClient()


def build_panel_gateway(dry_run: bool, registry: Any | None) -> Any:
    """The real gateway, or the deterministic mock gateway for ``--dry-run``."""
    if dry_run:
        from mindflow.eval.adapters import MockPanelGateway

        return MockPanelGateway()
    if registry is None:  # pragma: no cover - guarded by the caller
        raise RuntimeError("live mode requires a provider registry")
    return registry.get_gateway()


def build_chat_model(
    dry_run: bool, registry: Any | None, budget: RunBudget, *, with_tools: bool,
) -> Any:
    """Budget-charged chat model (real registry model, or the dry-run fake)."""
    if dry_run:
        inner = build_dry_run_tool_model() if with_tools else build_dry_run_chat_model()
        return BudgetedChatModel(inner, budget)
    if registry is None:  # pragma: no cover - guarded by the caller
        raise RuntimeError("live mode requires a provider registry")
    model = registry.get_chat_model()
    if model is None:
        return None
    return BudgetedChatModel(model, budget)


def build_attribution_client(
    dry_run: bool, registry: Any | None, budget: RunBudget,
) -> Any:
    """Budget-charged L1 client (real registry client, or the dry-run fake)."""
    if dry_run:
        return BudgetedAttributionClient(build_dry_run_attribution_client(), budget)
    if registry is None:  # pragma: no cover - guarded by the caller
        raise RuntimeError("live mode requires a provider registry")
    client = registry.get_structured_attribution()
    if client is None:
        return None
    return BudgetedAttributionClient(client, budget)


# ═══════════════════════════════════════════════════════════════════════════════
# Artifacts
# ═══════════════════════════════════════════════════════════════════════════════


def run_directory(explicit: str, run_id: str) -> Path:
    """``data/experiments/<run-id>/`` — the repo's experiment convention."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    return (EXPERIMENTS_ROOT / run_id).resolve()


def write_artifacts(
    directory: Path,
    payload: dict[str, Any],
    accounting_by_target: dict[str, RequestAccounting],
    elapsed_by_target: dict[str, float],
) -> tuple[Path, Path]:
    """Write ``summary.json`` + ``report.md`` and return both paths."""
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / "summary.json"
    report_path = directory / "report.md"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    report_path.write_text(
        render_markdown_report(payload, accounting_by_target, elapsed_by_target),
        encoding="utf-8",
    )
    return summary_path, report_path


def _fmt_pct(value: float) -> str:
    return f"{value:.0%}"


def render_markdown_report(
    payload: dict[str, Any],
    accounting_by_target: dict[str, RequestAccounting],
    elapsed_by_target: dict[str, float],
) -> str:
    """Short markdown report: verdicts, thresholds, facts, no secrets."""
    env = payload["environment"]
    lines: list[str] = [
        "# MindFlow live LLM smoke report",
        "",
        f"- run_id: `{payload['run_id']}`",
        f"- mode: `{payload['mode']}`",
        f"- created_at: {payload['created_at']}",
        f"- provider: `{env['provider']}`  model: `{env['model'] or '(unresolved)'}`  "
        f"host: `{env['base_url_host'] or '(unset)'}`",
        f"- credential: `{env['credential_source']}` "
        f"(provenance: `{env.get('l1_provenance') or 'legacy settings shape'}`; "
        "value never recorded)",
        f"- data_dir: `{env['data_dir']}`",
        f"- budget: {payload['budget']['attempted']}/{payload['budget']['max_requests']} "
        f"requests attempted",
        "",
        "## Thresholds",
        "",
        f"- panel: schema_pass_rate = {SCHEMA_PASS_RATE_PANEL:.2f}, "
        f"citation_validity = {CITATION_VALIDITY_PANEL:.2f}, no degradation",
        f"- chat/tools: non-empty answer, failure_rate <= {MAX_FAILURE_RATE:.2f}",
        f"- tools: tool_call_rate >= {MIN_TOOL_CALL_RATE:.2f}",
        f"- attribution: schema_passes >= {MIN_SCHEMA_PASSES} and the L1 tier "
        "(``single_expert``) answered on its own",
        "- all targets: latency sound (total > 0 and total >= http), "
        "usage reported or explicitly missing, no credential in artifacts",
        "- ``--dry-run``: request/latency/usage gates are skipped (there is no "
        "real request to measure); artifacts are labelled ``dry_run``",
        "",
        "## Verdicts",
        "",
        "| target | verdict | requests | schema pass | citations | degradation | "
        "p95 ms | failure rate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for target in ALL_TARGETS:
        row = payload["targets"].get(target)
        if row is None:
            continue
        citations = row["citations"] or {}
        citation_cell = (
            f"{citations.get('valid', 0)}/{citations.get('total', 0)}"
            if citations else "n/a"
        )
        degradation = ",".join(row["degradation_path"]) or (
            row["degradation_marker"] or "none"
        )
        lines.append(
            f"| {target} | {row['verdict']} | {row['requests']} | "
            f"{_fmt_pct(row['schema_pass_rate'])} "
            f"({row['schema_passes']}/{row['schema_attempts']}) | {citation_cell} | "
            f"{degradation} | {row['p95_latency_ms']:.0f} | "
            f"{_fmt_pct(row['failure_rate'])} |"
        )

    lines.extend(["", "## Token usage (provider-reported)", ""])
    if not payload["targets"]:
        lines.append("- no target selected")
    for target, row in payload["targets"].items():
        tokens = row["tokens"]
        if row["usage_reported"]:
            lines.append(
                f"- `{target}`: input={tokens['input_tokens']} "
                f"output={tokens['output_tokens']} reasoning={tokens['reasoning_tokens']} "
                f"(provider usage)"
            )
        else:
            lines.append(
                f"- `{target}`: **usage not reported by the provider** — no token "
                "estimate is substituted (character counts are never token savings)"
            )

    lines.extend(["", "## Degradation and failures", ""])
    for target, row in payload["targets"].items():
        marker = row["degradation_marker"] or "none"
        failures = row["error_categories"] or []
        lines.append(
            f"- `{target}`: degraded={row['degraded']} marker=`{marker}` "
            f"path={row['degradation_path'] or '[]'} "
            f"error_categories={failures or '[]'}"
        )

    lines.extend(["", "## Per-target notes", ""])
    for target, row in payload["targets"].items():
        lines.append(f"- `{target}`: {row['note'] or '(no note)'}")
        for reason in row["reasons"]:
            lines.append(f"  - FAIL reason: {reason}")

    lines.extend([
        "",
        "## Recorded calls (allow-listed observability fields only)",
        "",
        "No prompt, credential, response body, or reasoning content can appear "
        "here: the record has no field for any of them, and label fields are "
        "whitespace-collapsed and truncated to 64 characters.",
        "",
        "| target | node | role | provider | model | queue ms | http ms | total ms "
        "| status | retries | in | out | reasoning |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for target, row in payload["targets"].items():
        for call in row["calls"]:
            lines.append(
                f"| {target} | {call['node']} | {call['role']} | {call['provider']} | "
                f"{call['model']} | {float(call['queue_latency_ms']):.0f} | "
                f"{float(call['http_latency_ms']):.0f} | "
                f"{float(call['total_latency_ms']):.0f} | {call['http_status_class']} | "
                f"{call['retry_count']} | {call['input_tokens']} | "
                f"{call['output_tokens']} | {call['reasoning_tokens']} |"
            )

    lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    """The script's entire CLI surface."""
    parser = argparse.ArgumentParser(
        description=(
            "Budgeted live LLM smoke test (real provider requests; never run "
            "automatically). See the module docstring for thresholds."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--panel", action="store_true", help="structured panel target")
    parser.add_argument("--chat", action="store_true", help="plain chat generation target")
    parser.add_argument(
        "--tools", action="store_true", help="chat turn that must call a tool",
    )
    parser.add_argument(
        "--attribution", action="store_true", help="single-expert L1 attribution target",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="explicit opt-in: acknowledge that real provider requests are billed",
    )
    parser.add_argument(
        "--max-requests", type=int, default=DEFAULT_MAX_REQUESTS, dest="max_requests",
        help=f"hard request budget (refused above {HARD_MAX_REQUESTS})",
    )
    parser.add_argument(
        "--data-dir", type=str, default="", dest="data_dir",
        help="isolated data directory (default: a fresh temporary directory)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="", dest="output_dir",
        help="artifact directory (default: data/experiments/<run-id>/)",
    )
    parser.add_argument(
        "--run-id", type=str, default="", dest="run_id",
        help="run identifier (default: UTC timestamp)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="exercise the whole pipeline with fake transports (no network)",
    )
    parser.add_argument(
        "--allow-degraded", action="store_true", dest="allow_degraded",
        help="report degraded paths as reported facts instead of failing the run",
    )
    parser.add_argument(
        "--scenario-count", type=int, default=1, dest="scenario_count",
        help=(
            "panel target: run the first N eval scenarios (default 1 = the "
            "built-in synthetic bundle) — this is what makes a run multi-scenario"
        ),
    )
    parser.add_argument(
        "--repeat", type=int, default=1, dest="repeat",
        help="run every selected target N times and aggregate the metrics",
    )
    return parser


def select_targets(args: argparse.Namespace) -> list[str]:
    """Selected targets in canonical order (empty selection is an error)."""
    return [name for name in ALL_TARGETS if getattr(args, name)]


class UsageError(RuntimeError):
    """Bad flags / missing credential — exit code 2, nothing was sent."""


def validate_args(args: argparse.Namespace) -> list[str]:
    """Validate the CLI before anything is built; returns the selected targets."""
    targets = select_targets(args)
    if not targets:
        msg = (
            "no target selected: pass at least one of "
            "--panel / --chat / --tools / --attribution"
        )
        raise UsageError(msg)
    if not args.yes:
        msg = (
            "refusing to run without --yes: this script sends real, billable "
            "provider requests"
        )
        raise UsageError(msg)
    if args.max_requests < MIN_REQUESTS:
        msg = f"--max-requests must be >= {MIN_REQUESTS}"
        raise UsageError(msg)
    if args.max_requests > HARD_MAX_REQUESTS:
        msg = (
            f"--max-requests {args.max_requests} exceeds the hard ceiling "
            f"{HARD_MAX_REQUESTS}; lower it deliberately"
        )
        raise UsageError(msg)
    if args.scenario_count < 1:
        msg = "--scenario-count must be >= 1"
        raise UsageError(msg)
    if args.repeat < 1:
        msg = "--repeat must be >= 1"
        raise UsageError(msg)
    estimate = 0
    for name in targets:
        # The panel scales with the scenario count, every target with --repeat.
        runs = args.repeat * (args.scenario_count if name == "panel" else 1)
        estimate += TARGET_COST[name] * runs
    if estimate > args.max_requests:
        msg = (
            f"selected targets need up to {estimate} requests but --max-requests "
            f"is {args.max_requests} (worst case: "
            + ", ".join(f"{name}={TARGET_COST[name]}" for name in targets)
            + "); raise the budget or drop a target"
        )
        raise UsageError(msg)
    return targets


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestration
# ═══════════════════════════════════════════════════════════════════════════════


async def run_targets(  # noqa: PLR0913 - one runner, all targets
    args: argparse.Namespace,
    targets: list[str],
    registry: Any | None,
    credential: Credential,
    *,
    gateway_override: Any | None = None,
) -> tuple[dict[str, TargetResult], dict[str, RequestAccounting], dict[str, float]]:
    """Run each selected target with its own budget-charged dependencies.

    ``gateway_override`` lets a caller (the offline tests) supply the panel
    gateway directly instead of building one; it is ignored for other targets.
    """
    accounting_by_target: dict[str, RequestAccounting] = {}
    elapsed_by_target: dict[str, float] = {}
    results: dict[str, TargetResult] = {}
    repo = EphemeralChatRepo()
    # Multi-scenario: the panel runs over the first N eval scenarios (diverse
    # bundles); every other target repeats its own fixed input. The budget is
    # per *run*, not per target: a second target must not quietly spend a fresh
    # allowance, so one counter is shared across every target.
    attempted = 0
    panel_bundles = _panel_bundles(args.scenario_count)
    adapter, tool = build_evidence_tool() if "tools" in targets else (None, None)

    for target in targets:
        budget = RunBudget(args.max_requests)
        budget.attempted = attempted
        accounting = RequestAccounting()
        accounting.reset()
        result = TargetResult(target=target)
        if args.dry_run:
            result.note = "[dry-run: fake transports, no provider request]"
        fanout_gateway = (
            None if (args.dry_run or registry is None or gateway_override is not None)
            else registry.get_fanout_gateway()
        )
        runs = args.repeat * (args.scenario_count if target == "panel" else 1)
        started = time.perf_counter()
        for run_index in range(max(1, runs)):
            run_bundle = (
                panel_bundles[run_index % len(panel_bundles)]
                if target == "panel" else None
            )
            try:
                await _run_one_target(
                    target, args, registry, budget, repo, run_bundle, tool, adapter,
                    result,
                    gateway_override=gateway_override,
                    fanout_gateway=fanout_gateway,
                )
            except BudgetRefusedError as exc:
                result.reasons.append(str(exc))
                result.note = f"{result.note} | budget refused: {exc}".strip(" |")
                break  # the shared budget is spent; further runs cannot happen
            except Exception as exc:  # noqa: BLE001 - report, never crash the run
                result.reasons.append(f"{type(exc).__name__}: {exc}")
                result.add_note(f"target raised {type(exc).__name__}")
                break
        elapsed_by_target[target] = time.perf_counter() - started
        attempted = budget.attempted
        accounting.collect()
        accounting_by_target[target] = accounting

        evaluator = {
            "panel": evaluate_panel,
            "tools": lambda r, a, c, **kw: evaluate_chat_like(r, a, c, require_tool=True, **kw),
        }.get(target)
        if evaluator is not None:
            evaluator(result, accounting, credential.value, dry_run=args.dry_run)
        elif target == "chat":
            evaluate_chat_like(
                result, accounting, credential.value,
                require_tool=False, dry_run=args.dry_run,
            )
        else:
            evaluate_attribution(result, accounting, credential.value, dry_run=args.dry_run)
        if args.allow_degraded:
            _reclassify_degraded(result)
        results[target] = result
    return results, accounting_by_target, elapsed_by_target


def _reclassify_degraded(result: TargetResult) -> None:
    """``--allow-degraded``: report a degraded path instead of failing on it.

    Only the degradation reasons are dropped. Every other gate (schema, citation,
    latency, tool call, budget) still decides the verdict, and the note records
    that the target ran degraded.
    """
    if not result.degraded:
        return
    result.reasons = [
        reason for reason in result.reasons
        if "degraded path" not in reason and "did not answer on its own" not in reason
    ]
    result.ok = not result.reasons
    result.verdict = "PASS" if result.ok else "FAIL"
    suffix = "[--allow-degraded: degradation reported, not passed as clean]"
    if suffix not in result.note:
        result.note = f"{result.note} {suffix}".strip()


def _panel_bundles(count: int) -> list[Any]:
    """The bundles the panel target runs over.

    ``count == 1`` keeps the historical single synthetic bundle; larger counts
    take the first N eval scenarios so a multi-scenario run covers genuinely
    different evidence shapes (diverse metrics, severities and baselines).
    """
    if count <= 1:
        return [build_synthetic_bundle()]
    from mindflow.eval.scenarios import ALL_SCENARIOS  # noqa: PLC0415

    return [scenario.bundle for scenario in ALL_SCENARIOS[:count]]


async def _run_one_target(  # noqa: PLR0913 - one dispatcher, one target
    target: str,
    args: argparse.Namespace,
    registry: Any | None,
    budget: RunBudget,
    repo: EphemeralChatRepo,
    bundle: Any,
    tool: Any,
    adapter: Any,
    result: TargetResult,
    *,
    gateway_override: Any | None = None,
    fanout_gateway: Any | None = None,
) -> None:
    """Build the target's dependencies and run it."""
    if target == "panel":
        inner = gateway_override or build_panel_gateway(args.dry_run, registry)
        gateway = BudgetedGateway(inner, budget)
        await run_panel_target(gateway, bundle, result, fanout_gateway)
        return
    if target == "chat":
        model = build_chat_model(args.dry_run, registry, budget, with_tools=False)
        await run_chat_target(model, repo, result)
        return
    if target == "tools":
        model = build_chat_model(args.dry_run, registry, budget, with_tools=True)
        await run_tools_target(model, repo, tool, adapter, result)
        return
    client = build_attribution_client(args.dry_run, registry, budget)
    await run_attribution_target(client, result)


def print_verdicts(
    results: dict[str, TargetResult], accounting_by_target: dict[str, RequestAccounting],
) -> None:
    """Explicit PASS/FAIL per target, with the numbers the verdict used."""
    print()
    print("=" * 72)
    print("  smoke verdicts")
    print("=" * 72)
    for target, result in results.items():
        accounting = accounting_by_target[target]
        citations = result.citations or {}
        citation_cell = (
            f"{citations.get('valid', 0)}/{citations.get('total', 0)}"
            if citations else "n/a"
        )
        print(
            f"  [{result.verdict}] {target:<12} requests={accounting.request_count} "
            f"schema={result.schema_passes}/{result.schema_attempts} "
            f"citations={citation_cell} failure_rate={accounting.failure_rate:.0%} "
            f"p95={accounting.p95_latency_ms:.0f}ms"
            + (
                f" budget_refused={len(accounting.refused)}"
                if accounting.refused else ""
            )
        )
        if not accounting.usage_reported:
            print("        usage: NOT reported by provider (no estimate substituted)")
        if result.degraded:
            print(
                f"        degraded: marker={result.degradation_marker or 'n/a'} "
                f"path={result.degradation_path or '[]'} (reported, not a pass)"
            )
        for reason in result.reasons:
            print(f"        FAIL reason: {reason}")
    print()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code (0/1/2)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        targets = validate_args(args)
    except UsageError as exc:
        print(f"[usage error] {exc}", file=sys.stderr)
        return 2

    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    directory = run_directory(args.output_dir, run_id)

    # Isolation first: settings must never see the production data dir.
    data_dir = prepare_data_dir(args.data_dir)

    config = import_module("mindflow.config")
    settings = config.get_settings()
    credential = resolve_credential(settings)
    registry: Any | None = None

    if not args.dry_run:
        if not credential.configured:
            pinned = credential.l1_available
            detail = (
                "the resolved L1 target has no credential"
                if pinned
                else "no L1 key in the settings object"
            )
            print(
                f"[config error] L1 is not configured: {detail} and none of the "
                "fallback env vars is set ("
                + ", ".join(_API_KEY_ENV_FALLBACKS)
                + ").\n"
                "  A missing L1 credential means the panel/attribution/chat tiers "
                "would silently answer from Ollama or the rule engine, so this run "
                "is refused instead of reporting a degraded path as a pass.\n"
                "  Set DEEPSEEK_API_KEY (or MINDFLOW_LLM__DEEPSEEK_API_KEY) and "
                "retry; use --dry-run to verify the plumbing offline.",
                file=sys.stderr,
            )
            return 2
        from mindflow.infrastructure.provider_registry import ProviderRegistry

        registry = ProviderRegistry(settings.llm)

    snapshot = provider_snapshot(settings, registry)
    _ACTIVE_PROVIDER[0] = snapshot["provider"]

    print(f"run_id={run_id} mode={'dry-run' if args.dry_run else 'live'} "
          f"targets={','.join(targets)} budget={args.max_requests} "
          f"scenario_count={args.scenario_count} repeat={args.repeat}")
    print(f"provider={snapshot['provider']} model={snapshot['model'] or '(unresolved)'} "
          f"host={snapshot['base_url_host'] or '(unset)'}")
    print(f"credential_source={credential.source} "
          f"provenance={credential.provenance or '(legacy settings shape)'} "
          "(value never printed)")
    print(f"data_dir={data_dir}")

    accounting_by_target: dict[str, RequestAccounting] = {}
    elapsed_by_target: dict[str, float] = {}
    results: dict[str, TargetResult] = {}
    try:
        results, accounting_by_target, elapsed_by_target = asyncio.run(
            run_targets(args, targets, registry, credential),
        )
    finally:
        if registry is not None:
            shutdown = getattr(registry, "shutdown", None)
            if shutdown is not None:
                asyncio.run(_maybe_await(shutdown()))

    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "mode": "dry_run" if args.dry_run else "live",
        "created_at": datetime.now(UTC).isoformat(),
        "environment": {
            **snapshot,
            "credential_source": credential.source,
            "credential_configured": credential.configured,
            "l1_provenance": credential.provenance or snapshot.get("l1_provenance", ""),
            "data_dir": str(data_dir),
        },
        "selection": {
            "targets": targets,
            "cost_estimate": sum(
                TARGET_COST[name]
                * args.repeat * (args.scenario_count if name == "panel" else 1)
                for name in targets
            ),
            "scenario_count": args.scenario_count,
            "repeat": args.repeat,
            "panel_fanout_concurrency": (
                registry.fanout_concurrency if registry is not None else None
            ),
            "allow_degraded": bool(args.allow_degraded),
        },
        "budget": {
            "max_requests": args.max_requests,
            "attempted": sum(a.request_count for a in accounting_by_target.values()),
        },
        "thresholds": {
            "schema_pass_rate_panel": SCHEMA_PASS_RATE_PANEL,
            "citation_validity_panel": CITATION_VALIDITY_PANEL,
            "max_failure_rate": MAX_FAILURE_RATE,
            "min_tool_call_rate": MIN_TOOL_CALL_RATE,
            "min_schema_passes": MIN_SCHEMA_PASSES,
        },
        "targets": {
            target: result.as_dict(
                accounting_by_target[target], elapsed_by_target.get(target, 0.0),
            )
            for target, result in results.items()
        },
    }

    summary_path, report_path = write_artifacts(
        directory, payload, accounting_by_target, elapsed_by_target,
    )
    print_verdicts(results, accounting_by_target)
    print(f"artifacts: {summary_path}")
    print(f"           {report_path}")

    failed = [target for target, result in results.items() if not result.ok]
    if failed:
        print(f"RESULT: FAIL ({', '.join(failed)})")
        return 1
    print(f"RESULT: PASS ({', '.join(results) or 'no targets'})")
    return 0


async def _maybe_await(value: Any) -> None:
    """Await *value* when it is awaitable (registries differ across versions)."""
    if hasattr(value, "__await__"):
        await value


if __name__ == "__main__":
    raise SystemExit(main())
