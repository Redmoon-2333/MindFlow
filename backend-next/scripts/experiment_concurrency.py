"""Offline concurrency experiment for the LLM layer (optimisation plan 4.4).

Question the script answers: **may ``max_concurrent_requests`` be raised above
the shipped default of 1?** The plan's rule is explicit —

    raise the default only if p95 latency drops by >= 20% *and* the error rate
    does not increase.

So the script runs one fixed scenario set at concurrency 1, 2 and 3 and reports,
per level: panel p50/p95 latency, 429 rate, timeout rate, parse-failure rate, the
observed maximum number of simultaneously in-flight requests, and total wall
time.

It is fully offline:

* no API key, no network — ``httpx.MockTransport`` is installed as the transport
  of every ``httpx.AsyncClient`` the SDK builds, so attempts are simulated;
* the *real* production path is exercised end to end: ``ProviderRegistry`` →
  ``LangChainGateway`` → ``ECNUChatModel`` → per-role ``CompletionPolicy``
  request body. Only the wire is simulated;
* service times and injected faults are a deterministic function of
  ``(scenario, role, occurrence)``, so the three levels serve byte-identical
  workloads and the only variable left is the gate.

The script **measures only** — it never changes the default. Artifacts go to
``data/experiments/<run-id>/`` (repo rule: no temporary reports outside that
directory).

Usage::

    uv run python scripts/experiment_concurrency.py
    uv run python scripts/experiment_concurrency.py --out data/experiments/manual_concurrency
    uv run python scripts/experiment_concurrency.py --levels 1,2,3 --scenarios 8 --repeat 3
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import json
import statistics
import time
import zlib
from collections.abc import Awaitable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from langchain_openai.chat_models import _client_utils

from mindflow.agents.experts import ANALYST, ATTRIBUTION_EXPERTS, CRITIC, MODERATOR
from mindflow.agents.llm_gateway import LangChainGateway
from mindflow.agents.policies import (
    CompletionPolicy,
    policy_for_expert,
    policy_for_role,
)
from mindflow.config import LLMSettings
from mindflow.infrastructure.llm.concurrency import LLMConcurrencyGate
from mindflow.services.llm_observability import (
    GroupStats,
    LLMObservabilityAggregator,
    percentile,
    reset_llm_observability,
)

# ── The fixed experiment configuration ────────────────────────────────────────

#: The panel's six LLM roles, each paired with the model tier it really uses.
#: The moderator is the only reasoner-tier call (``MODERATOR.model == "reasoner"``).
ROLE_TIERS: tuple[tuple[str, Literal["chat", "reasoner"]], ...] = (
    ("analyst", "chat"),
    ("cbt", "chat"),
    ("tmt", "chat"),
    ("emotion", "chat"),
    ("moderator", "reasoner"),
    ("critic", "chat"),
)

#: Concurrency levels measured. 1 is the shipped default and the baseline.
DEFAULT_LEVELS: tuple[int, ...] = (1, 2, 3)

#: Panel rounds. Two independent rounds per scenario make concurrency
#: observable: with more than one round outstanding, a request can be queued
#: behind a round-mate, which is exactly the queueing the gate controls.
DEFAULT_ROUNDS = 2

#: Number of fixed scenarios in the default grid.
DEFAULT_SCENARIOS = 8

#: The plan's acceptance rule.
P95_IMPROVEMENT_TARGET = 0.20

#: Base service time (ms) per scenario — the spread a real provider shows.
_SCENARIO_BASE_MS: tuple[float, ...] = (90.0, 120.0, 160.0, 210.0)

#: Per-role multiplier: the moderator is the slow, high-effort call.
_ROLE_FACTOR: dict[str, float] = {
    "analyst": 1.0,
    "cbt": 1.25,
    "tmt": 1.15,
    "emotion": 1.35,
    "moderator": 2.4,
    "critic": 0.85,
}

#: Deterministic fault plan: (scenario index, role) -> fault kind. Everything
#: not listed succeeds. These are the failure modes the default-concurrency
#: decision is actually about: a rate limit, a deadline, and an unparseable body.
_FAULT_PLAN: dict[tuple[int, str], str] = {
    (0, "critic"): "rate_limit_429",
    (1, "moderator"): "rate_limit_429",
    (2, "tmt"): "timeout",
    (3, "emotion"): "parse_failure",
    (4, "cbt"): "rate_limit_429",
    (5, "analyst"): "timeout",
    (6, "critic"): "parse_failure",
    (7, "tmt"): "rate_limit_429",
}

_SYSTEM_PROMPT = (
    "你是一个行为分析专家。请依据证据包输出 JSON 结论。"
)
_USER_PROMPT = "证据包：{\"focus_score\": 0.62, \"switch_rate\": 11.4}"

#: Text that is deliberately not valid JSON — the parse-failure fault.
_INVALID_JSON_CONTENT = "{ this is not json"

#: Per-role tag for the simulated transport.
#:
#: The transport must know which role a request belongs to, because the fault
#: plan and the service-time model are keyed by role. It cannot be recovered
#: from the request body: cbt / tmt / emotion / rebuttal all carry the same
#: output ceiling (1000 / 800), so a body-only heuristic silently collapses
#: them. A ContextVar is set by the harness right before each call and read by
#: the transport — the body itself stays exactly what production sends.
_CALL_ROLE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "mindflow_experiment_role", default="",
)


async def call_role(
    gateway: LangChainGateway,
    role: str,
    model: Literal["chat", "reasoner"],
    policy: CompletionPolicy | None,
) -> str:
    """One expert call, tagged with the role it runs as."""
    token = _CALL_ROLE.set(role)
    try:
        content: str = await gateway.complete(
            _SYSTEM_PROMPT, _USER_PROMPT, model=model, policy=policy,
        )
        return content
    finally:
        _CALL_ROLE.reset(token)


def current_call_role() -> str:
    """The role of the call being served (``""`` outside a harness call)."""
    return _CALL_ROLE.get()


# ── Deterministic workload model ──────────────────────────────────────────────


@dataclass(frozen=True)
class Fault:
    """One injected failure and what it should look like."""

    kind: str
    attempt: int


def fault_for(scenario_index: int, role: str, occurrence: int) -> str:
    """The fault kind for one attempt, or ``""`` for a successful response.

    A rate limit or a timeout fails the first attempt for its key and then
    succeeds, so the retry machinery is exercised without turning the whole
    scenario into a degradation run.
    """
    kind = _FAULT_PLAN.get((scenario_index % len(_SCENARIO_BASE_MS), role), "")
    if not kind:
        return ""
    if kind == "parse_failure":
        # A well-formed 200 with a body the caller cannot parse: retrying the
        # request cannot fix it, so it stays broken at every occurrence.
        return kind
    return kind if occurrence == 0 else ""


def _stable_offset(seed: str, modulus: int) -> int:
    """Deterministic non-negative offset in ``[0, modulus)``.

    ``hash()`` is salted per process, so it cannot be used here: the workload
    must be reproducible from one run (and one concurrency level) to the next.
    """
    return zlib.crc32(seed.encode("utf-8")) % modulus


def service_time_ms(scenario_index: int, role: str) -> float:
    """Deterministic simulated service time (ms) for one request.

    A fixed multiplier per role would make every request of a role take exactly
    the same time, which is unrealistic and would let the gate's effect be
    predicted by hand instead of measured. A small deterministic jitter derived
    from ``(scenario, role)`` keeps the workload identical across concurrency
    levels while still giving each role a plausible spread.
    """
    base = _SCENARIO_BASE_MS[scenario_index % len(_SCENARIO_BASE_MS)]
    jitter = 1.0 + (_stable_offset(f"{scenario_index}:{role}", 21) - 10) / 100.0
    return base * _ROLE_FACTOR.get(role, 1.0) * jitter


def deadline_ms(scenario_index: int, role: str) -> float:
    """A simulated deadline the ``timeout`` fault cannot beat."""
    return service_time_ms(scenario_index, role) - 1.0


# ── The simulated wire ────────────────────────────────────────────────────────


@dataclass
class RequestOutcome:
    """One attempt as the experiment observed it."""

    concurrency: int
    scenario: str
    role: str
    model_tier: str
    occurrence: int
    fault: str
    latency_ms: float
    status_code: int
    ok: bool
    retryable: bool
    parse_failure: bool

    @property
    def is_rate_limited(self) -> bool:
        return self.status_code == 429

    @property
    def is_timeout(self) -> bool:
        return self.fault == "timeout" and self.status_code == 0


@dataclass
class TransportRecorder:
    """In-flight bookkeeping for the simulated transport."""

    active: int = 0
    peak_in_flight: int = 0
    attempts: int = 0

    def enter(self) -> None:
        self.active += 1
        self.attempts += 1
        self.peak_in_flight = max(self.peak_in_flight, self.active)

    def leave(self) -> None:
        self.active -= 1


class ScenarioTransport:
    """Serves one scenario's requests with deterministic latency and faults.

    Request ordering is not assumed: the response is a function of the role the
    call was made as (see :func:`call_role`) and of a per-role attempt counter.
    """

    def __init__(self, *, concurrency: int, scenario_index: int, scenario_id: str) -> None:
        self.concurrency = concurrency
        self.scenario_index = scenario_index
        self.scenario_id = scenario_id
        self.recorder = TransportRecorder()
        self.outcomes: list[RequestOutcome] = []
        self._occurrences: dict[str, int] = {}

    # ── The handler ──────────────────────────────────────────────────────

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content or b"{}")
        role = current_call_role()
        if not role:  # pragma: no cover - guards a mis-wired harness
            raise AssertionError("request arrived without a role tag")
        tier = dict(ROLE_TIERS)[role]
        occurrence = self._occurrences.get(role, 0)
        self._occurrences[role] = occurrence + 1
        fault = fault_for(self.scenario_index, role, occurrence)

        self.recorder.enter()
        started = time.perf_counter()
        try:
            if fault == "timeout":
                await asyncio.sleep(deadline_ms(self.scenario_index, role) / 1000.0)
                self._record(
                    role, tier, occurrence, fault, started,
                    status_code=0, ok=False, retryable=True, parse_failure=False,
                )
                raise httpx.ReadTimeout("simulated deadline exceeded", request=request)

            await asyncio.sleep(service_time_ms(self.scenario_index, role) / 1000.0)

            if fault == "rate_limit_429":
                self._record(
                    role, tier, occurrence, fault, started,
                    status_code=429, ok=False, retryable=True, parse_failure=False,
                )
                return httpx.Response(
                    429,
                    headers={"Retry-After": "1"},
                    json={"error": {"message": "simulated rate limit"}},
                )

            content = _INVALID_JSON_CONTENT if fault == "parse_failure" else "{}"
            parse_failure = False
            if fault == "parse_failure":
                try:
                    json.loads(content)
                except json.JSONDecodeError:
                    parse_failure = True
            self._record(
                role, tier, occurrence, fault, started,
                status_code=200, ok=True, retryable=False, parse_failure=parse_failure,
            )
            return httpx.Response(200, json={
                "id": "simulated",
                "object": "chat.completion",
                "model": payload.get("model", "ecnu-max"),
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }],
                "usage": {
                    "prompt_tokens": 900,
                    "completion_tokens": 120,
                    "total_tokens": 1020,
                    "completion_tokens_details": {"reasoning_tokens": 40},
                },
            })
        finally:
            self.recorder.leave()

    def _record(
        self,
        role: str,
        tier: str,
        occurrence: int,
        fault: str,
        started: float,
        *,
        status_code: int,
        ok: bool,
        retryable: bool,
        parse_failure: bool,
    ) -> None:
        self.outcomes.append(RequestOutcome(
            concurrency=self.concurrency,
            scenario=self.scenario_id,
            role=role,
            model_tier=tier,
            occurrence=occurrence,
            fault=fault,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            status_code=status_code,
            ok=ok,
            retryable=retryable,
            parse_failure=parse_failure,
        ))


def _policy_for_key(role: str) -> CompletionPolicy | None:
    """Resolve a role key to its policy (expert instances included)."""
    experts = {
        "analyst": ANALYST,
        "cbt": ATTRIBUTION_EXPERTS[0],
        "tmt": ATTRIBUTION_EXPERTS[1],
        "emotion": ATTRIBUTION_EXPERTS[2],
        "critic": CRITIC,
        "moderator": MODERATOR,
    }
    expert = experts.get(role)
    if expert is not None:
        return policy_for_expert(expert)
    return policy_for_role(role)


# ── Running one concurrency level ─────────────────────────────────────────────


@dataclass
class LevelResult:
    """Everything measured at one concurrency level."""

    concurrency: int
    rounds: int
    scenarios: int
    calls: int
    attempts: int
    wall_time_s: float
    panel_latencies_ms: list[float]
    p50_ms: float
    p95_ms: float
    mean_ms: float
    max_ms: float
    rate_limit_429: int
    timeouts: int
    parse_failures: int
    failures: int
    gate_reported_max_in_flight: int
    gate_reported_acquired: int
    observed_peak_in_flight: int
    gate_snapshot: dict[str, int | None] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return max(1, self.attempts)

    @property
    def rate_limit_rate(self) -> float:
        return self.rate_limit_429 / self.total

    @property
    def timeout_rate(self) -> float:
        return self.timeouts / self.total

    @property
    def parse_failure_rate(self) -> float:
        return self.parse_failures / self.total

    @property
    def failure_rate(self) -> float:
        return self.failures / self.total

    def metrics(self) -> dict[str, Any]:
        return {
            "concurrency": self.concurrency,
            "rounds": self.rounds,
            "scenarios": self.scenarios,
            "logical_calls": self.calls,
            "http_attempts": self.attempts,
            "wall_time_s": round(self.wall_time_s, 4),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "rate_limit_429": self.rate_limit_429,
            "timeouts": self.timeouts,
            "parse_failures": self.parse_failures,
            "failures": self.failures,
            "rate_limit_rate": round(self.rate_limit_rate, 6),
            "timeout_rate": round(self.timeout_rate, 6),
            "parse_failure_rate": round(self.parse_failure_rate, 6),
            "failure_rate": round(self.failure_rate, 6),
            "observed_peak_in_flight": self.observed_peak_in_flight,
            "gate_reported_max_in_flight": self.gate_reported_max_in_flight,
            "gate_reported_acquired": self.gate_reported_acquired,
        }


def _settings(concurrency: int) -> LLMSettings:
    """ECNU-shaped settings for the harness (no key, no network)."""
    return LLMSettings(
        api_key="offline-experiment",
        base_url="https://chat.ecnu.edu.cn/open/api/v1",
        model="ecnu-max",
        provider="ecnu",
        thinking_enabled=True,
        reasoning_effort="max",
        timeout_s=30,
        max_retries=1,
        max_output_tokens=16384,
        max_concurrent_requests=concurrency,
    )


async def _run_scenario(
    gateway: LangChainGateway,
    scenario_id: str,
    rounds: int,
) -> None:
    """One scenario: ``rounds`` x six expert calls, all in flight together."""
    calls: list[Awaitable[str]] = []
    for _ in range(rounds):
        for role, tier in ROLE_TIERS:
            calls.append(call_role(gateway, role, tier, _policy_for_key(role)))
    await asyncio.gather(*calls, return_exceptions=True)


async def run_level(
    level: int,
    *,
    scenarios: int,
    rounds: int,
) -> tuple[LevelResult, list[RequestOutcome], LLMObservabilityAggregator]:
    """Run the fixed scenario set once at *level* and collect every metric."""
    settings = _settings(level)
    gate = LLMConcurrencyGate(settings.max_concurrent_requests)
    aggregator = LLMObservabilityAggregator()

    transports: list[ScenarioTransport] = []
    per_scenario_ms: list[float] = []
    calls = 0

    original_init = httpx.AsyncClient.__init__
    _client_utils._cached_async_httpx_client.cache_clear()
    _client_utils._cached_sync_httpx_client.cache_clear()

    started_wall = time.perf_counter()
    gateway = LangChainGateway(
        api_key=settings.api_key,
        base_url=settings.base_url,
        llm_settings=settings,
        concurrency=gate,
        observability=aggregator,
        max_retries=settings.max_retries,
    )

    current: dict[str, ScenarioTransport] = {}

    def patched_init(client: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
        kwargs["transport"] = httpx.MockTransport(_dispatch)
        kwargs["trust_env"] = False
        original_init(client, *args, **kwargs)

    async def _dispatch(request: httpx.Request) -> httpx.Response:
        # Exactly one transport is live at a time: scenarios run sequentially,
        # their expert calls concurrently, so the role tag identifies the caller.
        transport = current["active"]
        return await transport(request)

    # ``setattr`` rather than attribute assignment: the patched signature is
    # deliberately permissive, and this is a test-style seam, not production.
    setattr(httpx.AsyncClient, "__init__", patched_init)  # noqa: B010
    try:
        for index in range(scenarios):
            scenario_id = f"SCN-{index + 1:02d}"
            transport = ScenarioTransport(
                concurrency=level, scenario_index=index, scenario_id=scenario_id,
            )
            current["active"] = transport
            transports.append(transport)
            scenario_started = time.perf_counter()
            await _run_scenario(gateway, scenario_id, rounds)
            per_scenario_ms.append((time.perf_counter() - scenario_started) * 1000.0)
            calls += rounds * len(ROLE_TIERS)
    finally:
        setattr(httpx.AsyncClient, "__init__", original_init)  # noqa: B010
        await gateway.close()

    wall_time_s = time.perf_counter() - started_wall
    outcomes = [outcome for transport in transports for outcome in transport.outcomes]
    peak = max((transport.recorder.peak_in_flight for transport in transports), default=0)
    attempts = sum(transport.recorder.attempts for transport in transports)

    latencies = sorted(per_scenario_ms)
    snapshot = gate.snapshot()
    result = LevelResult(
        concurrency=level,
        rounds=rounds,
        scenarios=scenarios,
        calls=calls,
        attempts=attempts,
        wall_time_s=wall_time_s,
        panel_latencies_ms=latencies,
        p50_ms=percentile(latencies, 0.50),
        p95_ms=percentile(latencies, 0.95),
        mean_ms=statistics.fmean(latencies) if latencies else 0.0,
        max_ms=max(latencies, default=0.0),
        rate_limit_429=sum(1 for o in outcomes if o.is_rate_limited),
        timeouts=sum(1 for o in outcomes if o.is_timeout),
        parse_failures=sum(1 for o in outcomes if o.parse_failure),
        failures=sum(1 for o in outcomes if not o.ok or o.parse_failure),
        gate_reported_max_in_flight=int(snapshot["max_in_flight"] or 0),
        gate_reported_acquired=int(snapshot["acquired"] or 0),
        observed_peak_in_flight=peak,
        gate_snapshot=snapshot,
    )
    return result, outcomes, aggregator


# ── Reporting ─────────────────────────────────────────────────────────────────


def evaluate_rule(baseline: LevelResult, candidate: LevelResult) -> dict[str, Any]:
    """Apply the plan's rule to one candidate level."""
    drop = (
        (baseline.p95_ms - candidate.p95_ms) / baseline.p95_ms
        if baseline.p95_ms > 0 else 0.0
    )
    error_delta = candidate.failure_rate - baseline.failure_rate
    return {
        "concurrency": candidate.concurrency,
        "baseline_p95_ms": round(baseline.p95_ms, 3),
        "candidate_p95_ms": round(candidate.p95_ms, 3),
        "p95_drop": round(drop, 6),
        "meets_p95_target": drop >= P95_IMPROVEMENT_TARGET,
        "baseline_error_rate": round(baseline.failure_rate, 6),
        "candidate_error_rate": round(candidate.failure_rate, 6),
        "error_rate_delta": round(error_delta, 6),
        "no_error_increase": error_delta <= 0.0,
        "passes_rule": drop >= P95_IMPROVEMENT_TARGET and error_delta <= 0.0,
    }


def _verdict(baseline: LevelResult, evaluations: Sequence[dict[str, Any]]) -> str:
    winners = [evaluation for evaluation in evaluations if evaluation["passes_rule"]]
    if not winners:
        detail = "; ".join(
            f"c={e['concurrency']}: p95 {e['candidate_p95_ms']:.1f}ms "
            f"({e['p95_drop']:+.1%}), 错误率 Δ{e['error_rate_delta']:+.2%}"
            for e in evaluations
        )
        return (
            f"方案规则未通过，保持 max_concurrent_requests=1（基线 p95 "
            f"{baseline.p95_ms:.1f}ms）。{detail}。"
        )
    best = max(winners, key=lambda e: e["p95_drop"])
    return (
        f"方案规则通过：c={best['concurrency']} 使 p95 下降 {best['p95_drop']:.1%}"
        f"（{best['baseline_p95_ms']:.1f}ms → {best['candidate_p95_ms']:.1f}ms），"
        f"且错误率未上升（Δ{best['error_rate_delta']:+.2%}）。"
        "是否修改默认值仍属人工决策；本次实验只做测量。"
    )


_HEADER = (
    "| concurrency | attempts | panel p50 (ms) | panel p95 (ms) | 429 rate | "
    "timeout rate | parse-failure rate | failure rate | observed max in-flight | "
    "gate max in-flight | wall time (s) |"
)
_SEPARATOR = "|---|---|---|---|---|---|---|---|---|---|---|"


def _table_row(result: LevelResult) -> str:
    return (
        f"| {result.concurrency} | {result.attempts} | {result.p50_ms:.1f} | "
        f"{result.p95_ms:.1f} | {result.rate_limit_rate:.1%} | "
        f"{result.timeout_rate:.1%} | {result.parse_failure_rate:.1%} | "
        f"{result.failure_rate:.1%} | {result.observed_peak_in_flight} | "
        f"{result.gate_reported_max_in_flight} | {result.wall_time_s:.3f} |"
    )


def _comparison_markdown(
    *,
    run_id: str,
    results: Sequence[LevelResult],
    evaluations: Sequence[dict[str, Any]],
    verdict: str,
    scenario_count: int,
    rounds: int,
    invariants: dict[str, Any],
) -> str:
    baseline = results[0]
    lines = [
        "# LLM 并发实验（离线，MockTransport）",
        "",
        f"- run: `{run_id}`",
        f"- 场景数: {scenario_count}，每场景轮次: {rounds}，"
        f"每轮 {len(ROLE_TIERS)} 个角色调用",
        f"- 并发档位: {', '.join(str(r.concurrency) for r in results)}",
        f"- 交付默认值: `max_concurrent_requests={baseline.concurrency}`（脚本不修改配置）",
        f"- 规则: p95 下降 ≥ {P95_IMPROVEMENT_TARGET:.0%} 且错误率不上升",
        "",
        "## 1 vs 2 vs 3 对比",
        "",
        _HEADER,
        _SEPARATOR,
        *(_table_row(result) for result in results),
        "",
        "## 方案规则判定",
        "",
        "| candidate | baseline p95 (ms) | candidate p95 (ms) | p95 drop | "
        "error rate Δ | 判定 |",
        "|---|---|---|---|---|---|",
    ]
    for evaluation in evaluations:
        lines.append(
            f"| {evaluation['concurrency']} | {evaluation['baseline_p95_ms']:.1f} | "
            f"{evaluation['candidate_p95_ms']:.1f} | {evaluation['p95_drop']:+.1%} | "
            f"{evaluation['error_rate_delta']:+.2%} | "
            f"{'通过' if evaluation['passes_rule'] else '不通过'} |"
        )
    lines.extend([
        "",
        f"**结论**: {verdict}",
        "",
        "## 工作负载与不变式",
        "",
        f"- 每个档位的 HTTP 尝试总数: "
        f"{', '.join(f'c={r.concurrency}:{r.attempts}' for r in results)}",
        f"- 每次尝试的回答角色分布一致: {invariants['role_mix_identical']}",
        f"- 每次尝试的故障类型分布一致: {invariants['fault_mix_identical']}",
        f"- 门统计的可观测并发 == 传输层实测并发: "
        f"{invariants['peak_matches_gate']}",
        f"- 实测并发从未超过档位上限: {invariants['peak_within_limit']}",
        "",
    ])
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────


def invariants_hold(
    results: Sequence[LevelResult],
    outcomes_by_level: dict[int, list[RequestOutcome]],
) -> dict[str, Any]:
    """Prove the levels really served the same workload, and the gate is honest."""
    role_mixes = {
        level: sorted(
            f"{o.scenario}:{o.role}:{o.occurrence}:{o.fault}"
            for o in outcomes
        )
        for level, outcomes in outcomes_by_level.items()
    }
    reference = role_mixes[results[0].concurrency]
    return {
        "role_mix_identical": all(mix == reference for mix in role_mixes.values()),
        "fault_mix_identical": all(mix == reference for mix in role_mixes.values()),
        "peak_matches_gate": all(
            result.observed_peak_in_flight == result.gate_reported_max_in_flight
            for result in results
        ),
        "peak_within_limit": all(
            result.observed_peak_in_flight <= result.concurrency for result in results
        ),
    }


async def run_experiment(
    *,
    levels: Sequence[int],
    scenarios: int,
    rounds: int,
) -> dict[str, Any]:
    """Run every level and assemble the report payload."""
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_concurrency")

    results: list[LevelResult] = []
    outcomes_by_level: dict[int, list[RequestOutcome]] = {}
    observability: dict[str, Any] = {}

    for level in levels:
        reset_llm_observability()
        result, outcomes, aggregator = await run_level(
            level, scenarios=scenarios, rounds=rounds,
        )
        results.append(result)
        outcomes_by_level[level] = outcomes
        snapshot = aggregator.snapshot()
        observability[f"c{level}"] = {
            "total": _stats_dict(snapshot["total"]),
            "by_role": {role: _stats_dict(stats) for role, stats in snapshot["by_role"].items()},
        }

    results.sort(key=lambda result: result.concurrency)
    baseline = results[0]
    evaluations = [evaluate_rule(baseline, result) for result in results[1:]]
    invariants = invariants_hold(results, outcomes_by_level)
    verdict = _verdict(baseline, evaluations)

    return {
        "run_id": run_id,
        "levels": [result.concurrency for result in results],
        "scenario_count": scenarios,
        "rounds": rounds,
        "roles": [role for role, _ in ROLE_TIERS],
        "rule": {
            "p95_improvement_target": P95_IMPROVEMENT_TARGET,
            "max_error_rate_increase": 0.0,
        },
        "delivered_default_concurrency": LLMSettings().max_concurrent_requests,
        "results": [result.metrics() for result in results],
        "evaluations": evaluations,
        "invariants": invariants,
        "verdict": verdict,
        "observability": observability,
        "_results": results,
        "_outcomes": outcomes_by_level,
    }


def _stats_dict(stats: GroupStats) -> dict[str, Any]:
    return {
        "count": stats["count"],
        "requests": stats["requests"],
        "failure_rate": round(stats["failure_rate"], 6),
        "p50_latency_ms": round(stats["p50_latency_ms"], 3),
        "p95_latency_ms": round(stats["p95_latency_ms"], 3),
        "mean_latency_ms": round(stats["mean_latency_ms"], 3),
        "mean_cost_units": round(stats["mean_cost_units"], 8),
        "total_cost_units": round(stats["total_cost_units"], 8),
        "mean_retries": round(stats["mean_retries"], 4),
        "retry_rate": round(stats["retry_rate"], 6),
        "input_tokens": stats["input_tokens"],
        "output_tokens": stats["output_tokens"],
        "reasoning_tokens": stats["reasoning_tokens"],
    }


def write_artifacts(payload: dict[str, Any], out_dir: Path) -> Path:
    """Write manifest/results/summary/comparison under *out_dir*."""
    results: list[LevelResult] = payload["_results"]
    outcomes: dict[int, list[RequestOutcome]] = payload["_outcomes"]
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": payload["run_id"],
        "created_at": datetime.now(UTC).isoformat(),
        "kind": "llm_concurrency_offline",
        "network": "none (httpx.MockTransport)",
        "api_key_required": False,
        "levels": payload["levels"],
        "scenarios": payload["scenario_count"],
        "rounds": payload["rounds"],
        "roles": payload["roles"],
        "delivered_default_concurrency": payload["delivered_default_concurrency"],
        "rule": payload["rule"],
        "producer": "scripts/experiment_concurrency.py",
    }
    _write_json(out_dir / "manifest.json", manifest)

    summary = {
        key: value for key, value in payload.items()
        if not key.startswith("_")
    }
    _write_json(out_dir / "summary.json", summary)

    per_attempt = {
        f"c{level}": [asdict(outcome) for outcome in level_outcomes]
        for level, level_outcomes in sorted(outcomes.items())
    }
    _write_json(out_dir / "attempts.json", per_attempt)

    _write_json(out_dir / "panel_latencies.json", {
        f"c{result.concurrency}": [round(value, 3) for value in result.panel_latencies_ms]
        for result in results
    })

    markdown = _comparison_markdown(
        run_id=payload["run_id"],
        results=results,
        evaluations=payload["evaluations"],
        verdict=payload["verdict"],
        scenario_count=payload["scenario_count"],
        rounds=payload["rounds"],
        invariants=payload["invariants"],
    )
    (out_dir / "comparison.md").write_text(markdown, encoding="utf-8")
    return out_dir


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels", type=str, default=",".join(str(level) for level in DEFAULT_LEVELS),
        help="Comma-separated concurrency levels (default: 1,2,3)",
    )
    parser.add_argument(
        "--scenarios", type=int, default=DEFAULT_SCENARIOS,
        help="Number of fixed scenarios to run (default: 8)",
    )
    parser.add_argument(
        "--rounds", type=int, default=DEFAULT_ROUNDS,
        help="Panel rounds per scenario (default: 2)",
    )
    parser.add_argument(
        "--out", type=str, default="", dest="out",
        help="Artifact directory (default: data/experiments/<run-id>)",
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Repeat the whole grid N times and keep the last run (stability check)",
    )
    args = parser.parse_args()

    levels = [int(part) for part in args.levels.split(",") if part.strip()]
    if not levels or min(levels) < 1:
        parser.error("--levels must be positive integers")

    payload: dict[str, Any] = {}
    for _ in range(max(1, args.repeat)):
        payload = asyncio.run(run_experiment(
            levels=levels, scenarios=args.scenarios, rounds=args.rounds,
        ))

    out_dir = Path(args.out) if args.out else Path("data/experiments") / payload["run_id"]
    write_artifacts(payload, out_dir)

    print(_comparison_markdown(
        run_id=payload["run_id"],
        results=payload["_results"],
        evaluations=payload["evaluations"],
        verdict=payload["verdict"],
        scenario_count=payload["scenario_count"],
        rounds=payload["rounds"],
        invariants=payload["invariants"],
    ))
    print(f"Artifacts: {out_dir}")


if __name__ == "__main__":
    main()
