"""ChatGraph LLM-request metrics parity + offline smoke-script plumbing.

Two contracts are pinned here.

**1. Chat metrics parity.** ``ChatGraph`` has two generation calls — the normal
``model_call`` turn and the forbidden-word ``correction_loop`` retry — and both
must emit the *same* aggregated, PII-free record the panel path emits
(``mindflow.services.llm_observability``): role, graph node, provider, model,
queue/HTTP/total latency, status, retry count, and provider-reported token
usage. The sentinel test drives prompt text, an API key, and a provider body
containing forbidden reasoning content through the real call path, then scans
the records and the default aggregator's snapshot for all three.

**2. Smoke script plumbing.** ``scripts/smoke_live_llm.py`` must never run
automatically and must refuse anything without ``--yes``; its budget guard must
refuse *before* a request reaches a transport. ``--dry-run`` exercises the whole
pipeline with fake transports, and these tests drive that mode end to end,
including artifact emission.

Offline only: fake models, fake gateways, and the script's own dry-run fakes.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from mindflow.graph.chat_graph import CHAT_SYSTEM_PROMPT, ChatGraph
from mindflow.infrastructure.security.crisis_detector import CrisisDetector, CrisisLevel
from mindflow.services.llm_observability import (
    ALLOWED_FIELDS,
    default_aggregator,
    reset_llm_observability,
)

#: Strings that must never survive into recorded metadata.
PROMPT_SENTINEL = "SENTINEL_PROMPT_聊天窗口标题.docx"
KEY_SENTINEL = "SENTINEL_API_KEY_sk-live-0000"
BODY_SENTINEL = "SENTINEL_BODY_reasoning_content_不要泄漏"

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(autouse=True)
def _clean_default_aggregator() -> None:
    """The default aggregator is process-wide; never leak records into the suite."""
    reset_llm_observability()
    yield
    reset_llm_observability()


@pytest.fixture(scope="module")
def smoke() -> Any:
    """The smoke script imported as a module (it is not part of the package)."""
    path = _SCRIPTS_DIR / "smoke_live_llm.py"
    spec = importlib.util.spec_from_file_location("smoke_live_llm_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


class _ScriptedChatModel:
    """Scripted chat model with usage metadata and per-call prompt capture."""

    def __init__(
        self, responses: list[AIMessage], *, model_name: str = "sentinel-model",
    ) -> None:
        self._responses = list(responses)
        self._index = 0
        self.model_name = model_name
        self.seen_messages: list[list[Any]] = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> _ScriptedChatModel:
        _ = (tools, kwargs)
        return self

    async def ainvoke(self, messages: list[Any], **kwargs: Any) -> AIMessage:
        _ = kwargs
        self.seen_messages.append(list(messages))
        if not self._responses:
            return AIMessage(content="")
        response = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return response


class _InMemoryChatRepo:
    """Minimal ``ChatRepository`` stand-in that never touches a database."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self._counter = 0

    async def append(
        self, session_id: str, role: str, content: str, *,
        user_id: int = 1, message_id: str | None = None,
    ) -> dict[str, Any]:
        self._counter += 1
        row = {
            "id": message_id or f"row-{self._counter}",
            "user_id": user_id,
            "session_id": session_id,
            "role": role,
            "content": content,
        }
        self.rows.append(row)
        if message_id and message_id in [r["id"] for r in self.rows]:
            # Keep the durable turn_id as the row id (mirrors production).
            row["id"] = message_id
        return dict(row)

    async def recent(
        self, session_id: str, *, limit: int = 20, user_id: int | None = None,
    ) -> list[dict[str, Any]]:
        rows = [
            dict(row) for row in self.rows
            if row["session_id"] == session_id
            and (user_id is None or row["user_id"] == user_id)
        ]
        return rows[-limit:]


def _chat_graph(model: Any, repo: Any | None = None) -> ChatGraph:
    detector = MagicMock(spec=CrisisDetector)
    detector.scan.return_value = (CrisisLevel.NONE, None)
    return ChatGraph(
        chat_repo=repo or _InMemoryChatRepo(),
        crisis_detector=detector,
        model=model,
        tools=[],
        tool_adapters=[],
        provider="generic",
    )


def _answer(content: str, *, reasoning: int = 0) -> AIMessage:
    return AIMessage(
        content=content,
        usage_metadata={
            "input_tokens": 120,
            "output_tokens": 30,
            "total_tokens": 150,
            "output_token_details": {"reasoning": reasoning},
        },
    )


def _records_json() -> str:
    return json.dumps(
        [record.as_dict() for record in default_aggregator().records()],
        ensure_ascii=False,
        default=str,
    )


def _snapshot_json() -> str:
    return json.dumps(default_aggregator().snapshot(), ensure_ascii=False, default=str)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Normal generation produces a record
# ═══════════════════════════════════════════════════════════════════════════════


async def test_normal_generation_emits_one_chat_record() -> None:
    """A tool-less turn is one measured record with chat labels and real usage."""
    model = _ScriptedChatModel([_answer("根据行为数据，建议先做一次 25 分钟冲刺。")])
    graph = _chat_graph(model)

    result = await graph.ask(user_id=1, session_id="s1", message="我该怎么做？")

    assert result.answer == "根据行为数据，建议先做一次 25 分钟冲刺。"
    records = default_aggregator().records()
    assert len(records) == 1
    record = records[0]
    assert record.graph == "chat"
    assert record.node == "model_call"
    assert record.role == "chat"
    assert record.provider == "generic"
    assert record.model == "sentinel-model"
    assert record.ok is True
    assert record.failed is False
    assert record.retry_count == 0
    assert record.http_status_class == "2xx"
    assert record.error_category == ""
    assert record.input_tokens == 120
    assert record.output_tokens == 30
    assert set(record.as_dict()) == ALLOWED_FIELDS


async def test_normal_generation_latency_is_non_negative_and_total_covers_http() -> None:
    """Latency fields are non-negative and the total is at least the HTTP span."""
    model = _ScriptedChatModel([_answer("好的。")])
    graph = _chat_graph(model)

    await graph.ask(user_id=1, session_id="s1", message="你好")

    record = default_aggregator().records()[0]
    assert record.queue_latency_ms >= 0.0
    assert record.http_latency_ms >= 0.0
    assert record.total_latency_ms >= 0.0
    assert record.total_latency_ms >= record.http_latency_ms
    assert record.is_event is False
    assert record.recorded_at > 0.0


async def test_provider_metadata_usage_path_is_honoured() -> None:
    """A provider that only fills ``response_metadata`` still reports tokens."""
    response = AIMessage(
        content="根据数据，建议休息。",
        response_metadata={
            "token_usage": {
                "prompt_tokens": 64,
                "completion_tokens": 12,
                "completion_tokens_details": {"reasoning_tokens": 5},
            },
        },
    )
    model = _ScriptedChatModel([response])
    graph = _chat_graph(model)

    await graph.ask(user_id=1, session_id="s1", message="建议")

    record = default_aggregator().records()[0]
    assert (record.input_tokens, record.output_tokens, record.reasoning_tokens) == (64, 12, 5)


async def test_missing_usage_records_zero_not_an_estimate() -> None:
    """No provider usage → zeros. Character counts are never turned into tokens."""
    model = _ScriptedChatModel([AIMessage(content="一段很长的回答" * 50)])
    graph = _chat_graph(model)

    await graph.ask(user_id=1, session_id="s1", message="长回答")

    record = default_aggregator().records()[0]
    assert (record.input_tokens, record.output_tokens, record.reasoning_tokens) == (0, 0, 0)
    assert record.cost_units == 0.0


async def test_model_failure_is_recorded_as_not_ok() -> None:
    """A failing generation is a measured request too (failure rate must move)."""

    class _StatusError(Exception):
        status_code = 503

    model = _ScriptedChatModel([])
    model.ainvoke = AsyncMock(side_effect=_StatusError("provider down"))
    graph = _chat_graph(model)

    result = await graph.ask(user_id=1, session_id="s1", message="你好")

    assert result.degraded is True
    record = default_aggregator().records()[0]
    assert record.ok is False
    assert record.failed is True
    assert record.http_status_class == "5xx"
    assert record.error_category == "transport"
    snapshot = default_aggregator().snapshot()
    assert snapshot["total"]["failure_rate"] == 1.0
    assert snapshot["by_graph"]["chat"]["count"] == 1


async def test_budget_refusal_is_an_event_not_a_measured_request() -> None:
    """A refusal before the transport is recorded, but never as a provider call."""
    from mindflow.graph.chat_graph import _is_budget_refusal

    class BudgetRefusedError(RuntimeError):
        """Named like the smoke harness's own refusal type."""

    model = _ScriptedChatModel([])
    model.ainvoke = AsyncMock(side_effect=BudgetRefusedError("budget exhausted"))
    graph = _chat_graph(model)

    result = await graph.ask(user_id=1, session_id="s1", message="你好")

    assert result.degraded is True
    record = default_aggregator().records()[0]
    assert _is_budget_refusal(BudgetRefusedError("x")) is True
    assert record.ok is False
    assert record.fallback_reason == "request_budget_refused"
    assert record.error_category == "", "a caller budget is not a transport error"
    assert record.total_latency_ms == 0.0
    assert record.is_event is True, "no provider request was measured"
    stats = default_aggregator().snapshot()["by_graph"]["chat"]
    assert stats["count"] == 1
    assert stats["requests"] == 0, "event records are excluded from latency stats"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Correction generation produces its own, distinguishable record
# ═══════════════════════════════════════════════════════════════════════════════


async def test_correction_generation_emits_its_own_record() -> None:
    """Both generations record: normal (model_call) then correction (retry 1)."""
    model = _ScriptedChatModel([
        _answer("根据诊断结果，需要调整作息。", reasoning=3),  # forbidden word
        _answer("根据行为数据，建议调整作息。", reasoning=7),  # clean retry
    ])
    graph = _chat_graph(model)

    result = await graph.ask(user_id=1, session_id="s1", message="分析")

    assert "诊断" not in result.answer
    records = default_aggregator().records()
    assert len(records) == 2, "both generations are measured requests"

    normal, correction = records
    for record in records:
        assert record.graph == "chat"
        assert record.role == "chat"
        assert record.provider == "generic"
        assert record.model == "sentinel-model"
        assert record.total_latency_ms >= record.http_latency_ms >= 0.0
        assert set(record.as_dict()) == ALLOWED_FIELDS

    assert normal.node == "model_call"
    assert normal.retry_count == 0
    assert correction.node == "correction_loop"
    assert correction.retry_count == 1
    # Distinguishable on the record itself, with the correction's own usage.
    assert correction.node != normal.node
    assert correction.retry_count != normal.retry_count
    assert (correction.input_tokens, correction.output_tokens, correction.reasoning_tokens) == (
        120, 30, 7,
    )
    assert normal.reasoning_tokens == 3

    snapshot = default_aggregator().snapshot()
    assert snapshot["total"]["count"] == 2
    assert snapshot["by_graph"]["chat"]["count"] == 2
    assert snapshot["by_graph"]["chat"]["requests"] == 2


async def test_failed_correction_retry_is_recorded_as_not_ok() -> None:
    """A correction call that raises is recorded with ok=False and retry_count=1."""

    class _StatusError(Exception):
        status_code = 429

    model = _ScriptedChatModel([_answer("诊断结果如下。")])
    calls = {"n": 0}
    original = model.ainvoke

    async def flaky(messages: list[Any], **kwargs: Any) -> AIMessage:
        calls["n"] += 1
        if calls["n"] == 2:
            raise _StatusError("rate limited")
        return await original(messages, **kwargs)

    model.ainvoke = flaky  # type: ignore[method-assign]
    graph = _chat_graph(model)

    await graph.ask(user_id=1, session_id="s1", message="分析")

    normal, correction = default_aggregator().records()
    assert normal.ok is True and normal.node == "model_call"
    assert correction.ok is False
    assert correction.node == "correction_loop"
    assert correction.retry_count == 1
    assert correction.http_status_class == "4xx"
    assert default_aggregator().snapshot()["by_graph"]["chat"]["failure_rate"] == 0.5


async def test_tool_loop_calls_each_get_a_record() -> None:
    """Every model pass in a tool loop is its own measured request."""
    tool_call = AIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": "query_evidence", "args": {"days_back": 7}}],
        usage_metadata={"input_tokens": 200, "output_tokens": 10, "total_tokens": 210},
    )
    model = _ScriptedChatModel([tool_call, _answer("根据证据，建议休息。")])

    from langchain_core.tools import tool as lc_tool

    @lc_tool(name_or_callable="query_evidence")
    async def query_evidence(days_back: int = 7) -> str:
        """查询证据。"""
        _ = days_back
        return json.dumps({
            "evidence": [{"metric": "focus_score", "value": 0.31}],
            "behavior_summary": {"duration_min": 120.0},
        })

    detector = MagicMock(spec=CrisisDetector)
    detector.scan.return_value = (CrisisLevel.NONE, None)
    graph = ChatGraph(
        chat_repo=_InMemoryChatRepo(),
        crisis_detector=detector,
        model=model,
        tools=[query_evidence],
        tool_adapters=[],
        provider="generic",
    )

    result = await graph.ask(user_id=1, session_id="s1", message="查证据")

    assert result.tools_used == ("query_evidence",)
    records = default_aggregator().records()
    assert len(records) == 2
    assert all(record.node == "model_call" for record in records)
    assert all(record.role == "chat" for record in records)
    assert records[0].output_tokens == 10


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Privacy: prompt / key / body sentinels never reach the metadata
# ═══════════════════════════════════════════════════════════════════════════════

_SENTINELS = (PROMPT_SENTINEL, KEY_SENTINEL, BODY_SENTINEL)


def _assert_no_sentinels(text: str, *, where: str) -> None:
    for sentinel in _SENTINELS:
        assert sentinel not in text, f"{sentinel!r} leaked into {where}"


async def test_prompt_key_and_body_never_reach_records_or_snapshot() -> None:
    """Sentinels ride the real call path and are then absent from all metadata."""
    hostile_body = f"推理过程 {BODY_SENTINEL} key={KEY_SENTINEL}"
    model = _ScriptedChatModel([
        # Forbidden word in the first completion → the correction retry runs,
        # so both records are produced with hostile bodies in between.
        _answer(f"根据诊断结果 {hostile_body}"),
        _answer("干净的纠正回答"),
    ])
    graph = _chat_graph(model)
    assert KEY_SENTINEL not in CHAT_SYSTEM_PROMPT  # sanity: it is only in our prompt

    await graph.ask(
        user_id=1,
        session_id="s1",
        message=f"{PROMPT_SENTINEL} 请分析（api_key={KEY_SENTINEL}）",
    )

    # The sentinels really were on the wire (prompt and key in the request, the
    # provider body in the completion), so their absence below is meaningful.
    sent = " ".join(
        str(getattr(message, "content", ""))
        for messages in model.seen_messages
        for message in messages
    )
    assert PROMPT_SENTINEL in sent
    assert KEY_SENTINEL in sent
    assert BODY_SENTINEL in hostile_body

    _assert_no_sentinels(_records_json(), where="records")
    _assert_no_sentinels(_snapshot_json(), where="snapshot")
    records = default_aggregator().records()
    assert len(records) == 2
    for record in records:
        assert PROMPT_SENTINEL not in record.node
        assert KEY_SENTINEL not in record.model
        assert KEY_SENTINEL not in record.provider
        assert BODY_SENTINEL not in record.error_category
    assert ALLOWED_FIELDS.isdisjoint(
        {"prompt", "system", "user", "messages", "api_key", "content", "reasoning"}
    )


async def test_error_category_never_carries_the_failure_message() -> None:
    """A provider error text is reduced to a category, never stored verbatim."""
    model = _ScriptedChatModel([])
    model.ainvoke = AsyncMock(
        side_effect=RuntimeError(f"boom {PROMPT_SENTINEL} {KEY_SENTINEL}"),
    )
    graph = _chat_graph(model)

    await graph.ask(user_id=1, session_id="s1", message=f"问题 {PROMPT_SENTINEL}")

    record = default_aggregator().records()[0]
    assert record.error_category == "transport"
    _assert_no_sentinels(_records_json(), where="records")
    _assert_no_sentinels(_snapshot_json(), where="snapshot")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Smoke script: CLI contract and budget guard
# ═══════════════════════════════════════════════════════════════════════════════


def test_smoke_requires_yes_and_a_target(smoke: Any) -> None:
    """No ``--yes`` → refuse; no target → refuse. Both are exit code 2."""
    assert smoke.main(["--dry-run", "--panel"]) == 2
    assert smoke.main(["--dry-run", "--yes"]) == 2
    assert smoke.main(["--dry-run"]) == 2


def test_smoke_refuses_a_budget_below_the_selected_targets(smoke: Any) -> None:
    """The worst-case cost of the selection must fit inside the budget."""
    assert smoke.main(["--dry-run", "--yes", "--panel", "--max-requests", "3"]) == 2
    assert smoke.main(
        ["--dry-run", "--yes", "--panel", "--chat", "--tools", "--attribution",
         "--max-requests", "8"],
    ) == 2


def test_smoke_refuses_or_clamps_nothing_above_the_hard_ceiling(smoke: Any) -> None:
    """An oversized budget is refused, never silently clamped."""
    assert smoke.main(
        ["--dry-run", "--yes", "--chat", "--max-requests",
         str(smoke.HARD_MAX_REQUESTS + 1)],
    ) == 2
    assert smoke.main(["--dry-run", "--yes", "--chat", "--max-requests", "0"]) == 2


def test_smoke_defaults_are_small_and_isolated(smoke: Any) -> None:
    """Defaults: a small budget and no explicit data dir (a temp dir is used)."""
    args = smoke.build_parser().parse_args(["--yes", "--chat"])
    assert args.max_requests == smoke.DEFAULT_MAX_REQUESTS
    assert smoke.DEFAULT_MAX_REQUESTS <= 12
    assert args.data_dir == ""
    assert args.dry_run is False
    assert args.output_dir == ""


def test_run_budget_refuses_before_the_transport(smoke: Any) -> None:
    """The guard raises on the request *after* the budget, not on the response."""
    budget = smoke.RunBudget(max_requests=2)
    assert budget.reserve() == 1
    assert budget.reserve() == 2
    assert budget.remaining == 0
    with pytest.raises(smoke.BudgetRefusedError, match="budget exhausted"):
        budget.reserve()
    assert budget.attempted == 2, "a refused request is not counted as sent"


async def test_budgeted_gateway_and_model_charge_every_call(smoke: Any) -> None:
    """Both budget proxies charge exactly one unit per outbound call."""
    gateway_budget = smoke.RunBudget(max_requests=1)
    inner_gateway = AsyncMock()
    inner_gateway.complete = AsyncMock(return_value="{}")
    gateway = smoke.BudgetedGateway(inner_gateway, gateway_budget)

    assert await gateway.complete("s", "u") == "{}"
    assert gateway_budget.attempted == 1
    with pytest.raises(smoke.BudgetRefusedError):
        await gateway.complete("s", "u")

    model_budget = smoke.RunBudget(max_requests=1)
    inner_model = AsyncMock()
    inner_model.ainvoke = AsyncMock(return_value=AIMessage(content="hi"))
    model = smoke.BudgetedChatModel(inner_model, model_budget)

    await model.ainvoke([])
    assert model_budget.attempted == 1
    with pytest.raises(smoke.BudgetRefusedError):
        await model.ainvoke([])


def test_citation_extraction_and_alias_resolution(smoke: Any) -> None:
    """``[证据: id]`` is extracted, and bare names resolve to unique ids."""
    assert smoke.extract_citation_ids("结论 [证据: focus.switch_rate] 完") == [
        "focus.switch_rate",
    ]
    assert smoke.extract_citation_ids("无引用") == []
    assert smoke.extract_citation_ids("[证据: a] [证据: b]") == ["a", "b"]

    catalog = frozenset({"focus.switch_rate", "focus.focus_score"})
    report = smoke.citation_report(["focus.switch_rate", "switch_rate"], catalog)
    assert (report["total"], report["valid"], report["invalid"]) == (2, 2, [])
    assert report["validity"] == 1.0

    bogus = smoke.citation_report(["focus.does_not_exist"], catalog)
    assert bogus["validity"] == 0.0
    assert bogus["invalid"] == ["focus.does_not_exist"]
    assert smoke.citation_report([], catalog)["validity"] == 1.0


def test_accounting_aggregates_are_arithmetically_exact(smoke: Any) -> None:
    """p95, failure rate, token totals and the latency soundness gate."""
    from mindflow.services.llm_observability import LLMRequestRecord

    accounting = smoke.RequestAccounting()
    aggregator = default_aggregator()
    aggregator.record(LLMRequestRecord(
        graph="chat", node="model_call", role="chat", total_latency_ms=100.0,
        http_latency_ms=100.0, input_tokens=10, output_tokens=5, ok=True,
    ))
    aggregator.record(LLMRequestRecord(
        graph="chat", node="model_call", role="chat", total_latency_ms=300.0,
        http_latency_ms=300.0, input_tokens=20, output_tokens=5, ok=False,
    ))
    accounting.collect()

    assert accounting.request_count == 2
    assert accounting.failure_rate == 0.5
    assert accounting.token_totals["input_tokens"] == 30
    assert accounting.usage_reported is True
    assert accounting.p95_latency_ms >= 100.0
    sound, reason = accounting.latency_sound
    assert sound is True and reason == ""

    unsound = smoke.RequestAccounting(records=[
        dict(aggregator.records()[0].as_dict(), total_latency_ms=5.0, http_latency_ms=50.0),
    ])
    sound, reason = unsound.latency_sound
    assert sound is False and "http" in reason


def test_soundness_gate_rejects_missing_measurement_and_leaked_credential(smoke: Any) -> None:
    """A live target with no measured request cannot pass; nor can a leaked key."""
    empty = smoke.RequestAccounting()
    reasons = smoke._soundness_reasons(empty, "", dry_run=False)
    assert any("no measured request" in reason for reason in reasons)
    assert smoke._soundness_reasons(empty, "", dry_run=True) == []

    # A realistic allow-listed record shape, with a key-shaped label smuggled in.
    leaked = smoke.RequestAccounting(records=[{
        "total_latency_ms": 10.0, "http_latency_ms": 10.0, "ok": True,
        "parse_failure": False, "forbidden_word_failure": False, "error_category": "",
        "node": "model_call", "provider": KEY_SENTINEL, "model": "y",
        "http_status_class": "2xx", "input_tokens": 1, "output_tokens": 1,
        "reasoning_tokens": 0,
    }])
    reasons = smoke._soundness_reasons(leaked, KEY_SENTINEL, dry_run=False)
    assert any("credential" in reason for reason in reasons)
    assert smoke._soundness_reasons(leaked, KEY_SENTINEL, dry_run=True) == []


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Smoke script: degraded paths are reported, never silently accepted
# ═══════════════════════════════════════════════════════════════════════════════


class _ScriptedPanelGateway:
    """Offline panel gateway emitting valid, citation-carrying expert responses.

    Unlike the eval mock this one also records a measured request per call, so
    the panel target's request/latency gates are exercised for real.
    """

    def __init__(self, *, citation: str = "focus.switch_rate") -> None:
        self._citation = citation
        self.calls = 0

    async def complete(self, system: str, user: str, **kwargs: Any) -> str:
        _ = (system, kwargs)
        self.calls += 1
        from mindflow.services.llm_observability import (
            LLMRequestRecord,
            record_llm_request,
        )

        record_llm_request(LLMRequestRecord(
            graph="panel", node=f"node{self.calls}", role="panel",
            total_latency_ms=25.0, http_latency_ms=24.0,
            input_tokens=50, output_tokens=20, ok=True, http_status_class="2xx",
        ))
        if '"patterns"' in user or "patterns" in system:
            return json.dumps({
                "patterns": [
                    {"name": "冲动分心模式", "severity": "severe", "description": "高切换"},
                ],
                "anomalies": [],
                "top_concerns": ["冲动分心模式"],
                "evidence_citations": [self._citation],
            }, ensure_ascii=False)
        if "attribution_types" in system:
            bare = self._citation.rsplit(".", 1)[-1]
            return json.dumps({
                "attribution_types": ["impulsivity"],
                "confidence": {"impulsivity": 0.8},
                "argument": f"高切换与短专注块支持冲动分心模式 [证据: {bare}]",
                "evidence_citations": [self._citation],
            }, ensure_ascii=False)
        if "types" in system and "rationale" in system:
            return json.dumps({
                "types": ["impulsivity"],
                "confidence": {"impulsivity": 0.8},
                "recommended_technique": "stimulus_control",
                "rationale": f"综合专家意见 [证据: {self._citation}]",
                "dissent": [],
            }, ensure_ascii=False)
        return json.dumps({
            "approved": True, "issues": [], "critique_detail": "通过。",
        }, ensure_ascii=False)

    async def close(self) -> None:
        return None


async def test_panel_target_records_schema_citations_and_requests(smoke: Any) -> None:
    """The panel target resolves citations against the catalog and counts calls."""
    smoke._ACTIVE_PROVIDER[0] = "generic"
    bundle = smoke.build_synthetic_bundle()
    result = smoke.TargetResult(target="panel")
    gateway = smoke.BudgetedGateway(_ScriptedPanelGateway(), smoke.RunBudget(12))

    accounting = smoke.RequestAccounting()
    accounting.reset()
    await smoke.run_panel_target(gateway, bundle, result)
    accounting.collect()

    assert result.schema_attempts >= 1
    assert result.schema_passes == result.schema_attempts
    assert result.citations["total"] >= 1
    assert result.citations["invalid"] == []
    assert result.citations["validity"] == 1.0
    assert result.degraded is False
    assert accounting.request_count >= 1
    assert accounting.usage_reported is True

    verdict = smoke.evaluate_panel(result, accounting, "", dry_run=False)
    assert verdict.ok is True and verdict.verdict == "PASS"


async def test_panel_target_flags_a_hallucinated_citation(smoke: Any) -> None:
    """A citation outside the bundle catalog fails the panel citation gate.

    The graph itself drops opinions carrying hallucinated citations, so the
    bogus id is pushed through the script's own scan — the surface the script
    is responsible for checking.
    """
    from mindflow.domain.evidence_facts import build_evidence_catalog, evidence_catalog_ids

    smoke._ACTIVE_PROVIDER[0] = "generic"
    catalog = frozenset(evidence_catalog_ids(
        build_evidence_catalog(smoke.build_synthetic_bundle()),
    ))
    result = smoke.TargetResult(target="panel")
    result.citations = smoke.citation_report(
        smoke.extract_citation_ids("综合意见 [证据: focus.not_in_catalog]"), catalog,
    )

    assert result.citations["invalid"] == ["focus.not_in_catalog"]
    assert result.citations["validity"] == 0.0
    verdict = smoke.evaluate_panel(
        result, smoke.RequestAccounting(records=[{
            "total_latency_ms": 10.0, "http_latency_ms": 10.0, "ok": True,
            "parse_failure": False, "forbidden_word_failure": False, "error_category": "",
            "node": "moderator", "provider": "generic", "model": "m",
            "http_status_class": "2xx", "input_tokens": 1, "output_tokens": 1,
            "reasoning_tokens": 0,
        }]), "", dry_run=False,
    )
    assert verdict.ok is False
    assert any("citation_validity" in reason for reason in verdict.reasons)


async def test_attribution_target_reports_the_l1_tier(smoke: Any) -> None:
    """A successful L1 answer passes; a missing credential is reported, not passed."""
    smoke._ACTIVE_PROVIDER[0] = "generic"
    result = smoke.TargetResult(target="attribution")
    accounting = smoke.RequestAccounting()
    accounting.reset()
    await smoke.run_attribution_target(
        smoke.build_attribution_client(True, None, smoke.RunBudget(4)), result,
    )
    accounting.collect()

    assert result.schema_passes == 1
    assert result.degraded is False
    assert result.degradation_marker == "single_expert"
    assert "impulsivity" in result.note

    missing = smoke.TargetResult(target="attribution")
    await smoke.run_attribution_target(None, missing)
    assert missing.degraded is True
    assert missing.degradation_marker == "deepseek_not_configured"
    verdict = smoke.evaluate_attribution(missing, smoke.RequestAccounting(), "", dry_run=True)
    assert verdict.ok is False
    assert any("did not answer on its own" in reason for reason in verdict.reasons)


async def test_chat_and_tools_targets_run_offline(smoke: Any) -> None:
    """The dry-run chat and tools fakes drive real graphs with real usage data."""
    smoke._ACTIVE_PROVIDER[0] = "generic"
    repo = smoke.EphemeralChatRepo()

    chat_result = smoke.TargetResult(target="chat")
    accounting = smoke.RequestAccounting()
    accounting.reset()
    await smoke.run_chat_target(
        smoke.build_chat_model(True, None, smoke.RunBudget(4), with_tools=False),
        repo, chat_result,
    )
    accounting.collect()
    assert chat_result.note
    assert accounting.request_count == 1
    assert accounting.token_totals["input_tokens"] == 96
    assert accounting.usage_reported is True
    assert smoke.evaluate_chat_like(
        chat_result, accounting, "", require_tool=False, dry_run=False,
    ).ok is True

    adapter, tool = smoke.build_evidence_tool()
    tools_result = smoke.TargetResult(target="tools")
    tools_accounting = smoke.RequestAccounting()
    tools_accounting.reset()
    await smoke.run_tools_target(
        smoke.build_chat_model(True, None, smoke.RunBudget(4), with_tools=True),
        repo, tool, adapter, tools_result,
    )
    tools_accounting.collect()
    assert tools_result.tool_calls == 1
    assert tools_result.tool_call_rate == 1.0
    assert tools_accounting.request_count == 2
    assert tools_accounting.token_totals["input_tokens"] == 338
    assert smoke.evaluate_chat_like(
        tools_result, tools_accounting, "", require_tool=True, dry_run=False,
    ).ok is True


async def test_tools_target_fails_when_the_model_never_calls_a_tool(smoke: Any) -> None:
    """The whole point of ``--tools``: an answer without a call must FAIL."""
    smoke._ACTIVE_PROVIDER[0] = "generic"
    result = smoke.TargetResult(target="tools")
    adapter, tool = smoke.build_evidence_tool()
    # The chat scripted model answers immediately and never proposes a call.
    await smoke.run_tools_target(
        smoke.build_chat_model(True, None, smoke.RunBudget(4), with_tools=False),
        smoke.EphemeralChatRepo(), tool, adapter, result,
    )

    assert result.tool_calls == 0
    verdict = smoke.evaluate_chat_like(
        result, smoke.RequestAccounting(records=[{
            "total_latency_ms": 1.0, "http_latency_ms": 1.0, "ok": True,
            "parse_failure": False, "forbidden_word_failure": False, "error_category": "",
            "node": "model_call", "provider": "generic", "model": "m",
            "http_status_class": "2xx", "input_tokens": 1, "output_tokens": 1,
            "reasoning_tokens": 0,
        }]), "", require_tool=True, dry_run=False,
    )
    assert verdict.ok is False
    assert any("tool_call_rate" in reason for reason in verdict.reasons)


async def test_chat_gate_rejects_placeholder_answer(smoke: Any) -> None:
    result = smoke.TargetResult(target="chat", note="(empty answer)")
    verdict = smoke.evaluate_chat_like(
        result,
        smoke.RequestAccounting(),
        "",
        require_tool=False,
        dry_run=True,
    )

    assert verdict.ok is False
    assert "empty answer" in verdict.reasons


async def test_chat_gate_rejects_degraded_answer(smoke: Any) -> None:
    result = smoke.TargetResult(
        target="chat",
        note="safe fallback response",
        degraded=True,
        degradation_path=["rule_engine"],
    )
    verdict = smoke.evaluate_chat_like(
        result,
        smoke.RequestAccounting(),
        "",
        require_tool=False,
        dry_run=True,
    )

    assert verdict.ok is False
    assert any("degraded path" in reason for reason in verdict.reasons)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Smoke script: dry-run end to end + config guard + artifacts
# ═══════════════════════════════════════════════════════════════════════════════


async def test_run_targets_shares_one_budget_across_targets(
    smoke: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second target cannot quietly spend a fresh allowance.

    Chat + tools fit in a budget of 3 only because they share one counter
    (chat 1 + tools 2); a per-target budget would let the pair spend 4.
    """
    monkeypatch.setattr(smoke, "_ACTIVE_PROVIDER", ["generic"])
    args = smoke.build_parser().parse_args([
        "--dry-run", "--yes", "--chat", "--tools", "--max-requests", "3",
    ])
    credential = smoke.Credential("", "unconfigured")

    results, accounting, _elapsed = await smoke.run_targets(
        args, ["chat", "tools"], None, credential,
    )

    assert results["chat"].ok is True
    assert results["tools"].ok is True
    assert results["tools"].tool_call_rate == 1.0
    charged = accounting["chat"].request_count + accounting["tools"].request_count
    assert charged == 3, "one shared budget: 1 chat call + 2 tool-loop calls"


async def test_shared_budget_refuses_the_second_target_when_spent(
    smoke: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With only one unit left, the tools target cannot spend a second request.

    The refusal surfaces *inside* the graph (the model proxy raises), so the turn
    degrades instead of reaching the provider — which is the required outcome:
    no request over budget, and a FAIL for the target rather than a silent pass.
    """
    monkeypatch.setattr(smoke, "_ACTIVE_PROVIDER", ["generic"])
    args = smoke.build_parser().parse_args([
        "--dry-run", "--yes", "--chat", "--tools", "--max-requests", "2",
    ])

    results, accounting, _elapsed = await smoke.run_targets(
        args, ["chat", "tools"], None, smoke.Credential("", "unconfigured"),
    )

    assert results["chat"].ok is True
    assert results["tools"].ok is False
    assert results["tools"].tool_call_rate == 0.0
    # One tool-phase call was charged inside the budget; the second was refused
    # *before* the transport and is recorded as an event with zero latency.
    refused = accounting["tools"].refused
    assert len(refused) == 1
    assert refused[0]["fallback_reason"] == smoke._BUDGET_REFUSED_REASON
    assert refused[0]["total_latency_ms"] == 0.0
    assert accounting["tools"].request_count == 1, "the refused attempt sent nothing"
    assert any("tool_call_rate" in reason for reason in results["tools"].reasons)


async def test_budget_refusal_is_reported_when_it_escapes_the_target(
    smoke: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refusal raised by the target itself is reported as a budget failure."""

    class _RefusingGateway:
        """Panel gateway whose very first call exceeds the budget."""

        async def complete(self, system: str, user: str, **kwargs: Any) -> str:
            _ = (system, user, kwargs)
            raise smoke.BudgetRefusedError("request budget exhausted: 0/0 used")

        async def close(self) -> None:
            return None

    monkeypatch.setattr(smoke, "_ACTIVE_PROVIDER", ["generic"])
    args = smoke.build_parser().parse_args([
        "--dry-run", "--yes", "--panel", "--max-requests", str(smoke.MIN_REQUESTS),
    ])

    results, _accounting, _elapsed = await smoke.run_targets(
        args, ["panel"], None, smoke.Credential("", "unconfigured"),
        gateway_override=_RefusingGateway(),
    )

    assert results["panel"].ok is False
    assert any("budget" in reason for reason in results["panel"].reasons)


def test_dry_run_end_to_end_writes_artifacts(
    smoke: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--dry-run`` runs all four targets and emits JSON + markdown artifacts."""
    monkeypatch.setattr(smoke, "_ACTIVE_PROVIDER", [""])
    out = tmp_path / "out"
    data = tmp_path / "data"
    code = smoke.main([
        "--dry-run", "--yes", "--panel", "--chat", "--tools", "--attribution",
        "--data-dir", str(data), "--output-dir", str(out), "--run-id", "dryrun-test",
    ])

    assert code == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "dry_run"
    assert summary["run_id"] == "dryrun-test"
    assert set(summary["targets"]) == {"panel", "chat", "tools", "attribution"}
    assert all(
        row["verdict"] == "PASS" for row in summary["targets"].values()
    ), summary["targets"]
    assert summary["budget"]["attempted"] <= summary["budget"]["max_requests"]
    assert summary["thresholds"]["citation_validity_panel"] == 1.0
    # Every recorded call is the allow-listed scalar shape — nothing else.
    for row in summary["targets"].values():
        for call in row["calls"]:
            assert set(call) == ALLOWED_FIELDS
    assert summary["targets"]["tools"]["tool_calls"] == 1
    assert summary["targets"]["chat"]["usage_reported"] is True

    report = (out / "report.md").read_text(encoding="utf-8")
    assert "MindFlow live LLM smoke report" in report
    assert "## Thresholds" in report
    assert "tool_call_rate" in report


def test_dry_run_artifacts_never_contain_the_api_key(
    smoke: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither artifact may carry the credential, whatever its source."""
    monkeypatch.setattr(smoke, "_ACTIVE_PROVIDER", [""])
    out = tmp_path / "out"
    monkeypatch.setenv("DEEPSEEK_API_KEY", KEY_SENTINEL)
    code = smoke.main([
        "--dry-run", "--yes", "--chat", "--data-dir", str(tmp_path / "data"),
        "--output-dir", str(out), "--run-id", "no-key",
    ])

    assert code == 0
    summary_text = (out / "summary.json").read_text(encoding="utf-8")
    report_text = (out / "report.md").read_text(encoding="utf-8")
    _assert_no_sentinels(summary_text, where="summary.json")
    _assert_no_sentinels(report_text, where="report.md")
    assert "value never recorded" in report_text
    assert "value never printed" not in summary_text  # the key itself is not echoed


def test_live_mode_without_credential_fails_fast(
    smoke: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No key → exit 2 with a clear message; never a silent Ollama/rule fallback."""
    stub = SimpleNamespace(
        llm=SimpleNamespace(api_key=None, is_ecnu=False, model=None, base_url=None),
    )
    monkeypatch.setattr(
        smoke, "resolve_credential",
        lambda settings: smoke.Credential("", "unconfigured"),
    )
    monkeypatch.setattr(
        importlib.import_module("mindflow.config"), "get_settings", lambda: stub,
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    code = smoke.main([
        "--yes", "--attribution", "--data-dir", str(tmp_path / "data"),
        "--output-dir", str(tmp_path / "out"), "--run-id", "no-cred",
    ])

    assert code == 2
    assert not (tmp_path / "out").exists()


def test_resolve_credential_prefers_l1_target_then_env(
    smoke: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L1 target wins; the DeepSeek env spellings are the documented fallback."""
    pinned = SimpleNamespace(llm=SimpleNamespace(
        api_key="legacy-ecnu-key",
        l1_target=lambda: SimpleNamespace(
            api_key="from-l1-target", provider="generic",
            base_url="https://api.deepseek.com", model="deepseek-flash",
            provenance="deepseek-direct",
        ),
    ))
    resolved = smoke.resolve_credential(pinned)
    assert resolved.value == "from-l1-target"
    assert resolved.source == "settings.llm.l1_target().api_key"
    assert resolved.provenance == "deepseek-direct"
    assert resolved.configured is True

    unconfigured = SimpleNamespace(llm=SimpleNamespace(
        api_key=None, l1_target=lambda: SimpleNamespace(api_key=None),
    ))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-env")
    resolved = smoke.resolve_credential(unconfigured)
    assert resolved.value == "from-env"
    assert resolved.source == "env:DEEPSEEK_API_KEY"
    assert resolved.configured is True

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("MINDFLOW_LLM__DEEPSEEK_API_KEY", "from-prefixed-env")
    assert smoke.resolve_credential(unconfigured).source == (
        "env:MINDFLOW_LLM__DEEPSEEK_API_KEY"
    )

    monkeypatch.delenv("MINDFLOW_LLM__DEEPSEEK_API_KEY", raising=False)
    missing = smoke.resolve_credential(unconfigured)
    assert missing.configured is False
    assert missing.source == "unconfigured"


def test_a_legacy_ecnu_key_is_not_borrowed_for_the_deepseek_pin(smoke: Any) -> None:
    """The pin refuses an ECNU-shaped key, so the script must refuse to run.

    This is the live-branch guard: a lingering ECNU credential in ``.env``
    yields ``l1_target().api_key is None``, and reporting a "configured" L1 from
    the legacy field would let a degraded run look like a passing smoke test.
    """
    settings = SimpleNamespace(llm=SimpleNamespace(
        api_key="legacy-ecnu-key",
        l1_target=lambda: SimpleNamespace(
            api_key=None, provider="generic", base_url="https://api.deepseek.com",
            model="deepseek-flash", provenance="deepseek-direct",
        ),
    ))
    resolved = smoke.resolve_credential(settings)
    assert resolved.configured is False
    assert resolved.l1_available is True
    assert resolved.source == "unconfigured"

    # An older settings shape (no l1_target) still falls back to the raw key.
    legacy_shape = SimpleNamespace(llm=SimpleNamespace(api_key="from-settings"))
    resolved = smoke.resolve_credential(legacy_shape)
    assert resolved.source == "settings.llm.api_key"
    assert resolved.configured is True


def test_provider_snapshot_records_host_not_the_full_url(smoke: Any) -> None:
    """The report carries provider/model/host — never a URL that could hide a key."""
    settings = SimpleNamespace(llm=SimpleNamespace(
        api_key=KEY_SENTINEL,
        base_url="https://chat.ecnu.edu.cn/open/api/v1?key=" + KEY_SENTINEL,
        model="ecnu-max",
        provider="ecnu",
        is_ecnu=True,
        thinking_enabled=True,
        max_output_tokens=4096,
        max_concurrent_requests=1,
        timeout_s=180,
        max_retries=1,
        l1_target=lambda: SimpleNamespace(
            provider="generic", api_key="hidden", model="deepseek-flash",
            base_url="https://api.deepseek.com", provenance="deepseek-direct",
        ),
    ))

    snapshot = smoke.provider_snapshot(settings, None)
    # Provider *identity* (what chat/panel records carry), not the wire
    # protocol: the pinned DeepSeek target reports "deepseek" so per-provider
    # aggregation matches the panel gateway's records.
    assert snapshot["provider"] == "deepseek"
    assert snapshot["model"] == "deepseek-flash"
    assert snapshot["base_url_host"] == "api.deepseek.com"
    assert snapshot["l1_provenance"] == "deepseek-direct"
    assert KEY_SENTINEL not in json.dumps(snapshot, ensure_ascii=False)


def test_synthetic_inputs_are_schema_valid(smoke: Any) -> None:
    """The script's own fixtures build a valid bundle and a realistic summary."""
    bundle = smoke.build_synthetic_bundle()
    from mindflow.domain.evidence_facts import build_evidence_catalog, evidence_catalog_ids

    catalog = evidence_catalog_ids(build_evidence_catalog(bundle))
    assert "focus.switch_rate" in catalog
    assert len(catalog) >= 6

    summary_json = json.loads(smoke.build_synthetic_summary_json())
    assert summary_json["session"]["duration_min"] > 0
    assert summary_json["metrics"]["social_media_ratio"] > 0.5
    assert summary_json["metrics"]["start_delay_min"] > 10
    assert "社交媒体占比高" in summary_json["pattern_summary"]
