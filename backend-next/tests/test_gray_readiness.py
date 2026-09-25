"""Gray-release readiness regressions.

Covers the three changes the small-scale rollout depends on:

1. **Fallback usage capture** — the single-expert L1 path (raw
   ``DeepSeekClient``) must carry the provider-reported token triple into its
   observability record; "usage not reported" is only acceptable when the
   provider actually omitted it.
2. **Panel fan-out gateway** — ``panel_fanout_concurrency > 1`` builds a
   dedicated gateway with its own burst gate for the panel's parallel batches
   while the global gate stays at ``max_concurrent_requests``; the default (1)
   keeps exactly one gateway/pool as before.
3. **Smoke aggregation helpers** — ``TargetResult`` must merge notes,
   citations and degradation facts across repeated runs so a multi-scenario
   smoke reports aggregate truth rather than the last run's view.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from mindflow.config import LLMSettings
from mindflow.graph.fallback_nodes import (
    FallbackRunContext,
    FallbackState,
    _record_fallback_attempt,
    single_expert_node,
)
from mindflow.infrastructure.llm.client import DeepSeekClient
from mindflow.infrastructure.llm.schemas import LLMAttributionResult
from mindflow.services.llm_observability import (
    default_aggregator,
    reset_llm_observability,
)

# tests/__init__.py puts tests/ on sys.path.
from tests._llm_test_support import ECNU_BASE_URL  # noqa: I001


def _records() -> list[dict[str, object]]:
    return [record.as_dict() for record in default_aggregator().records()]


def _valid_attribution() -> LLMAttributionResult:
    return LLMAttributionResult(
        procrastination_types=["impulsivity"],
        type_confidence={"impulsivity": 0.82},
        cognitive_distortions=["all-or-nothing thinking"],
        cbt_technique="stimulus_control",
        response_text="测试回应",
        next_action="测试行动",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Fallback usage capture
# ═══════════════════════════════════════════════════════════════════════════════


def test_record_fallback_attempt_forwards_provider_usage() -> None:
    reset_llm_observability()
    _record_fallback_attempt(
        provider="deepseek",
        node="single_expert",
        role="single_expert",
        model="deepseek-flash",
        attempts=[],
        ok=True,
        final_source="deepseek",
        usage=(111, 222, 33),
    )

    record = _records()[-1]
    assert record["input_tokens"] == 111
    assert record["output_tokens"] == 222
    assert record["reasoning_tokens"] == 33


def test_record_fallback_attempt_without_usage_stays_zero() -> None:
    reset_llm_observability()
    _record_fallback_attempt(
        provider="deepseek",
        node="single_expert",
        role="single_expert",
        model="deepseek-flash",
        attempts=[],
        ok=False,
        fallback_reason="deepseek_transport",
    )

    record = _records()[-1]
    assert record["input_tokens"] == 0
    assert record["output_tokens"] == 0
    assert record["reasoning_tokens"] == 0


async def test_single_expert_node_records_the_client_usage() -> None:
    """The node reads ``client.last_usage`` after a successful call."""
    reset_llm_observability()
    client = MagicMock(spec=DeepSeekClient)
    client.analyze = AsyncMock(return_value=_valid_attribution())
    client.last_usage = (12, 34, 5)

    state: FallbackState = {
        "user_id": 1,
        "summary_json": '{"session": {}}',
        "degradation_path": [],
        "runtime": FallbackRunContext(deepseek_client=client),
    }
    update = await single_expert_node(state)

    assert update.get("error") is None
    record = _records()[-1]
    assert record["node"] == "single_expert"
    assert record["input_tokens"] == 12
    assert record["output_tokens"] == 34
    assert record["reasoning_tokens"] == 5


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Panel fan-out gateway
# ═══════════════════════════════════════════════════════════════════════════════


def _registry(**overrides: object) -> object:  # noqa: ANN202 - test helper
    from mindflow.infrastructure.provider_registry import ProviderRegistry

    defaults: dict[str, object] = {
        "api_key": "sk-legacy",
        "base_url": ECNU_BASE_URL,
        "model": "ecnu-max",
        "provider": "ecnu",
        "max_retries": 0,
    }
    defaults.update(overrides)
    return ProviderRegistry(LLMSettings(**defaults))  # type: ignore[arg-type]


async def test_default_burst_is_three_with_dedicated_gateway() -> None:
    """The measured default: burst 3 for the panel fan-out, global gate at 1."""
    registry = _registry(deepseek_api_key="sk-sentinel")
    try:
        assert registry.fanout_concurrency == 3
        fanout = registry.get_fanout_gateway()
        assert fanout is not registry.get_gateway()
        assert fanout._concurrency.limit == 3  # noqa: SLF001 - registry contract
        assert registry.get_gateway()._concurrency.limit == 1  # noqa: SLF001
    finally:
        await registry.shutdown()


async def test_burst_one_collapses_to_the_shared_gateway() -> None:
    """``panel_fanout_concurrency=1`` keeps exactly one gateway/pool."""
    registry = _registry(deepseek_api_key="sk-sentinel", panel_fanout_concurrency=1)
    try:
        assert registry.fanout_concurrency == 1
        assert registry.get_fanout_gateway() is registry.get_gateway()
    finally:
        await registry.shutdown()


async def test_burst_builds_a_dedicated_gateway_with_its_own_gate() -> None:
    registry = _registry(deepseek_api_key="sk-sentinel", panel_fanout_concurrency=3)
    try:
        fanout = registry.get_fanout_gateway()
        assert fanout is not registry.get_gateway()
        # Burst gate vs the untouched global gate.
        assert fanout._concurrency.limit == 3  # noqa: SLF001 - registry contract
        assert registry.get_gateway()._concurrency.limit == 1  # noqa: SLF001
        # Same resolved endpoint and model — only the gate differs.
        assert fanout._model_id == registry.get_gateway()._model_id  # noqa: SLF001
        assert fanout._base_url == registry.get_gateway()._base_url  # noqa: SLF001
        # Shutdown closes the dedicated pool too.
        await registry.shutdown()
        assert registry._fanout_gateway is None  # noqa: SLF001
    finally:
        await registry.shutdown()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Smoke aggregation helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _result() -> object:  # noqa: ANN202 - test helper
    from scripts.smoke_live_llm import TargetResult

    return TargetResult(target="panel")


def test_target_result_merges_notes_and_degradation() -> None:
    result = _result()
    result.add_note("panel verdict: impulsivity escalated(rebuttal round)")
    result.merge_degradation(False, [], "")
    result.add_note("panel verdict: perfectionism")
    result.merge_degradation(True, ["rule_engine"], "rule_engine")

    assert result.runs == 0  # runs is counted by the runners, not the helpers
    assert "escalated" in result.note and "perfectionism" in result.note
    assert result.degraded is True
    assert result.degradation_path == ["rule_engine"]
    assert result.degradation_marker == "rule_engine"


def test_target_result_merge_citations_recomputes_validity() -> None:
    result = _result()
    # citation_report stores ``valid`` as a count and ``invalid`` as id lists.
    result.merge_citations({
        "total": 2, "valid": 2, "invalid": [], "validity": 1.0, "catalog_size": 5,
    })
    result.merge_citations({
        "total": 1, "valid": 0, "invalid": ["x"], "validity": 0.0, "catalog_size": 5,
    })

    assert result.citations["total"] == 3
    assert result.citations["valid"] == 2
    assert result.citations["invalid"] == ["x"]
    assert result.citations["validity"] == pytest.approx(2 / 3)


def test_panel_bundles_multi_scenario_is_diverse() -> None:
    from scripts.smoke_live_llm import _panel_bundles

    single = _panel_bundles(1)
    assert len(single) == 1
    multi = _panel_bundles(3)
    assert len(multi) == 3
    # Distinct evidence shapes, not the same bundle three times.
    assert len({id(b) for b in multi}) == 3
