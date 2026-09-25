"""Regression tests for role-level ``CompletionPolicy`` (optimisation plan 2.1).

Four things are asserted, in the order the plan states them:

1. **The table** — all six roles carry exactly the reasoning effort and output
   ceiling the plan specifies, and the lookup resolves the expert role strings
   the panel actually uses.
2. **Per-request, not per-model** — a policy reaches the wire as request-body
   fields on the *shared* model instance, with chat deliberately keeping the
   provider's configured reasoning effort. The spelling is the provider's:
   ECNU gets ``reasoning_effort`` + ``max_completion_tokens``, DeepSeek (the
   production L1) gets a top-level ``reasoning_effort`` plus
   ``extra_body={"max_tokens": …, "thinking": {"type": "enabled"}}``.
3. **One client, one model per tier** — running all six roles constructs no
   second HTTP client and no second model instance. This is the property that
   motivated the change: per-role clients would each own a connection pool.
   ``"chat"``/``"reasoner"`` are tier labels, not model ids: both tiers request
   the resolved model and differ only in the JSON output constraint.
4. **Backward compatibility** — a gateway whose ``complete()`` predates the
   parameter still works; ``gateway_accepts_policy`` reports False and the panel
   helper passes no policy at all.

Offline: the transport is mocked, so these tests never reach a provider.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek

from mindflow.agents.experts import ANALYST, ATTRIBUTION_EXPERTS, CRITIC, MODERATOR
from mindflow.agents.llm_gateway import (
    LangChainGateway,
    gateway_accepts_policy,
)
from mindflow.agents.policies import (
    ANALYST_POLICY,
    ATTRIBUTION_POLICY,
    CBT_POLICY,
    CHAT_POLICY,
    CRITIC_POLICY,
    EMOTION_POLICY,
    MODERATOR_POLICY,
    POLICIES,
    REBUTTAL_POLICY,
    ROLE_ANALYST,
    ROLE_ATTRIBUTION,
    ROLE_CHAT,
    ROLE_CRITIC,
    ROLE_MODERATOR,
    ROLE_REBUTTAL,
    THINKING_TOKEN_FLOOR,
    TMT_POLICY,
    CompletionPolicy,
    policy_for_expert,
    policy_for_role,
)
from mindflow.graph.panel_graph import _call_with_budget
from mindflow.infrastructure.llm.ecnu import build_ecnu_model
from tests._llm_test_support import (
    ECNU_BASE_URL,
    GENERIC_BASE_URL,
    MockLLMWire,
    chat_completion,
    make_generic_settings,
    make_settings,
)

#: Every role the panel can run as, with the policy the plan assigns it.
ROLE_TABLE: tuple[tuple[str, str | None, int], ...] = (
    (ROLE_ANALYST, "low", 1200),
    (ROLE_ATTRIBUTION, "low", 1000),
    (ROLE_REBUTTAL, "low", 800),
    (ROLE_MODERATOR, "high", 1600),
    (ROLE_CRITIC, "low", 500),
    (ROLE_CHAT, None, 2048),
)


def _ok_wire() -> MockLLMWire:
    async def handler(request: httpx.Request) -> httpx.Response:
        return chat_completion('{"ok": true}')

    return MockLLMWire(handler)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. The role → policy table
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(("role", "effort", "max_tokens"), ROLE_TABLE)
def test_policy_table_matches_plan(role: str, effort: str | None, max_tokens: int) -> None:
    """Each of the plan's six rows carries its documented numbers."""
    policy = POLICIES[role]
    assert policy.role == role
    assert policy.reasoning_effort == effort
    assert policy.max_output_tokens == max_tokens
    assert policy.node, "every policy needs an observability node label"
    assert policy.intent, "the plan's intent column must not be dropped"


def test_policy_table_has_exactly_the_planned_roles() -> None:
    assert set(POLICIES) == {role for role, _, _ in ROLE_TABLE}


def test_attribution_experts_share_one_row_but_stay_distinguishable() -> None:
    """CBT/TMT/emotion share the plan's numbers yet aggregate separately."""
    for policy in (CBT_POLICY, TMT_POLICY, EMOTION_POLICY):
        assert policy.reasoning_effort == ATTRIBUTION_POLICY.reasoning_effort == "low"
        assert policy.max_output_tokens == ATTRIBUTION_POLICY.max_output_tokens == 1000
    # Same numbers, different role keys: per-role aggregates stay meaningful.
    assert len({CBT_POLICY.role, TMT_POLICY.role, EMOTION_POLICY.role}) == 3
    assert {CBT_POLICY.role, TMT_POLICY.role, EMOTION_POLICY.role} == {
        "cbt", "tmt", "emotion",
    }


def test_chat_is_independent_of_the_panel() -> None:
    """Chat keeps the provider default effort rather than inheriting a tier."""
    assert CHAT_POLICY.reasoning_effort is None
    assert CHAT_POLICY.max_output_tokens == 2048
    assert CHAT_POLICY.max_output_tokens not in {
        ANALYST_POLICY.max_output_tokens,
        ATTRIBUTION_POLICY.max_output_tokens,
        REBUTTAL_POLICY.max_output_tokens,
        MODERATOR_POLICY.max_output_tokens,
        CRITIC_POLICY.max_output_tokens,
    }


def test_policy_lookup_by_canonical_role_and_alias() -> None:
    assert policy_for_role("analyst") is ANALYST_POLICY
    assert policy_for_role("attribution") is ATTRIBUTION_POLICY
    assert policy_for_role("rebuttal") is REBUTTAL_POLICY
    assert policy_for_role("moderator") is MODERATOR_POLICY
    assert policy_for_role("critic") is CRITIC_POLICY
    assert policy_for_role("chat") is CHAT_POLICY
    assert policy_for_role("emotion_regulation") is EMOTION_POLICY


def test_policy_lookup_by_expert_role_string_and_instance() -> None:
    """The panel looks policies up by ``ExpertDef``, not by a canonical key."""
    assert policy_for_role(ANALYST) is ANALYST_POLICY
    assert policy_for_expert(ANALYST) is ANALYST_POLICY
    assert policy_for_role(MODERATOR) is MODERATOR_POLICY
    assert policy_for_expert(CRITIC) is CRITIC_POLICY
    for expert, expected in zip(
        ATTRIBUTION_EXPERTS,
        (CBT_POLICY, TMT_POLICY, EMOTION_POLICY),
        strict=True,
    ):
        assert policy_for_expert(expert) is expected
        assert policy_for_role(expert) is expected
    # The Chinese role strings resolve too (that is what the graph passes).
    assert policy_for_role(ANALYST.role) is ANALYST_POLICY
    assert policy_for_role(MODERATOR.role) is MODERATOR_POLICY


def test_reconstructed_expert_resolves_by_role_string() -> None:
    """An equal-but-not-identical ExpertDef still maps to its role policy."""
    clone = type(ANALYST)(
        role=ANALYST.role,
        perspective=ANALYST.perspective,
        system_prompt=ANALYST.system_prompt,
    )
    assert clone is not ANALYST
    assert policy_for_expert(clone) is ANALYST_POLICY


@pytest.mark.parametrize("unknown", [None, "", "   ", "unknown_role", "主持人"])
def test_unknown_role_means_no_overrides(unknown: object) -> None:
    """An unrecognised role must not silently inherit some other role's budget."""
    assert policy_for_role(unknown) is None  # type: ignore[arg-type]


def test_every_panel_expert_has_a_policy() -> None:
    """No expert may be left unshaped — a missing row means provider defaults."""
    for expert in (*ATTRIBUTION_EXPERTS, ANALYST, CRITIC, MODERATOR):
        assert policy_for_role(expert) is not None, expert.role


def test_policy_serialises_without_prompt_text() -> None:
    """``as_dict`` is the evidence-trail view: scalars only, no prompt body."""
    payload = ANALYST_POLICY.as_dict()
    assert set(payload) == {
        "role", "reasoning_effort", "max_output_tokens", "temperature", "intent", "node",
    }
    assert all(isinstance(value, (str, int, float, type(None))) for value in payload.values())
    """``as_dict`` is the evidence-trail view: scalars only, no prompt body."""
    payload = ANALYST_POLICY.as_dict()
    assert set(payload) == {
        "role", "reasoning_effort", "max_output_tokens", "temperature", "intent", "node",
    }
    assert all(isinstance(value, (str, int, float, type(None))) for value in payload.values())


def test_request_overrides_omit_unset_fields() -> None:
    """A field the policy leaves ``None`` must not appear as an override."""
    chat = CHAT_POLICY.request_overrides(provider="ecnu")
    assert "reasoning_effort" not in chat  # provider default, not "send nothing"
    assert chat["max_completion_tokens"] == 2048

    bare = CompletionPolicy(role="x", reasoning_effort=None, max_output_tokens=None)
    assert bare.request_overrides(provider="ecnu") == {}


def test_request_overrides_spell_the_cap_per_provider() -> None:
    """ECNU gets ``max_completion_tokens``; DeepSeek/generic get ``extra_body``.

    ``langchain-openai`` rewrites a top-level ``max_tokens`` into
    ``max_completion_tokens``, which DeepSeek does not document; ``extra_body``
    reaches the wire unchanged.
    """
    ecnu = ANALYST_POLICY.request_overrides(provider="ecnu")
    assert ecnu["max_completion_tokens"] == 1200
    assert ecnu["reasoning_effort"] == "low"
    assert "max_tokens" not in ecnu
    assert "extra_body" not in ecnu

    # The production L1 (DeepSeek): documented cap in ``extra_body``, the
    # thinking switch next to it, the effort top-level where DeepSeek reads it.
    # With thinking on, thinking tokens are billed against the cap, so the
    # request carries the thinking floor instead of the answer-only cap
    # (real-request observation 2026-09-25: a 1200-token thinking request
    # truncated with ``LengthFinishReasonError`` before emitting any JSON).
    deepseek = ANALYST_POLICY.request_overrides(provider="deepseek")
    assert deepseek["reasoning_effort"] == "low"
    assert deepseek["extra_body"] == {
        "max_tokens": max(ANALYST_POLICY.max_output_tokens, THINKING_TOKEN_FLOOR),
        "thinking": {"type": "enabled"},
    }
    assert deepseek["extra_body"]["max_tokens"] == THINKING_TOKEN_FLOOR
    assert "max_completion_tokens" not in deepseek

    # Without thinking the policy's own cap is exactly what goes out.
    generic = ANALYST_POLICY.request_overrides(provider="generic")
    assert generic["extra_body"] == {"max_tokens": 1200}
    assert "max_completion_tokens" not in generic
    # The thinking tier is a campus-gateway concept; it never travels generically.
    assert "reasoning_effort" not in generic
    assert "thinking" not in generic["extra_body"]


def test_thinking_requests_respect_the_floor_and_non_thinking_ones_do_not() -> None:
    """The floor applies only where thinking is actually requested."""
    thinking = MODERATOR_POLICY.request_overrides(provider="deepseek")
    assert MODERATOR_POLICY.reasoning_effort == "high"
    assert thinking["extra_body"]["max_tokens"] == max(
        MODERATOR_POLICY.max_output_tokens, THINKING_TOKEN_FLOOR,
    )

    # A policy whose cap already exceeds the floor is not lowered to it.
    generous = replace(ANALYST_POLICY, max_output_tokens=THINKING_TOKEN_FLOOR + 500)
    assert generous.request_overrides(provider="deepseek")["extra_body"]["max_tokens"] == (
        THINKING_TOKEN_FLOOR + 500
    )


def test_chat_policy_sends_only_its_cap_on_the_deepseek_path() -> None:
    """Chat rides the provider default effort — no effort, no thinking field.

    The plan's chat row asks for 2048 output tokens and the provider's own
    thinking default (DeepSeek documents thinking on / effort ``high``), so the
    DeepSeek body must carry the cap and *nothing* that would override it.
    """
    overrides = CHAT_POLICY.request_overrides(provider="deepseek")
    assert overrides == {"extra_body": {"max_tokens": 2048}}
    assert "reasoning_effort" not in overrides
    assert "thinking" not in overrides["extra_body"]


def test_temperature_is_sent_only_when_policy_sets_it() -> None:
    """Thinking mode constrains sampling, so the panel leaves it unset by default."""
    assert "temperature" not in ANALYST_POLICY.request_overrides(provider="ecnu")
    warm = CompletionPolicy(role="x", reasoning_effort=None, max_output_tokens=64, temperature=0.7)
    assert warm.request_overrides(provider="ecnu")["temperature"] == 0.7


def test_completion_policy_is_frozen() -> None:
    """The table is shared state; a mutating caller must not be able to edit it."""
    with pytest.raises(AttributeError):
        ANALYST_POLICY.max_output_tokens = 9999  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Policy reaches the wire as a per-request override
# ═══════════════════════════════════════════════════════════════════════════════


async def test_complete_passes_policy_fields_on_the_request_body() -> None:
    """The mocked transport sees the policy's numbers, not the model's defaults."""
    wire = _ok_wire()
    settings = make_settings()
    gateway = LangChainGateway(
        api_key="test-key", base_url=ECNU_BASE_URL, llm_settings=settings,
    )
    with wire.patch_async_client():
        await gateway.complete(
            "system", "user", model="chat", policy=ANALYST_POLICY,
        )
        await gateway.complete(
            "system", "user", model="reasoner", policy=MODERATOR_POLICY,
        )
        await gateway.close()

    assert wire.recorder.total_requests == 2
    analyst_body, moderator_body = wire.recorder.payloads
    assert analyst_body["reasoning_effort"] == "low"
    assert analyst_body["max_completion_tokens"] == 1200
    assert moderator_body["reasoning_effort"] == "high"
    assert moderator_body["max_completion_tokens"] == 1600
    # The instance default (settings.reasoning_effort = "max") never leaks in.
    assert "max" not in {analyst_body["reasoning_effort"], moderator_body["reasoning_effort"]}


async def test_no_policy_preserves_provider_defaults() -> None:
    """``policy=None`` sends nothing extra — the pre-2.1 behaviour, byte for byte."""
    wire = _ok_wire()
    settings = make_settings(reasoning_effort="max")
    gateway = LangChainGateway(
        api_key="test-key", base_url=ECNU_BASE_URL, llm_settings=settings,
    )
    with wire.patch_async_client():
        await gateway.complete("system", "user", model="chat")
        await gateway.close()

    body = wire.recorder.last
    assert body["reasoning_effort"] == "max"  # the model's own configured effort
    assert body["max_completion_tokens"] == settings.max_output_tokens


async def test_chat_policy_keeps_the_provider_default_effort() -> None:
    """Chat overrides only the output ceiling; effort stays the provider's."""
    wire = _ok_wire()
    settings = make_settings(reasoning_effort="high")
    gateway = LangChainGateway(
        api_key="test-key", base_url=ECNU_BASE_URL, llm_settings=settings,
    )
    with wire.patch_async_client():
        await gateway.complete("system", "user", model="chat", policy=CHAT_POLICY)
        # Same gateway, panel role next: the chat call must not have frozen
        # anything onto the shared model instance.
        await gateway.complete("system", "user", model="chat", policy=CRITIC_POLICY)
        await gateway.close()

    chat_body, critic_body = wire.recorder.payloads
    assert chat_body["reasoning_effort"] == "high"  # provider default, unchanged
    assert chat_body["max_completion_tokens"] == 2048
    assert critic_body["reasoning_effort"] == "low"
    assert critic_body["max_completion_tokens"] == 500


async def test_every_panel_role_sends_its_own_numbers_on_one_gateway() -> None:
    """All six roles through one gateway: each body carries its own policy."""
    wire = _ok_wire()
    gateway = LangChainGateway(
        api_key="test-key", base_url=ECNU_BASE_URL, llm_settings=make_settings(),
    )
    calls = [
        (ROLE_CHAT, "chat", CHAT_POLICY),
        *((role, "chat", POLICIES[role]) for role, _, _ in ROLE_TABLE if role != ROLE_CHAT),
        (ROLE_MODERATOR, "reasoner", MODERATOR_POLICY),
    ]
    with wire.patch_async_client():
        for _, model, policy in calls:
            await gateway.complete("s", "u", model=model, policy=policy)  # type: ignore[arg-type]
        await gateway.close()

    assert wire.recorder.total_requests == len(calls)
    for (role, _, policy), body in zip(calls, wire.recorder.payloads, strict=True):
        assert body["max_completion_tokens"] == policy.max_output_tokens, role
        if policy.reasoning_effort is None:
            assert body["reasoning_effort"] == "max", role  # provider default
        else:
            assert body["reasoning_effort"] == policy.reasoning_effort, role


async def test_temperature_override_reaches_the_wire_on_the_generic_path() -> None:
    """The OpenAI-compatible path does send an explicit sampling temperature."""
    wire = _ok_wire()
    settings = make_generic_settings()
    gateway = LangChainGateway(
        api_key="test-key", base_url=GENERIC_BASE_URL, llm_settings=settings,
    )
    warm = CompletionPolicy(
        role=ROLE_CHAT, reasoning_effort=None, max_output_tokens=512, temperature=0.35,
    )
    with wire.patch_async_client():
        await gateway.complete("s", "u", model="chat", policy=warm)
        await gateway.complete("s", "u", model="chat", policy=CHAT_POLICY)
        await gateway.close()

    assert wire.recorder.payloads[0]["temperature"] == 0.35
    assert wire.recorder.payloads[0]["max_tokens"] == 512  # via extra_body
    # A policy that does not set temperature keeps the model's own value; it is
    # never removed, because the generic provider has no thinking-mode conflict.
    assert wire.recorder.payloads[1]["temperature"] is not None


def test_ecnu_thinking_mode_drops_the_temperature_override() -> None:
    """The campus gateway's thinking mode forbids sampling params.

    This is the documented trade-off, and the point of the regression test is
    that it is *deliberate*: the adapter drops ``temperature`` whenever thinking
    is on, so a panel policy cannot accidentally send an unsupported parameter
    to ECNU. With thinking off the override passes through untouched.
    """
    overrides = CompletionPolicy(
        role=ROLE_CHAT, reasoning_effort=None, max_output_tokens=512, temperature=0.35,
    ).request_overrides(provider="ecnu")
    assert overrides["temperature"] == 0.35  # the policy still expresses it

    thinking_on = build_ecnu_model(
        model="ecnu-max", api_key="k", base_url=ECNU_BASE_URL,
        reasoning_effort="max", thinking_enabled=True,
    )._get_request_payload([HumanMessage(content="x")], **overrides)
    assert "temperature" not in thinking_on
    assert thinking_on["max_completion_tokens"] == 512
    assert thinking_on["extra_body"]["thinking"] == {"type": "enabled"}

    thinking_off = build_ecnu_model(
        model="ecnu-max", api_key="k", base_url=ECNU_BASE_URL,
        reasoning_effort="max", thinking_enabled=False,
    )._get_request_payload([HumanMessage(content="x")], **overrides)
    assert thinking_off["temperature"] == 0.35


async def test_model_construction_does_not_carry_the_effort_override() -> None:
    """The override rides the request; the model keeps its configured default.

    If a policy were applied at construction time, the shared instance would
    silently acquire whichever role happened to be constructed first.
    """
    seen: list[dict[str, Any]] = []
    real_build = build_ecnu_model

    def recording_build(**kwargs: Any) -> Any:
        seen.append(dict(kwargs))
        return real_build(**kwargs)

    wire = _ok_wire()
    settings = make_settings(reasoning_effort="max")
    gateway = LangChainGateway(
        api_key="test-key", base_url=ECNU_BASE_URL, llm_settings=settings,
    )
    import mindflow.agents.llm_gateway as gateway_module

    original = gateway_module.build_ecnu_model
    gateway_module.build_ecnu_model = recording_build
    try:
        with wire.patch_async_client():
            await gateway.complete("s", "u", model="chat", policy=CRITIC_POLICY)
            await gateway.complete("s", "u", model="chat", policy=MODERATOR_POLICY)
            await gateway.close()
    finally:
        gateway_module.build_ecnu_model = original

    assert len(seen) == 1, "the chat tier must be built once for both roles"
    assert seen[0]["reasoning_effort"] == "max"  # the settings value, not "low"/"high"
    assert seen[0]["max_tokens"] == settings.max_output_tokens
    assert [body["reasoning_effort"] for body in wire.recorder.payloads] == ["low", "high"]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. No new HTTP client and no new model instance per role
# ═══════════════════════════════════════════════════════════════════════════════


async def test_all_six_roles_construct_exactly_one_http_client_per_tier() -> None:
    """Six roles, two tiers ⇒ two clients total, and no per-role construction."""
    wire = _ok_wire()
    gateway = LangChainGateway(
        api_key="test-key", base_url=ECNU_BASE_URL, llm_settings=make_settings(),
    )
    with wire.patch_async_client():
        chat_model = gateway._get_model("chat")
        for policy in (
            ANALYST_POLICY, CBT_POLICY, TMT_POLICY,
            EMOTION_POLICY, MODERATOR_POLICY, CRITIC_POLICY, CHAT_POLICY,
        ):
            await gateway.complete("s", "u", model="chat", policy=policy)
            assert gateway._get_model("chat") is chat_model
        reasoner_model = gateway._get_model("reasoner")
        await gateway.complete("s", "u", model="reasoner", policy=MODERATOR_POLICY)
        assert gateway._get_model("reasoner") is reasoner_model
        await gateway.close()

    assert wire.recorder.total_requests == 8
    # One client for the chat tier, one for the reasoner tier — never one per role.
    assert wire.constructions == 2
    assert chat_model is not reasoner_model
    # Eight roles' worth of calls reused two client objects; no pool per role.
    assert len(wire.clients) == 2


def test_chatdeepseek_is_constructed_once_per_tier_on_the_generic_path() -> None:
    """The DeepSeek path builds one model per tier, not one per role.

    The tier label no longer selects a provider model id — both tiers request
    the *resolved* model (``deepseek-flash`` under the production pin) and
    differ only in whether JSON output mode is requested.
    """
    constructions: list[dict[str, Any]] = []
    real_init = ChatDeepSeek.__init__

    def recording_init(self: ChatDeepSeek, **kwargs: Any) -> None:
        constructions.append(dict(kwargs))
        real_init(self, **kwargs)

    settings = make_generic_settings()
    gateway = LangChainGateway(
        api_key="test-key", base_url=GENERIC_BASE_URL, llm_settings=settings,
    )
    ChatDeepSeek.__init__ = recording_init  # type: ignore[method-assign]
    try:
        for _ in range(3):
            gateway._get_model("chat")
        for _ in range(2):
            gateway._get_model("reasoner")
    finally:
        ChatDeepSeek.__init__ = real_init  # type: ignore[method-assign]

    assert len(constructions) == 2, "one model per tier, cached across roles"
    # One model id for both tiers: the resolved one (``deepseek-flash`` in
    # production, the configured provider model on this generic endpoint).
    assert {kwargs["model"] for kwargs in constructions} == {gateway._model_id}
    assert gateway._model_id == settings.model
    # The JSON-mode restriction is a *tier* property, still enforced per tier.
    assert constructions[0]["model_kwargs"] == {"response_format": {"type": "json_object"}}
    assert "model_kwargs" not in constructions[1]
    assert all("http_async_client" in kwargs for kwargs in constructions)


async def test_role_policies_share_the_same_client_object() -> None:
    """The client identity is stable across roles — pooling is actually reused."""
    wire = _ok_wire()
    gateway = LangChainGateway(
        api_key="test-key", base_url=ECNU_BASE_URL, llm_settings=make_settings(),
    )
    with wire.patch_async_client():
        for policy in (ANALYST_POLICY, CBT_POLICY, CRITIC_POLICY):
            await gateway.complete("s", "u", model="chat", policy=policy)
        await gateway.close()

    assert len(wire.clients) == 1
    client = wire.clients[0]
    assert client.is_closed, "gateway.close() must shut the one pool it built"


async def test_legacy_gateway_still_works_without_policy() -> None:
    """A three-argument gateway is invoked with no ``policy`` at all."""
    legacy = _LegacyGateway()
    runtime = _PanelRuntime()
    expert = ANALYST

    assert gateway_accepts_policy(legacy) is False
    result = await _call_with_budget(runtime, legacy, expert, "user message")

    assert result == '{"legacy": true}'
    assert legacy.calls and legacy.calls[0]["policy"] is None
    assert legacy.calls[0]["model"] == expert.model


async def test_policy_aware_gateway_receives_the_resolved_policy() -> None:
    """The modern path passes the expert's own policy, not a shared one."""
    modern = _RecordingGateway()
    runtime = _PanelRuntime()

    await _call_with_budget(runtime, modern, MODERATOR, "user message")
    await _call_with_budget(runtime, modern, ANALYST, "user message")
    await _call_with_budget(runtime, modern, MODERATOR, "user message", policy=REBUTTAL_POLICY)

    assert gateway_accepts_policy(modern) is True
    assert [call["policy"] for call in modern.calls] == [
        MODERATOR_POLICY, ANALYST_POLICY, REBUTTAL_POLICY,
    ]
    # The rebuttal round reuses the moderator expert under a tighter policy —
    # the call site's explicit policy must win over the expert's own row.
    assert modern.calls[2]["policy"] is not None
    assert modern.calls[2]["policy"].max_output_tokens == 800


def test_gateway_accepts_policy_probe_shapes() -> None:
    """The probe is a capability check: signature, ``**kwargs``, or neither."""

    class VarKwargs:
        async def complete(self, system: str, user: str, **kwargs: Any) -> str:
            return ""

    class NoComplete:
        pass

    assert gateway_accepts_policy(_LegacyGateway()) is False
    assert gateway_accepts_policy(_RecordingGateway()) is True
    assert gateway_accepts_policy(VarKwargs()) is True
    assert gateway_accepts_policy(NoComplete()) is False
    assert gateway_accepts_policy(LangChainGateway(api_key="k", base_url=ECNU_BASE_URL)) is True


async def test_panel_labels_the_call_for_observability() -> None:
    """The graph owns graph/node/role labels; the policy supplies node and role."""
    from mindflow.services.llm_observability import current_llm_labels

    seen: list[tuple[str, str, str]] = []

    class LabelCapturingGateway:
        async def complete(
            self,
            system: str,
            user: str,
            model: str = "chat",
            policy: CompletionPolicy | None = None,
        ) -> str:
            seen.append(current_llm_labels())
            return "{}"

    await _call_with_budget(_PanelRuntime(), LabelCapturingGateway(), MODERATOR, "u")
    assert seen == [("panel", MODERATOR_POLICY.node, MODERATOR_POLICY.role)]


async def test_experts_module_still_exposes_the_role_strings_the_table_keys_on() -> None:
    """A rename in experts.py would silently orphan the lookup — pin it."""
    assert ANALYST.role == "数据分析师"
    assert MODERATOR.role == "综合主持人"
    assert CRITIC.role == "批评家"
    assert [expert.role for expert in ATTRIBUTION_EXPERTS] == [
        "CBT归因专家", "TMT归因专家", "情绪调节归因专家",
    ]
    # The table is built against the live instances, not a copy of their names.
    for expert in (*ATTRIBUTION_EXPERTS, ANALYST, CRITIC, MODERATOR):
        assert policy_for_expert(expert) is policy_for_role(expert.role)


async def test_policy_overrides_survive_a_concurrent_gateway_run() -> None:
    """Concurrent roles on one gateway must not observe each other's overrides."""
    wire = _ok_wire()
    gateway = LangChainGateway(
        api_key="test-key", base_url=ECNU_BASE_URL, llm_settings=make_settings(),
    )
    roles = (ANALYST_POLICY, MODERATOR_POLICY, CRITIC_POLICY, CHAT_POLICY)
    with wire.patch_async_client():
        await asyncio.gather(*(
            gateway.complete("s", f"user-{policy.role}", model="chat", policy=policy)
            for policy in roles
        ))
        await gateway.close()

    caps = sorted(body["max_completion_tokens"] for body in wire.recorder.payloads)
    assert caps == sorted(policy.max_output_tokens for policy in roles)


# ── Test doubles ──────────────────────────────────────────────────────────────


@dataclass
class _LegacyGateway:
    """A gateway whose ``complete`` predates ``CompletionPolicy``."""

    calls: list[dict[str, Any]] = field(default_factory=list)

    async def complete(
        self,
        system: str,
        user: str,
        model: Literal["chat", "reasoner"] = "chat",
    ) -> str:
        self.calls.append({"system": system, "user": user, "model": model, "policy": None})
        return '{"legacy": true}'

    async def close(self) -> None:
        return None


@dataclass
class _RecordingGateway:
    """A policy-aware gateway that records the policy it was handed."""

    calls: list[dict[str, Any]] = field(default_factory=list)

    async def complete(
        self,
        system: str,
        user: str,
        model: Literal["chat", "reasoner"] = "chat",
        policy: CompletionPolicy | None = None,
    ) -> str:
        self.calls.append({"system": system, "user": user, "model": model, "policy": policy})
        return "{}"

    async def close(self) -> None:
        return None


@dataclass
class _PanelRuntime:
    """Minimal stand-in for ``_PanelRunContext`` (budget bookkeeping only)."""

    call_count: int = 0
    phase_usage: dict[str, int] = field(default_factory=dict)
    budget_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
