"""Explicit campus selection uses one isolated provider across all L1 entry points."""

from __future__ import annotations

import httpx
import pytest

from mindflow.config import LLMSettings
from mindflow.infrastructure.llm.ecnu import ECNUChatModel
from mindflow.infrastructure.llm.schemas import LLMAttributionResult
from mindflow.infrastructure.provider_registry import ProviderRegistry
from tests._llm_test_support import MockLLMWire, chat_completion


@pytest.mark.parametrize("model", ["ecnu-max", "ecnu-plus"])
async def test_explicit_ecnu_routes_registry_and_requests_to_campus(model: str) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer sk-campus-sentinel"
        content = LLMAttributionResult(
            procrastination_types=["impulsivity"],
            type_confidence={"impulsivity": 0.82},
            cbt_technique="stimulus_control",
            response_text="测试回应",
            next_action="测试行动",
        ).model_dump_json()
        return chat_completion(content, model=model)

    wire = MockLLMWire(handler)
    with wire.patch_async_client():
        settings = LLMSettings(
            service="ecnu",
            ecnu_model=model,
            ecnu_api_key="sk-campus-sentinel",
            deepseek_api_key="sk-direct-never-send",
            max_retries=0,
        )
        registry = ProviderRegistry(settings)
        try:
            assert settings.ecnu_compat_enabled is False
            assert isinstance(registry.get_chat_model(), ECNUChatModel)
            structured = registry.get_structured_attribution()
            assert structured is not None
            assert structured.model == model
            assert structured._base_url == "https://chat.ecnu.edu.cn/open/api/v1"
            assert registry.get_fanout_gateway() is registry.get_gateway()
            await registry.get_gateway().complete("system", "user", model="chat")
            await registry.get_gateway().complete("system", "user", model="reasoner")
            result = await structured.analyze("{}")
            assert result.procrastination_types == ["impulsivity"]
            chat_model = registry.get_chat_model()
            assert chat_model is not None
            await chat_model.ainvoke("测试问题")
        finally:
            await registry.shutdown()
    assert wire.recorder.total_requests == 4
    assert all("chat.ecnu.edu.cn/open/api/v1/chat/completions" in u for u in wire.recorder.urls)
    assert all(p["model"] == model for p in wire.recorder.payloads)
    assert all(p["thinking"]["type"] == "enabled" for p in wire.recorder.payloads)


async def test_selected_ecnu_without_key_does_not_borrow_direct_or_legacy_key() -> None:
    registry = ProviderRegistry(
        LLMSettings(service="ecnu", deepseek_api_key="sk-direct-test", api_key="sk-legacy-test")
    )
    try:
        assert registry.get_chat_model() is None
        assert registry.get_structured_attribution() is None
        assert registry.describe()["credential_present"] is False
    finally:
        await registry.shutdown()
