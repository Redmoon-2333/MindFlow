"""Unit tests for the panel parallel-batch retry helper.

Covers the transient-transport-failure resilience added 2026-08-20: when a
whole parallel expert batch comes back empty (all ``_safe_call_with_budget``
calls swallowed a connection error into ''), the batch is retried once after a
short backoff.
"""

from __future__ import annotations

import pytest

from mindflow.graph import panel_graph
from mindflow.graph.panel_graph import _fanout_raw_with_batch_retry


@pytest.mark.asyncio
async def test_retries_batch_once_when_all_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All-empty first attempt -> the batch is retried and returns the retry."""
    monkeypatch.setattr(panel_graph, "_BATCH_RETRY_DELAY_S", 0.0)
    state = {"n": 0}

    async def factory() -> str:
        state["n"] += 1
        return "" if state["n"] <= 2 else "cached"

    result = await _fanout_raw_with_batch_retry(
        [factory, factory], batch_label="attribution",
    )
    assert result == ["cached", "cached"]
    assert state["n"] == 4  # 2 per attempt x 2 attempts


@pytest.mark.asyncio
async def test_no_retry_when_some_responses_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partially successful batch is not retried (only all-empty triggers)."""
    monkeypatch.setattr(panel_graph, "_BATCH_RETRY_DELAY_S", 0.0)
    calls: list[str] = []

    async def factory_a() -> str:
        calls.append("a")
        return "opinion"

    async def factory_b() -> str:
        calls.append("b")
        return ""

    result = await _fanout_raw_with_batch_retry(
        [factory_a, factory_b], batch_label="attribution",
    )
    assert result == ["opinion", ""]
    assert len(calls) == 2  # no retry


@pytest.mark.asyncio
async def test_still_empty_after_retry_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistently empty batch stays empty after the single retry."""
    monkeypatch.setattr(panel_graph, "_BATCH_RETRY_DELAY_S", 0.0)
    calls = 0

    async def factory() -> str:
        nonlocal calls
        calls += 1
        return ""

    result = await _fanout_raw_with_batch_retry(
        [factory, factory], batch_label="rebuttal",
    )
    assert result == ["", ""]
    assert calls == 4


@pytest.mark.asyncio
async def test_empty_task_list_returns_empty() -> None:
    assert await _fanout_raw_with_batch_retry([], batch_label="x") == []


@pytest.mark.asyncio
async def test_exceptions_propagate_from_batch() -> None:
    """Non-budget exceptions from a raw call still propagate (caller handles)."""

    async def factory() -> str:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await _fanout_raw_with_batch_retry([factory], batch_label="x")
