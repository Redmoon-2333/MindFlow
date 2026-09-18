"""Shared concurrency gate for outbound LLM calls.

The ECNU gateway rejects requests beyond 3 in flight per user per model, and
the project rule for this migration is to start at 1 and only raise the cap
with measured evidence. Centralising the gate here keeps every entry point
(panel experts, chat, attribution, intervention copy) on the same budget
instead of each call site inventing its own.

The limiter is deliberately simple and provider-neutral: an ``asyncio.Semaphore``
plus a small counter for observability. It is *not* a rate limiter — retry and
backoff policy stays with the callers.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass
class ConcurrencyStats:
    """Observable counters for the concurrency gate."""

    acquired: int = 0
    max_in_flight: int = 0
    current_in_flight: int = 0
    _peak_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def snapshot(self) -> dict[str, int]:
        return {
            "acquired": self.acquired,
            "max_in_flight": self.max_in_flight,
            "current_in_flight": self.current_in_flight,
        }


class LLMConcurrencyGate:
    """Bound the number of simultaneous generations across all consumers.

    Args:
        limit: Maximum concurrent generations. ``None`` or ``< 1`` means
            unbounded (used by offline paths that never touch the network).
    """

    def __init__(self, limit: int | None = 1) -> None:
        self._limit = limit if (limit is not None and limit >= 1) else None
        self._semaphore = asyncio.Semaphore(self._limit) if self._limit else None
        self.stats = ConcurrencyStats()

    @property
    def limit(self) -> int | None:
        return self._limit

    async def __aenter__(self) -> LLMConcurrencyGate:
        if self._semaphore is not None:
            await self._semaphore.acquire()
        self.stats.acquired += 1
        self.stats.current_in_flight += 1
        if self.stats.current_in_flight > self.stats.max_in_flight:
            self.stats.max_in_flight = self.stats.current_in_flight
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.stats.current_in_flight -= 1
        if self._semaphore is not None:
            self._semaphore.release()

    def acquire(self) -> _GateContext | _NullAsyncContext:
        """Return an async context manager for one generation slot."""
        if self._semaphore is None:
            return _NullAsyncContext()
        return _GateContext(self)

    def snapshot(self) -> dict[str, int | None]:
        return {"limit": self._limit, **self.stats.snapshot()}


class _GateContext:
    def __init__(self, gate: LLMConcurrencyGate) -> None:
        self._gate = gate

    async def __aenter__(self) -> None:
        await self._gate.__aenter__()

    async def __aexit__(self, *exc_info: object) -> None:
        await self._gate.__aexit__(*exc_info)


class _NullAsyncContext:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *exc_info: object) -> None:
        return None


__all__ = ["ConcurrencyStats", "LLMConcurrencyGate"]
