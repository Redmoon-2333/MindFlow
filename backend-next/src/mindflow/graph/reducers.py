"""Pure-function reducers for MindFlow LangGraph state channels.

Each reducer follows the LangGraph contract::

    reducer(current_value, update) → new_value

where ``current_value`` may be ``None`` (channel not yet initialized)
and ``update`` is a single element produced by a graph node.

All reducers are:
  - Pure: no I/O, no side effects, no mutation.
  - Deterministic: same inputs always produce the same outputs.
  - Idempotent with respect to duplicates (no double-insertion).
"""

from __future__ import annotations

import hashlib
from typing import overload

from mindflow.agents.types import ExpertOpinion, TranscriptEntry

# ═══════════════════════════════════════════════════════════════════════════════
# append_opinion — order-independent fan-in for parallel expert updates
# ═══════════════════════════════════════════════════════════════════════════════


def _opinion_sort_key(opinion: ExpertOpinion) -> tuple[str, str]:
    """Deterministic sort key: (role, perspective).

    When multiple attribution experts run in parallel and fan-in, the
    reducer must produce the same ordering regardless of completion
    order.  Sorting by role first, then perspective second, guarantees
    this since each expert has a unique ``(role, perspective)`` pair.
    """
    return (opinion.role, opinion.perspective)


@overload
def append_opinion(
    existing: None,
    new: ExpertOpinion,
) -> tuple[ExpertOpinion]: ...


@overload
def append_opinion(
    existing: tuple[ExpertOpinion, ...],
    new: ExpertOpinion,
) -> tuple[ExpertOpinion, ...]: ...


def append_opinion(
    existing: tuple[ExpertOpinion, ...] | None,
    new: ExpertOpinion,
) -> tuple[ExpertOpinion, ...]:
    """Accumulate expert opinions with deterministic ordering.

    Designed for parallel fan-in: when the graph fans out to multiple
    attribution expert nodes and each returns an ``ExpertOpinion``, the
    reducer collects them into a single sorted channel regardless of
    completion order.

    Duplicate detection: if an opinion with the same ``(role, perspective)``
    already exists, the newer one replaces it (upsert semantics).

    Args:
        existing: Current opinions tuple, or None if channel is uninitialized.
        new: An ``ExpertOpinion`` produced by a graph node.

    Returns:
        A new tuple of ``ExpertOpinion`` sorted by (role, perspective).
    """
    if existing is None:
        return (new,)

    # Upsert: replace existing opinion with same (role, perspective)
    seen: dict[tuple[str, str], ExpertOpinion] = {
        _opinion_sort_key(o): o for o in existing
    }
    seen[_opinion_sort_key(new)] = new

    return tuple(sorted(seen.values(), key=_opinion_sort_key))


# ═══════════════════════════════════════════════════════════════════════════════
# append_transcript — sequential transcript accumulation
# ═══════════════════════════════════════════════════════════════════════════════


@overload
def append_transcript(
    existing: None,
    new: TranscriptEntry,
) -> tuple[TranscriptEntry]: ...


@overload
def append_transcript(
    existing: tuple[TranscriptEntry, ...],
    new: TranscriptEntry,
) -> tuple[TranscriptEntry, ...]: ...


def append_transcript(
    existing: tuple[TranscriptEntry, ...] | None,
    new: TranscriptEntry,
) -> tuple[TranscriptEntry, ...]:
    """Append a new ``TranscriptEntry`` to the deliberation transcript.

    No deduplication — transcript entries are append-only and order is
    meaningful (round 0 analyst, round 1 attribution, etc.).

    Args:
        existing: Current transcript tuple, or None.
        new: A ``TranscriptEntry`` to append.

    Returns:
        New tuple with *new* appended at the end.
    """
    if existing is None:
        return (new,)
    return existing + (new,)


# ═══════════════════════════════════════════════════════════════════════════════
# accumulate_tool_messages — deduplicated tool call/result collection
# ═══════════════════════════════════════════════════════════════════════════════


def _tool_msg_digest(msg: dict[str, str]) -> str:
    """Stable fingerprint for deduplication.

    Uses SHA-256 over (type, name, content) so messages with the same
    semantic content are treated as duplicates regardless of insertion
    order or extraneous keys.
    """
    raw = f"{msg.get('type','')}|{msg.get('name','')}|{msg.get('content','')}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@overload
def accumulate_tool_messages(
    existing: None,
    new: dict[str, str],
) -> tuple[dict[str, str]]: ...


@overload
def accumulate_tool_messages(
    existing: tuple[dict[str, str], ...],
    new: dict[str, str],
) -> tuple[dict[str, str], ...]: ...


def accumulate_tool_messages(
    existing: tuple[dict[str, str], ...] | None,
    new: dict[str, str],
) -> tuple[dict[str, str], ...]:
    """Accumulate tool-call and tool-result messages, deduplicating by content.

    In a LangChain agent tool loop, the same tool may be invoked multiple
    times (or the LLM may hallucinate duplicate calls).  This reducer
    ensures each unique ``(type, name, content)`` triplet appears only once.

    Args:
        existing: Current tool messages tuple, or None.
        new: A tool message dict with keys ``"type"`` (``"call"`` or
            ``"result"``), ``"name"``, and ``"content"``.

    Returns:
        New tuple with *new* appended if unique, or original tuple if
        duplicate.
    """
    if existing is None:
        return (new,)

    new_digest = _tool_msg_digest(new)
    for msg in existing:
        if _tool_msg_digest(msg) == new_digest:
            return existing  # duplicate, no-op
    return existing + (new,)


# ═══════════════════════════════════════════════════════════════════════════════
# accumulate_errors — unique error collection by key
# ═══════════════════════════════════════════════════════════════════════════════


@overload
def accumulate_errors(
    existing: None,
    new: dict[str, str],
) -> tuple[dict[str, str]]: ...


@overload
def accumulate_errors(
    existing: tuple[dict[str, str], ...],
    new: dict[str, str],
) -> tuple[dict[str, str], ...]: ...


def accumulate_errors(
    existing: tuple[dict[str, str], ...] | None,
    new: dict[str, str],
) -> tuple[dict[str, str], ...]:
    """Collect unique errors by key, ignoring duplicates.

    Each error dict must have a ``"key"`` field (stable error identifier)
    and a ``"message"`` field (human-readable description).  Errors with
    the same key are treated as duplicates and silently dropped.

    Args:
        existing: Current errors tuple, or None.
        new: An error dict with ``"key"`` and ``"message"``.

    Returns:
        New tuple with *new* appended if the key is unique, or the
        original tuple unchanged.
    """
    new_key = new.get("key", "")
    if not new_key:
        if existing is None:
            return (new,)
        return existing + (new,)

    if existing is None:
        return (new,)

    for err in existing:
        if err.get("key") == new_key:
            return existing  # duplicate key, no-op

    return existing + (new,)
