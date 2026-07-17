"""Domain-level ID generation using UUIDv7 (time-sortable UUIDs).

Uses the uuid6 PyPI library for uuid7() to maintain Python 3.11 compatibility
(ADR-006: uuid.uuid7() is Python 3.12+ only).

UUIDv7 properties:
  - Time-sortable: monotonically increasing within the same millisecond
  - Improves B-tree locality for append-mostly tables
  - Reduces page splits vs UUIDv4 (random)
  - 122 bits of randomness after the timestamp component
"""

from __future__ import annotations

import uuid6


def new_id() -> str:
    """Generate a new time-sortable UUIDv7 string.

    Returns:
        36-character UUID string (e.g. "018f3a6b-7c8d-7e9f-a012-3456789abcde").
    """
    return str(uuid6.uuid7())
