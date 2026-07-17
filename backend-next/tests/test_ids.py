"""Tests for mindflow.domain.ids — UUIDv7 generation.

Tests cover:
  - new_id returns a valid UUID string
  - Generated IDs are time-sortable
  - Multiple IDs are unique
  - Format matches standard UUID (36 chars, 5 groups)
"""

from __future__ import annotations

import uuid as uuid_lib

from mindflow.domain.ids import new_id


class TestNewId:
    """Verify UUIDv7 generation."""

    def test_returns_string(self):
        """new_id returns a string."""
        uid = new_id()
        assert isinstance(uid, str)

    def test_uuid_format(self):
        """Generated ID matches standard UUID format (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)."""
        uid = new_id()
        # uuid.UUID can parse it — validates format
        parsed = uuid_lib.UUID(uid)
        assert str(parsed) == uid

    def test_version_7(self):
        """Generated ID is UUID version 7 (time-ordered)."""
        uid = new_id()
        parsed = uuid_lib.UUID(uid)
        assert parsed.version == 7, f"Expected UUIDv7, got v{parsed.version}"

    def test_unique_ids(self):
        """Multiple calls to new_id return unique values."""
        ids = {new_id() for _ in range(100)}
        assert len(ids) == 100

    def test_time_sortable(self):
        """IDs generated in sequence are monotonically increasing (time-sortable)."""
        ids = [new_id() for _ in range(100)]
        sorted_ids = sorted(ids)
        assert ids == sorted_ids, "IDs should already be in sorted order"

    def test_no_hyphens_in_hex_validation(self):
        """The UUID string uses standard hyphenated format (8-4-4-4-12)."""
        uid = new_id()
        parts = uid.split("-")
        assert len(parts) == 5
        assert len(parts[0]) == 8
        assert len(parts[1]) == 4
        assert len(parts[2]) == 4
        assert len(parts[3]) == 4
        assert len(parts[4]) == 12

    def test_hex_values_are_valid(self):
        """All hex characters in the UUID are valid hexadecimal."""
        uid = new_id()
        hex_part = uid.replace("-", "")
        int(hex_part, 16)  # Will raise ValueError if invalid hex

    def test_time_monotonic_across_batches(self):
        """IDs generated in separate batches maintain monotonicity."""
        batch1 = [new_id() for _ in range(10)]
        batch2 = [new_id() for _ in range(10)]
        all_ids = batch1 + batch2
        assert sorted(all_ids) == all_ids
