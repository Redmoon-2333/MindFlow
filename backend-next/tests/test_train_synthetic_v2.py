"""Tests for the synthetic v2 feature window generator.

Focuses on the ``generate_v2_synthetic_data`` function from
``mindflow.train.synthetic_v2``.
"""

from __future__ import annotations

import re

import pytest

from mindflow.train.synthetic_v2 import generate_v2_synthetic_data


class TestGenerateV2SyntheticData:
    """Exercise the v2 synthetic data generator."""

    def test_invalid_archetype_id_raises_valueerror(self) -> None:
        """Passing an archetype ID that matches no profile must raise
        ValueError (not NameError due to undefined ``_ARCHETYPES``)
        listing the available profile IDs."""
        with pytest.raises(ValueError, match=re.escape("freshman_cs")):
            generate_v2_synthetic_data(
                archetype_ids=["nonexistent_id_xyz"],
                days_per_archetype=1,
            )

    def test_valid_archetype_id_generates_data(self) -> None:
        """A single valid archetype ID produces non-empty output."""
        windows, feedback = generate_v2_synthetic_data(
            archetype_ids=["freshman_cs"],
            days_per_archetype=1,
        )
        assert len(windows) > 0
        assert all(isinstance(w, dict) for w in windows)
