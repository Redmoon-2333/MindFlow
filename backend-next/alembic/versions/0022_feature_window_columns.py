# -*- coding: utf-8 -*-
"""Add f01..f24 REAL columns to behavior_feature_windows (plan I).

The 24-dim feature vector was stored as a single features_json blob, forcing
every training/inference load to json.loads the whole column. This migration
adds one REAL column per feature so ML queries can read vectors directly via
SQL (and future single-feature indexes are possible), while keeping
features_json for backward compatibility.

Revision ID: 0022_feature_window_columns
Revises: 0021_create_collector_intervals
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0022_feature_window_columns"
down_revision: str | None = "0021_create_collector_intervals"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The canonical 24-feature vocabulary (order matches V2_FEATURE_NAMES).
_FEATURE_COLUMNS: tuple[str, ...] = (
    "f01", "f02", "f03", "f04", "f05", "f06",
    "f07", "f08", "f09", "f10", "f11", "f12",
    "f13", "f14", "f15", "f16", "f17", "f18",
    "f19", "f20", "f21", "f22", "f23", "f24",
)


def upgrade() -> None:
    """Add the 24 REAL feature columns (nullable; backfilled by code)."""
    for col in _FEATURE_COLUMNS:
        op.add_column(
            "behavior_feature_windows",
            sa.Column(col, sa.Float(), nullable=True),
        )


def downgrade() -> None:
    """Drop the feature columns (features_json remains authoritative)."""
    for col in _FEATURE_COLUMNS:
        op.drop_column("behavior_feature_windows", col)
