"""Add panel_transcript_json column to procrastination_analyses.

Stores expert panel deliberation transcripts (human-readable expert remarks)
for the G003 multi-expert panel integration. Nullable TEXT — only populated
when the expert panel is active (not degraded to single-expert mode).

Revision ID: 0002_add_panel_transcript
Revises: 0001_create_core_tables
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002_add_panel_transcript"
down_revision: Union[str, None] = "0001_create_core_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add panel_transcript_json column to procrastination_analyses."""
    op.add_column(
        "procrastination_analyses",
        sa.Column("panel_transcript_json", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    """Drop panel_transcript_json column."""
    op.drop_column("procrastination_analyses", "panel_transcript_json")
