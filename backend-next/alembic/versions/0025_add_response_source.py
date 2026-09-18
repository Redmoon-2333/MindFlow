"""Add response_source to intervention_logs.

Records whether a response came from a real user action ("human") or from the
reminder client's timeout default ("auto"). Without it the two were
indistinguishable, so an automatic "ignored" written when the popup timed out
could overwrite a response the user had actually given, and duplicate
submissions could rewrite the first answer.

Existing rows keep NULL, which callers treat as "unknown provenance" — the
same convention used for other pre-upgrade data.

Revision ID: 0025_add_response_source
Revises: 0024_create_training_jobs
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0025_add_response_source"
down_revision: str | None = "0024_create_training_jobs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the response_source column (nullable; historical rows stay NULL)."""
    op.add_column(
        "intervention_logs",
        sa.Column("response_source", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    """Drop the response_source column."""
    with op.batch_alter_table("intervention_logs") as batch:
        batch.drop_column("response_source")
