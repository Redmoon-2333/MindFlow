"""Add degradation metadata to procrastination_analyses.

Revision ID: 0015_add_panel_degradation_meta
"""

import sqlalchemy as sa

from alembic import op

revision: str = "0015_add_panel_degradation_meta"
down_revision: str | None = "0014_add_intervention_title_message"


def upgrade() -> None:
    op.add_column("procrastination_analyses", sa.Column("degraded", sa.Boolean(), nullable=True))
    op.add_column(
        "procrastination_analyses",
        sa.Column("degradation_path_json", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("procrastination_analyses", "degradation_path_json")
    op.drop_column("procrastination_analyses", "degraded")
