"""Add activity cleanup indexes and generated JSON projection columns.

Revision ID: 0008_optimize_activity_telemetry
Revises: 0007_create_telemetry_tables
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0008_optimize_activity_telemetry"
down_revision: str | None = "0007_create_telemetry_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add low-cost virtual projections and indexes for hot SQLite queries."""
    op.execute(
        "ALTER TABLE activity_events ADD COLUMN app_name TEXT "
        "GENERATED ALWAYS AS (json_extract(data_json, '$.app_name')) VIRTUAL"
    )
    op.execute(
        "ALTER TABLE activity_events ADD COLUMN process_name TEXT "
        "GENERATED ALWAYS AS (json_extract(data_json, '$.process_name')) VIRTUAL"
    )
    op.execute(
        "ALTER TABLE activity_events ADD COLUMN window_title TEXT "
        "GENERATED ALWAYS AS (json_extract(data_json, '$.window_title')) VIRTUAL"
    )
    op.execute(
        "ALTER TABLE activity_events ADD COLUMN is_idle INTEGER "
        "GENERATED ALWAYS AS (json_extract(data_json, '$.is_idle')) VIRTUAL"
    )
    op.create_index(
        "idx_events_cleanup_time",
        "activity_events",
        ["timestamp", "id"],
    )
    op.create_index(
        "idx_events_user_process_time",
        "activity_events",
        ["user_id", "process_name", "timestamp"],
    )
    op.create_index(
        "idx_events_user_time_id",
        "activity_events",
        ["user_id", "timestamp", "id"],
    )


def downgrade() -> None:
    """Remove activity performance projections and indexes."""
    op.drop_index("idx_events_user_time_id", table_name="activity_events")
    op.drop_index("idx_events_user_process_time", table_name="activity_events")
    op.drop_index("idx_events_cleanup_time", table_name="activity_events")
    op.execute("ALTER TABLE activity_events DROP COLUMN is_idle")
    op.execute("ALTER TABLE activity_events DROP COLUMN window_title")
    op.execute("ALTER TABLE activity_events DROP COLUMN process_name")
    op.execute("ALTER TABLE activity_events DROP COLUMN app_name")
