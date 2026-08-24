"""Add analysis_kind and source columns to procrastination_analyses.

Separates workflow identity (analysis_kind) from degradation provider (source),
enabling multiple analysis kinds to coexist per (user_id, date) without
overwriting each other.

The migration recreates the table because SQLite cannot ALTER a UNIQUE
constraint in-place and batch_alter_table would preserve the old unnamed
UNIQUE(user_id, date) alongside the new one.

Revision ID: 0011_add_analysis_kind
Revises: 0010_add_scheduled_job_heartbeat
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0011_add_analysis_kind"
down_revision: str | None = "0010_add_scheduled_job_heartbeat"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NEW_COLUMNS = """\
        id TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        procrastination_types_json TEXT,
        type_confidence_json TEXT,
        cognitive_distortions_json TEXT,
        cbt_technique TEXT,
        response_text TEXT,
        llm_model TEXT,
        llm_cost_usd REAL,
        panel_transcript_json TEXT,
        created_at TEXT NOT NULL
            DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
        analysis_kind TEXT NOT NULL DEFAULT 'daily_attribution',
        source TEXT"""

_OLD_COLUMNS = """\
        id TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        procrastination_types_json TEXT,
        type_confidence_json TEXT,
        cognitive_distortions_json TEXT,
        cbt_technique TEXT,
        response_text TEXT,
        llm_model TEXT,
        llm_cost_usd REAL,
        panel_transcript_json TEXT,
        created_at TEXT NOT NULL
            DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))"""

_NEW_COPY = """\
        SELECT id, user_id, date, procrastination_types_json,
               type_confidence_json, cognitive_distortions_json,
               cbt_technique, response_text, llm_model,
               llm_cost_usd, panel_transcript_json, created_at,
               'legacy_unknown', NULL
        FROM procrastination_analyses"""

_OLD_COPY = """\
        SELECT id, user_id, date, procrastination_types_json,
               type_confidence_json, cognitive_distortions_json,
               cbt_technique, response_text, llm_model,
               llm_cost_usd, panel_transcript_json, created_at
        FROM procrastination_analyses
        GROUP BY user_id, date
        HAVING id = MIN(id)"""


def _execute_sqlite_ddl(
    table_name: str, columns_sql: str, copy_sql: str, constraint_sql: str,
) -> None:
    """Recreate *table_name* with new columns and constraint.

    Executes each DDL statement separately because SQLite's execute()
    handles only one statement per call.  Alembic's outer transaction
    keeps the whole migration atomic.
    """
    op.execute("PRAGMA foreign_keys=off")
    op.execute(
        f"CREATE TABLE {table_name}_new (\n{columns_sql}\n{constraint_sql}\n)"
    )
    op.execute(f"INSERT INTO {table_name}_new {copy_sql}")
    op.execute(f"DROP TABLE {table_name}")
    op.execute(f"ALTER TABLE {table_name}_new RENAME TO {table_name}")
    op.execute("PRAGMA foreign_keys=on")


def upgrade() -> None:
    """Add analysis_kind + source, backfill, and widen unique constraint."""
    # ── Step 1: Recreate table with new columns + widened constraint ──
    _execute_sqlite_ddl(
        table_name="procrastination_analyses",
        columns_sql=_NEW_COLUMNS,
        copy_sql=_NEW_COPY,
        constraint_sql=",   UNIQUE(user_id, date, analysis_kind)",
    )

    # ── Step 2: Backfill analysis_kind for rows with known provenance ─
    op.execute(
        "UPDATE procrastination_analyses "
        "SET analysis_kind = 'daily_panel' "
        "WHERE panel_transcript_json IS NOT NULL"
    )
    op.execute(
        "UPDATE procrastination_analyses "
        "SET analysis_kind = 'daily_panel' "
        "WHERE llm_model = 'panel' AND analysis_kind = 'legacy_unknown'"
    )
    op.execute(
        "UPDATE procrastination_analyses "
        "SET analysis_kind = 'daily_attribution' "
        "WHERE analysis_kind = 'legacy_unknown'"
    )
    # Copy llm_model → source for recognised degradation tiers
    op.execute(
        "UPDATE procrastination_analyses "
        "SET source = llm_model "
        "WHERE llm_model IN ('panel', 'single_expert', 'ollama', 'rule_engine') "
        "AND source IS NULL"
    )


def downgrade() -> None:
    """Revert to the old (user_id, date) unique constraint."""
    _execute_sqlite_ddl(
        table_name="procrastination_analyses",
        columns_sql=_OLD_COLUMNS,
        copy_sql=_OLD_COPY,
        constraint_sql=",   UNIQUE(user_id, date)",
    )
