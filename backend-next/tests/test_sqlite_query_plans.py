"""Phase 4.1: SQLite query plans for the hot telemetry/chat queries.

The plan is explicit that optimisation must follow measurement: before adding
indexes or moving to PostgreSQL, the hot queries are run through
``EXPLAIN QUERY PLAN`` against the *migrated* schema (not the metadata-only
schema used by unit tests, which does not contain the alembic-created indexes).

Pinned here:
  * activity-range queries use an index on ``(user_id, timestamp)``,
  * feature-window queries (all + bounded range) use an index on
    ``(user_id, window_start_utc)``,
  * the chat "recent" query keeps its ``(session_id, created_at, id)`` index,
  * the inference projection reads only the columns inference needs.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from mindflow.infrastructure.migrations import run_migrations


@pytest.fixture(scope="module")
def migrated_db(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Apply the real alembic chain once per module and return the DB path."""
    import asyncio

    path = tmp_path_factory.mktemp("query_plans") / "plans.db"
    assert asyncio.run(run_migrations(f"sqlite+aiosqlite:///{path}")) is True
    yield path


@pytest.fixture
def conn(migrated_db: Path) -> Iterator[sqlite3.Connection]:
    """A synchronous connection with representative rows inserted."""
    connection = sqlite3.connect(migrated_db)
    connection.row_factory = sqlite3.Row
    try:
        _seed(connection)
        yield connection
    finally:
        connection.close()


def _seed(connection: sqlite3.Connection) -> None:
    """Insert a handful of rows so the planner has something to estimate from.

    Row counts are irrelevant to plan *shape* here (SQLite plans on structure,
    not statistics, for these queries), but ANALYZE is run anyway so the test
    stays honest if the planner ever starts using sqlite_stat1.
    """
    connection.executescript(
        """
        INSERT OR IGNORE INTO activity_events (id, user_id, timestamp, duration_s, data_json)
        VALUES ('e1', 1, '2026-09-01T10:00:00+00:00', 5.0, '{"process_name":"code.exe"}'),
               ('e2', 1, '2026-09-01T11:00:00+00:00', 5.0, '{"process_name":"code.exe"}'),
               ('e3', 2, '2026-09-01T11:00:00+00:00', 5.0, '{"process_name":"code.exe"}');

        INSERT OR IGNORE INTO behavior_feature_windows
            (id, user_id, window_start_utc, window_end_utc, feature_schema_version,
             features_json)
        VALUES ('w1', 1, '2026-09-01T10:00:00+00:00', '2026-09-01T10:05:00+00:00', 4, '{}'),
               ('w2', 1, '2026-09-01T10:05:00+00:00', '2026-09-01T10:10:00+00:00', 4, '{}'),
               ('w3', 1, '2026-09-01T10:10:00+00:00', '2026-09-01T10:15:00+00:00', 3, '{}');

        INSERT OR IGNORE INTO chat_messages (id, user_id, session_id, role, content, created_at)
        VALUES ('c1', 1, 's1', 'user', 'hi', '2026-09-01T10:00:00+00:00'),
               ('c2', 1, 's1', 'assistant', 'hello', '2026-09-01T10:00:01+00:00'),
               ('c3', 2, 's2', 'user', 'hi', '2026-09-01T10:00:02+00:00');
        """
    )
    connection.execute("ANALYZE")
    connection.commit()

    # The migration chain may use a different feature_schema_version value than
    # the seed above; keep only the rows the queries below are expected to see.
    connection.execute("DELETE FROM behavior_feature_windows WHERE id = 'w3'")
    connection.commit()


def _plan(connection: sqlite3.Connection, sql: str, params: tuple = ()) -> str:
    rows = connection.execute(f"EXPLAIN QUERY PLAN {sql}", params).fetchall()
    return " | ".join(str(row["detail"]) for row in rows)


# ── Hot query shapes (mirrored from the repositories) ───────────────────────

ACTIVITY_RANGE = (
    "SELECT id, user_id, timestamp, duration_s, data_json FROM activity_events "
    "WHERE user_id = ? AND timestamp >= ? AND timestamp < ? ORDER BY timestamp"
)

FEATURE_WINDOWS_ALL = (
    "SELECT * FROM behavior_feature_windows WHERE user_id = ? "
    "AND feature_schema_version = ? ORDER BY window_start_utc ASC"
)

FEATURE_WINDOWS_RANGE = (
    "SELECT * FROM behavior_feature_windows WHERE user_id = ? "
    "AND feature_schema_version = ? AND window_start_utc >= ? AND window_start_utc < ? "
    "ORDER BY window_start_utc ASC"
)

CHAT_RECENT = (
    "SELECT id, user_id, session_id, role, content, created_at FROM chat_messages "
    "WHERE session_id = ? ORDER BY created_at DESC, id DESC LIMIT 20"
)


@pytest.mark.parametrize(
    ("label", "sql", "params"),
    [
        ("activity range", ACTIVITY_RANGE,
         (1, "2026-09-01T00:00:00+00:00", "2026-09-02T00:00:00+00:00")),
        ("feature windows all", FEATURE_WINDOWS_ALL, (1, 4)),
        ("feature windows range", FEATURE_WINDOWS_RANGE,
         (1, 4, "2026-09-01T00:00:00+00:00", "2026-09-02T00:00:00+00:00")),
        ("chat recent", CHAT_RECENT, ("s1",)),
    ],
)
def test_hot_queries_use_indexes(label: str, sql: str, params: tuple, conn) -> None:
    """No hot query may full-scan its table."""
    detail = _plan(conn, sql, params)
    assert "USING INDEX" in detail or "USING COVERING INDEX" in detail, (
        f"{label} is not index-backed: {detail}"
    )
    assert "SCAN activity_events" not in detail
    assert "SCAN behavior_feature_windows" not in detail
    assert "SCAN chat_messages" not in detail


def test_activity_and_window_indexes_exist(conn) -> None:
    """The indexes the plans rely on are actually created by the migrations."""
    names = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
    }
    assert "idx_events_user_time_id" in names
    assert "idx_feature_windows_user_time" in names
    assert "idx_chat_session_recent" in names


def test_feature_window_columns_are_the_expected_projection(conn) -> None:
    """Feature windows expose the f01..f24 columns plus metadata for inference."""
    columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(behavior_feature_windows)")
    }
    for index in range(1, 25):
        assert f"f{index:02d}" in columns, f"missing f{index:02d}"
    for required in (
        "window_start_utc",
        "window_end_utc",
        "feature_schema_version",
        "quality_json",
        "user_id",
    ):
        assert required in columns
