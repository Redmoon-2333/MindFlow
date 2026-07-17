"""Tests for database migrations (Alembic).

Tests cover:
  - Upgrading from empty database to head
  - All 7 core tables exist after migration
  - Re-running migration is idempotent
  - Downgrade works
  - Migrations run via the async wrapper (run_migrations)

Note: Tests use a temporary database file (not :memory:) because
Alembic migration context requires a persistent URL.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mindflow.infrastructure.migrations import run_migrations


def _get_table_names(sync_url: str) -> set[str]:
    """Query table names from the SQLite database (synchronous)."""
    import sqlite3

    conn = sqlite3.connect(sync_url.replace("sqlite://", ""))
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = {row[0] for row in cursor.fetchall()}
    conn.close()
    return tables


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Create a path for the test database."""
    return tmp_path / "migration_test.db"


@pytest.fixture
def async_db_url(db_path: Path) -> str:
    """Async SQLAlchemy URL for the test database."""
    return f"sqlite+aiosqlite:///{db_path}"


@pytest.fixture
def sync_db_url(db_path: Path) -> str:
    """Sync SQLAlchemy URL for the test database."""
    return f"sqlite:///{db_path}"


@pytest.mark.asyncio
class TestMigrations:
    """Test suite for Alembic migration operations."""

    async def test_migration_upgrade_succeeds(self, async_db_url: str, sync_db_url: str):
        """Running migration on an empty database succeeds."""
        result = await run_migrations(async_db_url)
        assert result is True, "Migration should succeed"

    async def test_all_core_tables_exist(self, async_db_url: str, sync_db_url: str):
        """All 7 core tables exist after migration."""
        await run_migrations(async_db_url)
        tables = _get_table_names(sync_db_url)
        expected = {
            "activity_events",
            "focus_sessions",
            "daily_reports",
            "procrastination_analyses",
            "intervention_logs",
            "baseline_models",
            "user_preferences",
        }
        for table in expected:
            assert table in tables, f"Table {table} not found after migration"

    async def test_alembic_version_table_exists(self, async_db_url: str, sync_db_url: str):
        """Alembic version tracking table is created."""
        await run_migrations(async_db_url)
        tables = _get_table_names(sync_db_url)
        assert "alembic_version" in tables

    async def test_rerun_migration_is_idempotent(self, async_db_url: str, sync_db_url: str):
        """Running migration twice is safe (already at head)."""
        result1 = await run_migrations(async_db_url)
        assert result1 is True
        result2 = await run_migrations(async_db_url)
        assert result2 is True  # Second run should also succeed (no-op)

    async def test_migration_with_existing_tables(self, async_db_url: str, sync_db_url: str):
        """Migration works on a fresh database (no pre-existing tables)."""
        result = await run_migrations(async_db_url)
        assert result is True
        tables = _get_table_names(sync_db_url)
        assert len(tables) >= 7  # 7 core tables + alembic_version

    async def test_columns_have_correct_types(self, async_db_url: str, sync_db_url: str):
        """Verify key column attributes via PRAGMA table_info."""
        await run_migrations(async_db_url)

        sync_path = sync_db_url.replace("sqlite://", "")
        import sqlite3

        conn = sqlite3.connect(sync_path)
        cursor = conn.execute("PRAGMA table_info(activity_events)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}  # name -> type
        conn.close()

        assert columns["id"] == "TEXT"
        assert columns["user_id"] == "INTEGER"
        assert columns["timestamp"] == "TEXT"
        assert columns["data_json"] == "TEXT"
        assert columns["event_type"] == "TEXT"

    async def test_migration_downgrade_removes_core_tables(
        self, async_db_url: str, sync_db_url: str
    ):
        """Upgrade then downgrade to base drops all 7 core tables (P2 review fix)."""
        await run_migrations(async_db_url)

        def _downgrade() -> None:
            from alembic.config import Config

            from alembic import command
            from mindflow.infrastructure.migrations import BASE_DIR

            cfg = Config(str(BASE_DIR / "alembic.ini"))
            cfg.set_main_option("sqlalchemy.url", sync_db_url)
            command.downgrade(cfg, "base")

        import asyncio

        await asyncio.to_thread(_downgrade)

        tables = _get_table_names(sync_db_url)
        core = {
            "activity_events",
            "focus_sessions",
            "daily_reports",
            "procrastination_analyses",
            "intervention_logs",
            "baseline_models",
            "user_preferences",
        }
        assert not (core & tables), f"Core tables still present after downgrade: {core & tables}"

    async def test_indexes_exist(self, async_db_url: str, sync_db_url: str):
        """Verify indexes are created for activity_events."""
        await run_migrations(async_db_url)

        sync_path = sync_db_url.replace("sqlite://", "")
        import sqlite3

        conn = sqlite3.connect(sync_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index' ORDER BY name")
        indexes = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "idx_events_user_time" in indexes
        assert "idx_events_type" in indexes
        assert "idx_sessions_user_date" in indexes
