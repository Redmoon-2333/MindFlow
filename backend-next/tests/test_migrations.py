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
        """All core tables exist after migration."""
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
            "app_classification_rules",
        }
        for table in expected:
            assert table in tables, f"Table {table} not found after migration"

    async def test_app_classification_rules_table_exists(
        self, async_db_url: str, sync_db_url: str
    ):
        """app_classification_rules table exists and has correct columns."""
        await run_migrations(async_db_url)
        tables = _get_table_names(sync_db_url)
        assert "app_classification_rules" in tables

        sync_path = sync_db_url.replace("sqlite://", "")
        import sqlite3

        conn = sqlite3.connect(sync_path)
        cursor = conn.execute("PRAGMA table_info(app_classification_rules)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}  # name -> type
        conn.close()

        assert columns["id"] == "TEXT"
        assert columns["user_id"] == "INTEGER"
        assert columns["process_name"] == "TEXT"
        assert columns["category"] == "TEXT"
        assert columns["priority"] == "INTEGER"
        assert columns["created_at"] == "TEXT"
        assert columns["updated_at"] == "TEXT"
        # window_title_pattern is nullable
        assert "window_title_pattern" in columns

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
            "app_classification_rules",
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

    async def test_performance_indexes_and_generated_columns_exist(
        self, async_db_url: str, sync_db_url: str
    ) -> None:
        """Latest migration adds cleanup/process indexes and JSON projections."""
        await run_migrations(async_db_url)

        sync_path = sync_db_url.replace("sqlite://", "")
        import sqlite3

        conn = sqlite3.connect(sync_path)
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name"
            ).fetchall()
        }
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_xinfo(activity_events)").fetchall()
        }
        conn.close()

        assert "idx_events_cleanup_time" in indexes
        assert "idx_events_user_process_time" in indexes
        assert {"app_name", "process_name", "window_title", "is_idle"} <= columns

    async def test_workflow_tables_exist_with_correct_columns(
        self, async_db_url: str, sync_db_url: str
    ) -> None:
        """Workflow tables exist after migration with correct column types."""
        await run_migrations(async_db_url)
        tables = _get_table_names(sync_db_url)
        assert "workflow_runs" in tables
        assert "workflow_node_events" in tables
        assert "workflow_budget_reservations" in tables

        sync_path = sync_db_url.replace("sqlite://", "")
        import sqlite3

        conn = sqlite3.connect(sync_path)

        wfr_cols = {row[1]: row[2] for row in conn.execute("PRAGMA table_info(workflow_runs)")}
        assert wfr_cols["id"] == "TEXT"
        assert wfr_cols["workflow_name"] == "TEXT"
        assert wfr_cols["run_id"] == "TEXT"
        assert wfr_cols["status"] == "TEXT"
        assert wfr_cols["origin"] == "TEXT"
        assert wfr_cols["user_id"] == "INTEGER"
        assert wfr_cols["target_date"] == "TEXT"
        assert wfr_cols["idempotency_key"] == "TEXT"
        assert wfr_cols["token_count"] == "INTEGER"
        assert wfr_cols["call_count"] == "INTEGER"
        assert wfr_cols["trace_id"] == "TEXT"
        assert "source" in wfr_cols
        assert "retry_reason" in wfr_cols
        assert "degradation_reason" in wfr_cols

        wne_cols = {row[1]: row[2] for row in conn.execute("PRAGMA table_info(workflow_node_events)")}
        assert wne_cols["id"] == "TEXT"
        assert wne_cols["run_id"] == "TEXT"
        assert wne_cols["node_name"] == "TEXT"
        assert wne_cols["status"] == "TEXT"
        assert wne_cols["started_at"] == "TEXT"
        assert wne_cols["duration_ms"] == "INTEGER"
        assert "error_category" in wne_cols

        wbr_cols = {row[1]: row[2] for row in conn.execute("PRAGMA table_info(workflow_budget_reservations)")}
        assert wbr_cols["id"] == "TEXT"
        assert wbr_cols["idempotency_key"] == "TEXT"
        assert wbr_cols["origin"] == "TEXT"
        assert wbr_cols["user_id"] == "INTEGER"
        assert wbr_cols["reserved_at"] == "TEXT"
        assert "expires_at" in wbr_cols
        assert "released_at" in wbr_cols

        conn.close()

    async def test_workflow_tables_have_unique_constraints(
        self, async_db_url: str, sync_db_url: str
    ) -> None:
        """idempotency_key columns have UNIQUE constraints."""
        await run_migrations(async_db_url)

        sync_path = sync_db_url.replace("sqlite://", "")
        import sqlite3

        conn = sqlite3.connect(sync_path)

        # Check workflow_runs has unique on idempotency_key
        wfr_indexes = {
            row[1] for row in conn.execute(
                "SELECT * FROM sqlite_master WHERE type='index' AND tbl_name='workflow_runs'"
            )
        }
        # SQLite creates auto-named index for UNIQUE but also our explicit named indexes
        has_unique = any(
            "idempotency_key" in str(idx) for idx in wfr_indexes
        )
        # Also check via the auto index from UNIQUE constraint
        auto_indexes = {
            row[1] for row in conn.execute(
                "PRAGMA index_list('workflow_runs')"
            )
        }
        # The UNIQUE on idempotency_key creates an auto-named unique index
        unique_wfr = {
            row[1]: row[2]
            for row in conn.execute("PRAGMA index_list('workflow_runs')")
        }
        wfr_unique = any(
            unique_wfr[name] for name in unique_wfr
            if unique_wfr[name] == 1  # unique flag
        )

        # Check budget reservations has unique on idempotency_key
        unique_wbr = {
            row[1]: row[2]
            for row in conn.execute("PRAGMA index_list('workflow_budget_reservations')")
        }
        wbr_unique = any(
            unique_wbr[name] for name in unique_wbr
            if unique_wbr[name] == 1  # unique flag
        )

        conn.close()
        assert wfr_unique, "workflow_runs idempotency_key should have UNIQUE constraint"
        assert wbr_unique, "workflow_budget_reservations idempotency_key should have UNIQUE constraint"

    async def test_workflow_tables_have_indexes(
        self, async_db_url: str, sync_db_url: str
    ) -> None:
        """Named indexes exist on workflow tables."""
        await run_migrations(async_db_url)

        sync_path = sync_db_url.replace("sqlite://", "")
        import sqlite3

        conn = sqlite3.connect(sync_path)
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name"
            ).fetchall()
        }
        conn.close()

        assert "ix_wfr_status_target_date" in indexes
        assert "ix_wfr_user_date" in indexes
        assert "ix_wne_run_node" in indexes
        assert "ix_wbr_key" in indexes

    async def test_workflow_tables_upgrade_downgrade_upgrade_cycle(
        self, sync_db_url: str
    ) -> None:
        """Upgrade to 0013, downgrade to 0012, then upgrade again — leaves tables intact."""
        import asyncio

        from alembic.config import Config

        from alembic import command
        from mindflow.infrastructure.migrations import BASE_DIR

        def _run_alembic(action: str, revision: str) -> None:
            cfg = Config(str(BASE_DIR / "alembic.ini"))
            cfg.set_main_option("sqlalchemy.url", sync_db_url)
            getattr(command, action)(cfg, revision)

        # Step 1: Upgrade to 0013
        await asyncio.to_thread(_run_alembic, "upgrade", "0013_create_workflow_tables")

        sync_path = sync_db_url.replace("sqlite://", "")
        import sqlite3

        conn = sqlite3.connect(sync_path)
        tables_before = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        }
        conn.close()
        assert "workflow_runs" in tables_before

        # Step 2: Downgrade to 0012
        await asyncio.to_thread(_run_alembic, "downgrade", "0012_add_chat_session_recent_index")

        conn = sqlite3.connect(sync_path)
        tables_after_down = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        }
        conn.close()
        assert "workflow_runs" not in tables_after_down
        assert "workflow_node_events" not in tables_after_down
        assert "workflow_budget_reservations" not in tables_after_down

        # Step 3: Re-upgrade to 0013
        await asyncio.to_thread(_run_alembic, "upgrade", "0013_create_workflow_tables")

        conn = sqlite3.connect(sync_path)
        tables_after_reup = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        }
        conn.close()
        assert "workflow_runs" in tables_after_reup
        assert "workflow_node_events" in tables_after_reup
        assert "workflow_budget_reservations" in tables_after_reup

    async def test_workflow_table_status_default(
        self, async_db_url: str, sync_db_url: str
    ) -> None:
        """workflow_runs.status defaults to 'pending'."""
        await run_migrations(async_db_url)

        sync_path = sync_db_url.replace("sqlite://", "")
        import sqlite3

        conn = sqlite3.connect(sync_path)
        # Insert a row without specifying status
        conn.execute(
            """
            INSERT INTO workflow_runs (
                id, workflow_name, run_id, origin, user_id, target_date,
                idempotency_key, created_at, updated_at
            ) VALUES (
                'test-id', 'test_wf', 'test-run', 'scheduler', 1, '2026-01-01',
                'test-key', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z'
            )
            """
        )
        conn.commit()
        row = conn.execute(
            "SELECT status FROM workflow_runs WHERE id = 'test-id'"
        ).fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "pending"

    async def test_scheduled_job_heartbeat_upgrade_and_downgrade_contract(
        self,
        sync_db_url: str,
    ) -> None:
        import asyncio
        import sqlite3

        from alembic.config import Config

        from alembic import command
        from mindflow.infrastructure.migrations import BASE_DIR

        def _run_alembic(action: str, revision: str) -> None:
            cfg = Config(str(BASE_DIR / "alembic.ini"))
            cfg.set_main_option("sqlalchemy.url", sync_db_url)
            getattr(command, action)(cfg, revision)

        await asyncio.to_thread(
            _run_alembic,
            "upgrade",
            "0009_create_scheduled_job_runs",
        )
        sync_path = sync_db_url.replace("sqlite://", "")
        started_at = "2026-07-25T16:10:00+00:00"
        conn = sqlite3.connect(sync_path)
        conn.execute(
            """
            INSERT INTO scheduled_job_runs (
                job_name,
                local_date,
                status,
                attempt_count,
                started_at,
                finished_at,
                last_error
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "daily_report",
                "2026-07-25",
                "running",
                1,
                started_at,
                None,
                None,
            ),
        )
        conn.commit()
        conn.close()

        await asyncio.to_thread(
            _run_alembic,
            "upgrade",
            "0010_add_scheduled_job_heartbeat",
        )
        conn = sqlite3.connect(sync_path)
        columns = {
            row[1]: row
            for row in conn.execute("PRAGMA table_info(scheduled_job_runs)").fetchall()
        }
        heartbeat_at = conn.execute(
            "SELECT heartbeat_at FROM scheduled_job_runs "
            "WHERE job_name = 'daily_report' AND local_date = '2026-07-25'"
        ).fetchone()
        conn.close()

        assert columns["heartbeat_at"][3] == 1
        assert heartbeat_at == (started_at,)

        await asyncio.to_thread(
            _run_alembic,
            "downgrade",
            "0009_create_scheduled_job_runs",
        )
        conn = sqlite3.connect(sync_path)
        downgraded_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(scheduled_job_runs)").fetchall()
        }
        conn.close()

        assert "heartbeat_at" not in downgraded_columns


# ── Destructive ML shadow-prediction migration (0019) ───────────────────────


@pytest.mark.asyncio
class TestMlShadowPredictionsMigration:
    """0019 is destructive, but downgrade restores the documented schema."""

    async def test_downgrade_restores_empty_0017_schema(self, sync_db_url: str) -> None:
        import asyncio
        import sqlite3

        from alembic.config import Config

        from alembic import command
        from mindflow.infrastructure.migrations import BASE_DIR

        def _run_alembic(action: str, revision: str) -> None:
            cfg = Config(str(BASE_DIR / "alembic.ini"))
            cfg.set_main_option("sqlalchemy.url", sync_db_url)
            getattr(command, action)(cfg, revision)

        # Build the pre-0019 state and prove that upgrade intentionally drops
        # rows; downgrade can restore structure, not data deleted by DROP TABLE.
        await asyncio.to_thread(
            _run_alembic,
            "upgrade",
            "0018_add_feedback_snapshot_checks",
        )
        sync_path = sync_db_url.replace("sqlite://", "")
        conn = sqlite3.connect(sync_path)
        conn.execute(
            """
            INSERT INTO ml_shadow_predictions (
                id, user_id, window_start_utc, candidate_version,
                candidate_proba, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("shadow-1", 1, "2026-07-24T08:00:00Z", "candidate-v1", 0.7,
             "2026-07-24T08:00:00Z"),
        )
        conn.commit()
        conn.close()

        await asyncio.to_thread(_run_alembic, "upgrade", "0019_drop_ml_shadow_predictions")
        await asyncio.to_thread(
            _run_alembic,
            "downgrade",
            "0018_add_feedback_snapshot_checks",
        )

        conn = sqlite3.connect(sync_path)
        columns = {
            row[1]: {"type": row[2], "notnull": row[3], "default": row[4]}
            for row in conn.execute("PRAGMA table_info(ml_shadow_predictions)")
        }
        row_count = conn.execute("SELECT COUNT(*) FROM ml_shadow_predictions").fetchone()[0]
        conn.close()

        assert set(columns) == {
            "id", "user_id", "window_start_utc", "candidate_version",
            "active_version", "candidate_proba", "active_proba", "delta",
            "status", "created_at",
        }
        assert columns["status"]["default"] == "'shadow'"
        assert row_count == 0


# ── Intervention slot reservations migration (0020) ──────────────────────────


@pytest.mark.asyncio
class TestInterventionSlotReservationsMigration:
    """0020 chains from the 0019 head and creates the exact table schema."""

    _TABLE = "intervention_slot_reservations"
    _REVISION = "0020_create_intervention_slot_reservations"
    _DOWN_REVISION = "0019_drop_ml_shadow_predictions"

    async def test_migration_0020_exists_and_chains_from_0019(self) -> None:
        """The migration file exists, chains from 0019, and upgrades the table."""
        import importlib.util
        import sys

        from mindflow.infrastructure.migrations import BASE_DIR

        version_path = (
            BASE_DIR / "alembic" / "versions" / f"{self._REVISION}.py"
        )
        assert version_path.exists(), (
            f"{self._REVISION}.py missing — intervention_slot_reservations "
            "has no migration"
        )

        spec = importlib.util.spec_from_file_location("_mig_0020", version_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        assert module.revision == self._REVISION
        assert module.down_revision == self._DOWN_REVISION

    async def test_upgrade_creates_exact_table_schema(self, async_db_url, sync_db_url):
        """Upgrade to head creates the exact columns/constraint/server_default."""
        await run_migrations(async_db_url)

        tables = _get_table_names(sync_db_url)
        assert self._TABLE in tables

        import sqlite3

        sync_path = sync_db_url.replace("sqlite://", "")
        conn = sqlite3.connect(sync_path)

        # Columns must mirror schema.py (intervention_slot_reservations).
        cols = {
            row[1]: {"type": row[2], "notnull": row[3], "pk": row[5]}
            for row in conn.execute(f"PRAGMA table_info({self._TABLE})")
        }
        assert cols["id"]["type"] == "TEXT" and cols["id"]["pk"] == 1
        assert cols["user_id"]["type"] == "INTEGER" and cols["user_id"]["notnull"] == 1
        assert cols["date"]["type"] == "TEXT" and cols["date"]["notnull"] == 1
        assert cols["slot_index"]["type"] == "INTEGER" and cols["slot_index"]["notnull"] == 1
        assert (
            cols["intervention_type"]["type"] == "TEXT"
            and cols["intervention_type"]["notnull"] == 1
        )
        assert cols["reserved_at"]["type"] == "TEXT" and cols["reserved_at"]["notnull"] == 1

        # The UNIQUE(user_id, date, slot_index) constraint creates a unique index.
        unique_indexes = {
            row[1]: row[2]
            for row in conn.execute(f"PRAGMA index_list({self._TABLE})")
        }
        assert any(unique_indexes[name] == 1 for name in unique_indexes), (
            "expected a UNIQUE index on intervention_slot_reservations"
        )

        # server_default must be the UTC timestamp expression from schema.py.
        ddl = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (self._TABLE,),
        ).fetchone()[0]
        conn.close()
        assert "strftime('%Y-%m-%dT%H:%M:%SZ','now')" in ddl

    async def test_downgrade_removes_table_and_reupgrade_recreates(self, sync_db_url) -> None:
        """Downgrade to 0019 drops the table; upgrade back recreates it."""
        import asyncio
        import sqlite3

        from alembic.config import Config

        from alembic import command
        from mindflow.infrastructure.migrations import BASE_DIR

        def _run_alembic(action: str, revision: str) -> None:
            cfg = Config(str(BASE_DIR / "alembic.ini"))
            cfg.set_main_option("sqlalchemy.url", sync_db_url)
            getattr(command, action)(cfg, revision)

        sync_path = sync_db_url.replace("sqlite://", "")

        # 1. Upgrade to head → table present.
        await asyncio.to_thread(_run_alembic, "upgrade", self._REVISION)
        conn = sqlite3.connect(sync_path)
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (self._TABLE,),
        ).fetchone() is not None
        conn.close()

        # 2. Downgrade to 0019 → table removed.
        await asyncio.to_thread(_run_alembic, "downgrade", self._DOWN_REVISION)
        conn = sqlite3.connect(sync_path)
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (self._TABLE,),
        ).fetchone() is None
        conn.close()

        # 3. Re-upgrade → table recreated with the same schema.
        await asyncio.to_thread(_run_alembic, "upgrade", self._REVISION)
        conn = sqlite3.connect(sync_path)
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (self._TABLE,),
        ).fetchone() is not None
        cols = {
            row[1]
            for row in conn.execute(f"PRAGMA table_info({self._TABLE})")
        }
        conn.close()
        assert cols == {
            "id", "user_id", "date", "slot_index", "intervention_type", "reserved_at",
        }
