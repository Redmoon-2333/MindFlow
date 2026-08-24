"""Test fixtures for MindFlow Wave 1-4 infrastructure and API tests.

Uses tmp_path for database isolation and overridable settings.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.config import Settings
from mindflow.infrastructure.database import create_engine, create_session_factory
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
    activity_events,
)
from mindflow.infrastructure.repositories.preferences import (
    PreferencesRepository,
)
from mindflow.infrastructure.schema import metadata as schema_metadata


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """Return a temporary database file path."""
    return tmp_path / "test_mindflow.db"


@pytest.fixture
def db_url(tmp_db_path: Path) -> str:
    """Return a SQLite async URL pointing to the temporary database."""
    return f"sqlite+aiosqlite:///{tmp_db_path}"


@pytest.fixture
async def engine(db_url: str) -> AsyncIterator:
    """Create an engine for a temporary database, disposing after the test."""
    engine = create_engine(db_url)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
async def session_factory(engine) -> async_sessionmaker[AsyncSession]:
    """Create an async_sessionmaker bound to the test engine."""
    return create_session_factory(engine)


@pytest.fixture
def settings_factory(tmp_path: Path):
    """Factory fixture to create Settings with overridden values."""
    def _make(**kwargs) -> Settings:
        return Settings(**kwargs)
    return _make


@pytest.fixture
async def create_tables(engine):
    """Create all core tables that tests depend on.

    Uses the full schema metadata (activity_events, preferences, telemetry,
    intervention, workflow tables, ...) so new tables are covered without
    editing this fixture (audit report — conftest only created two tables).
    """
    async with engine.begin() as conn:
        await conn.run_sync(schema_metadata.create_all)
        # activity_events lives in its own module-level metadata (kept in sync
        # with the schema module); create it too.
        await conn.run_sync(activity_events.metadata.create_all)


@pytest.fixture
async def activity_repo(session_factory, create_tables) -> SQLAlchemyActivityRepository:
    """Create an ActivityRepository with tables created."""
    return SQLAlchemyActivityRepository(session_factory=session_factory)


@pytest.fixture
async def preferences_repo(session_factory, create_tables) -> PreferencesRepository:
    """Create a PreferencesRepository with tables created."""
    return PreferencesRepository(session_factory=session_factory)


@pytest.fixture
def anyio_backend() -> str:
    """Use asyncio as the anyio backend for FastAPI test client."""
    return "asyncio"


@pytest.fixture(autouse=True)
def isolate_file_logging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Keep setup_logging() file sinks out of the production log directory.

    ``create_app()`` calls ``setup_logging()``, which points the process-global
    loguru at the shared user-data log dir. Without this fixture, running the
    test suite writes fault-injection noise (and test-only analysis runs)
    into the real application logs on this machine. Redirect the resolver to
    *tmp_path* and remove any file sinks added during the test afterwards.
    """
    from loguru import logger

    import mindflow.logging_config as logging_config

    monkeypatch.setattr(logging_config, "_resolve_log_dir", lambda: tmp_path / "logs")
    handlers_before = set(logger._core.handlers)  # noqa: SLF001 - test teardown only
    yield
    for handler_id in set(logger._core.handlers) - handlers_before:
        with contextlib.suppress(ValueError):
            logger.remove(handler_id)


@pytest.fixture(autouse=True)
def reset_scheduler_global_state() -> None:
    """Reset module-level scheduler claim state before/after every test.

    ``_DAILY_PANEL_RUN_DATES`` is a module-level set used as the fallback
    claim store when no persistent repository is injected. Tests mutate it
    directly (and some only clean up on success), which made the suite
    order-dependent (audit report — module-level global state). Resetting
    in an autouse fixture keeps every test hermetic regardless of what a
    previous test did or failed to do.
    """
    import mindflow.services.scheduler as scheduler_module

    scheduler_module._DAILY_PANEL_RUN_DATES.clear()
    yield
    scheduler_module._DAILY_PANEL_RUN_DATES.clear()
