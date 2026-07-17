"""Test fixtures for MindFlow Wave 1-4 infrastructure and API tests.

Uses tmp_path for database isolation and overridable settings.
"""

from __future__ import annotations

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
    user_preferences,
)


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
    """Create all core tables that tests depend on."""
    async with engine.begin() as conn:
        await conn.run_sync(activity_events.metadata.create_all)
        await conn.run_sync(user_preferences.metadata.create_all)


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
