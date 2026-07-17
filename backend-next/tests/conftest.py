"""Test fixtures for MindFlow Wave 1 infrastructure tests.

Uses tmp_path for database isolation and overridable settings.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from mindflow.config import Settings
from mindflow.infrastructure.database import create_engine, create_session_factory


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """Return a temporary database file path.

    Each test gets an isolated path so tests do not share state.
    """
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
async def session_factory(engine):
    """Create an async_sessionmaker bound to the test engine."""
    return create_session_factory(engine)


@pytest.fixture
def settings_factory(tmp_path: Path):
    """Factory fixture to create Settings with overridden data_dir.

    Usage:
        settings = settings_factory(db_url="sqlite+aiosqlite:///other.db")
    """

    def _make(**kwargs) -> Settings:
        overrides = dict(kwargs)
        return Settings(**overrides)

    return _make
