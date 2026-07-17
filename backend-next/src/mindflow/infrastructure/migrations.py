"""Database migration runner.

Wraps Alembic's synchronous API inside asyncio.to_thread so it does
not block the async event loop (NF-R5, §5.3 of architecture doc).

On failure, returns False so the application can start with the
existing schema and expose migration_failed status on the health
endpoint (graceful degradation per M9/NF-R5 resolution).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


def _run_migrations_sync(db_url: str) -> None:
    """Synchronous Alembic migration runner.

    Uses a separate alembic.Config with the synchronous SQLite URL
    (sqlite:// instead of sqlite+aiosqlite://).

    Args:
        db_url: Synchronous SQLite URL for Alembic's sync engine.
    """
    from alembic.config import Config

    from alembic import command

    cfg = Config(str(BASE_DIR / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", db_url)
    command.upgrade(cfg, "head")


async def run_migrations(async_db_url: str) -> bool:
    """Run Alembic migrations to the latest revision.

    Converts the async URL to a sync URL for Alembic, then runs
    migration in a thread to avoid blocking the event loop.

    Args:
        async_db_url: Async SQLAlchemy URL (e.g. sqlite+aiosqlite:///path).

    Returns:
        True if migrations succeeded, False on failure (graceful degradation).
    """
    from sqlalchemy.engine import make_url

    # Map async drivers to their sync counterparts so future PostgreSQL
    # support (NF-A5) doesn't silently pass an async URL to Alembic.
    url = make_url(async_db_url)
    sync_drivers = {"sqlite+aiosqlite": "sqlite", "postgresql+asyncpg": "postgresql+psycopg"}
    sync_url = url.set(
        drivername=sync_drivers.get(url.drivername, url.drivername)
    ).render_as_string(hide_password=False)
    try:
        await asyncio.to_thread(_run_migrations_sync, sync_url)
        logger.info("Database migrations applied successfully")
        return True
    except Exception as exc:
        logger.critical("Database migration failed (NF-R5 degradation): {}", exc)
        return False
