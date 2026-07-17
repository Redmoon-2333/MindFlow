"""Database engine factory and connection management.

Creates SQLAlchemy AsyncEngine with SQLite WAL-mode PRAGMAs.
Not a module-level singleton — engines are created explicitly via factory functions.

WAL mode configuration (NF-P4):
  - journal_mode=WAL: concurrent reads + one writer
  - synchronous=NORMAL: balance of safety and performance
  - busy_timeout=5000: wait up to 5s instead of immediate SQLITE_BUSY
  - journal_size_limit=67108864: cap WAL file at 64 MB

Migration-failure resiliency (NF-R5):
  - integrity_check() runs at startup; on failure, logs and continues
  - backup_database() uses VACUUM INTO for crash-consistent snapshots
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def _set_wal_pragmas(dbapi_connection: Any, _connection_record: Any) -> None:
    """Configure SQLite WAL-mode PRAGMAs on new connection (NF-P4).

    Applied as an event listener on the engine so every new connection
    automatically gets these settings without callers needing to remember.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.execute("PRAGMA journal_size_limit=67108864")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def create_engine(db_url: str, **kwargs: Any) -> AsyncEngine:
    """Create a configured SQLAlchemy AsyncEngine for SQLite WAL.

    Args:
        db_url: SQLAlchemy async database URL (e.g. sqlite+aiosqlite:///path/to/db).
        **kwargs: Additional keyword args forwarded to create_async_engine.

    Returns:
        Configured AsyncEngine with WAL PRAGMA listeners attached.

    Warning:
        engine.dispose() must be called during application shutdown
        to ensure all connections are cleanly closed.
    """
    engine = create_async_engine(
        db_url,
        echo=False,
        connect_args={"check_same_thread": False} if "sqlite" in db_url else {},
        **kwargs,
    )

    if "sqlite" in db_url:
        event.listen(engine.sync_engine, "connect", _set_wal_pragmas)

    return engine


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create a configured async_sessionmaker for the given engine.

    Args:
        engine: Configured AsyncEngine instance.

    Returns:
        async_sessionmaker bound to the engine with sane defaults.
    """
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


async def integrity_check(engine: AsyncEngine) -> bool:
    """Run PRAGMA integrity_check on the database.

    Returns True if check passes (returns "ok"), False otherwise.
    Does NOT raise — logs warning on failure for graceful degradation (NF-R5).
    """
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("PRAGMA integrity_check"))
            row = result.fetchone()
            if row and row[0] == "ok":
                return True
            logger.warning("Database integrity check failed: {}", row)
            return False
    except Exception as exc:
        logger.error("Database integrity check raised exception: {}", exc)
        return False


async def backup_database(engine: AsyncEngine, dest: Path) -> bool:
    """Create a crash-consistent backup via VACUUM INTO.

    VACUUM INTO creates a new database file at *dest* that is
    transactionally consistent at the point of execution.
    This is the recommended SQLite backup method (3.27+).

    Args:
        engine: Source database engine.
        dest: Destination file path for the backup.

    Returns:
        True if backup succeeded, False otherwise.
    """
    try:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # VACUUM INTO must run via raw connection (sync pragma in async wrapper)
        async with engine.connect() as conn:
            await conn.execute(text(f"VACUUM INTO '{dest}'"))
            await conn.commit()
        logger.info("Database backed up to {}", dest)
        return True
    except Exception as exc:
        logger.error("Database backup failed: {}", exc)
        return False


def text(sql: str) -> Any:
    """Shorthand for SQLAlchemy text() — avoids top-level import."""
    from sqlalchemy import text as _text

    return _text(sql)
