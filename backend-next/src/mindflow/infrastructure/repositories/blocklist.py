"""SQLAlchemy-backed blocked-sites repository (environment_optimization).

Backs the execution side of the ``environment_optimization``
intervention: the intervention service records the distracting domain,
and the browser extension polls the enabled list and applies
declarativeNetRequest rules.

Table schema matches Alembic migration 0023:

  blocked_sites:
    id          TEXT PK (UUIDv7)
    user_id     INTEGER NOT NULL
    domain      TEXT NOT NULL (UNIQUE per user)
    enabled     BOOLEAN NOT NULL
    reason      TEXT (nullable — why the site was blocked)
    created_at  TEXT NOT NULL (ISO8601 UTC)
    updated_at  TEXT NOT NULL (ISO8601 UTC)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import sqlalchemy as sa
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.domain.ids import new_id
from mindflow.infrastructure.schema import blocked_sites as blocked_table


def _normalize_domain(domain: str) -> str:
    """Lower-case and strip ports/path from a hostname-ish string."""
    raw = domain.strip().lower()
    if not raw:
        return raw
    # Tolerate pasted URLs ("https://example.com/foo") and port suffixes.
    without_scheme = raw.split("://", 1)[-1]
    host = without_scheme.split("/", 1)[0].split(":", 1)[0]
    return host


class SQLAlchemyBlocklistRepository:
    """Blocked-sites persistence backed by SQLAlchemy Core + async SQLite.

    Args:
        session_factory: Async session maker bound to the application engine.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def list_enabled(self, user_id: int) -> list[str]:
        """Return the enabled domain list the extension should block."""
        stmt = (
            sa.select(blocked_table.c.domain)
            .where(
                blocked_table.c.user_id == user_id,
                blocked_table.c.enabled.is_(True),
            )
            .order_by(blocked_table.c.created_at)
        )
        async with self._session_factory() as session:
            rows = (await session.execute(stmt)).all()
        return [str(row.domain) for row in rows]

    async def list_all(self, user_id: int) -> list[dict[str, Any]]:
        """Return the full blocked-site rows for the management UI."""
        stmt = (
            sa.select(blocked_table)
            .where(blocked_table.c.user_id == user_id)
            .order_by(blocked_table.c.created_at)
        )
        async with self._session_factory() as session:
            rows = (await session.execute(stmt)).all()
        return [
            {
                "domain": row.domain,
                "enabled": bool(row.enabled),
                "reason": row.reason,
                "created_at": row.created_at,
            }
            for row in rows
        ]

    async def ensure_blocked(
        self,
        user_id: int,
        domain: str,
        *,
        reason: str | None = None,
    ) -> bool:
        """Block *domain*, re-enabling an existing disabled row.

        Returns True when a new row was inserted (vs. an existing row
        being re-enabled), so callers can log whether this was a fresh
        intervention execution.
        """
        normalized = _normalize_domain(domain)
        if not normalized:
            return False
        now = datetime.now(UTC).isoformat()
        async with self._session_factory() as session, session.begin():
            stmt = (
                sqlite_insert(blocked_table)
                .values(
                    id=new_id(),
                    user_id=user_id,
                    domain=normalized,
                    enabled=True,
                    reason=reason,
                    created_at=now,
                    updated_at=now,
                )
                .on_conflict_do_update(
                    index_elements=["user_id", "domain"],
                    set_={
                        "enabled": True,
                        "reason": reason,
                        "updated_at": now,
                    },
                )
            )
            result = await session.execute(stmt)
            # SQLite rowcount: 1 on insert, 1 on update (rowcount semantics
            # for upserts vary); we only report whether an upsert happened.
            return bool(cast(Any, result).rowcount)

    async def set_enabled(self, user_id: int, domain: str, enabled: bool) -> bool:
        """Toggle one blocked site; returns True when a row changed."""
        normalized = _normalize_domain(domain)
        if not normalized:
            return False
        async with self._session_factory() as session, session.begin():
            result = await session.execute(
                sa.update(blocked_table)
                .where(
                    blocked_table.c.user_id == user_id,
                    blocked_table.c.domain == normalized,
                )
                .values(enabled=enabled, updated_at=datetime.now(UTC).isoformat())
            )
            return bool(cast(Any, result).rowcount)

    async def remove(self, user_id: int, domain: str) -> bool:
        """Permanently delete a blocked-site row; True when a row existed."""
        normalized = _normalize_domain(domain)
        if not normalized:
            return False
        async with self._session_factory() as session, session.begin():
            result = await session.execute(
                sa.delete(blocked_table).where(
                    blocked_table.c.user_id == user_id,
                    blocked_table.c.domain == normalized,
                )
            )
            return bool(cast(Any, result).rowcount)
