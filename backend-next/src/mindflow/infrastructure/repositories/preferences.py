"""PreferencesRepository for user preferences (JSON key-value store).

Uses the ``user_preferences`` table defined in migration 0001.
The repository reads and writes the ``preferences_json`` column as a
JSON blob — no ORM mapping, just SQLAlchemy Core queries.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
import uuid6
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.infrastructure.schema import user_preferences


class PreferencesRepository:
    """JSON key-value preference store backed by the ``user_preferences`` table.

    Args:
        session_factory: Async session maker bound to the application engine.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._session_factory = session_factory

    async def get(self, user_id: int) -> dict[str, Any]:
        """Return the preferences dict for *user_id*, or an empty dict.

        Args:
            user_id: User identifier.

        Returns:
            Parsed JSON preferences dict, or ``{}`` if none exist.
        """
        stmt = sa.select(user_preferences).where(
            user_preferences.c.user_id == user_id
        )

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            row = result.fetchone()

        if row is None:
            return {}

        return json.loads(row.preferences_json) if row.preferences_json else {}

    async def set(self, user_id: int, preferences: dict[str, Any]) -> None:
        """Set the full preferences dict for *user_id*.

        Uses UPSERT (INSERT OR REPLACE) via SQLAlchemy Core.
        If the user already has preferences, replaces them atomically.

        Args:
            user_id: User identifier.
            preferences: Arbitrary JSON-serializable dict.
        """
        now = datetime.now(UTC).isoformat()
        raw = json.dumps(preferences, ensure_ascii=False)

        async with self._session_factory() as session, session.begin():
            updated_id = await session.scalar(
                sa.update(user_preferences)
                .where(user_preferences.c.user_id == user_id)
                .values(preferences_json=raw, updated_at=now)
                .returning(user_preferences.c.id)
            )
            if updated_id is None:
                await session.execute(
                    user_preferences.insert().values(
                        id=str(uuid6.uuid7()),
                        user_id=user_id,
                        preferences_json=raw,
                        updated_at=now,
                    )
                )
