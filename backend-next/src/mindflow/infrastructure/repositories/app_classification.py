"""AppClassificationRulesRepository for user app classification rules.

Uses the ``app_classification_rules`` table defined in migration 0006.
The repository manages per-process-name and per-window-title classification
rules that determine how MindFlow categorises activity.  All queries use
SQLAlchemy Core — no ORM mapping.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
import uuid6
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

# Known categories used by the classifier (same set as AppClassifier).
_KNOWN_CATEGORIES: frozenset[str] = frozenset({
    "code",
    "document",
    "browser_work",
    "communication",
    "entertainment",
    "social",
    "other",
})

# ── Table definition (matches migration 0006_create_app_classification_rules) ─

app_classification_rules = sa.Table(
    "app_classification_rules",
    sa.MetaData(),
    sa.Column("id", sa.Text(), primary_key=True),
    sa.Column("user_id", sa.Integer(), nullable=False),
    sa.Column("process_name", sa.Text(), nullable=False),
    sa.Column("window_title_pattern", sa.Text(), nullable=True),
    sa.Column("category", sa.Text(), nullable=False),
    sa.Column("priority", sa.Integer(), nullable=False, server_default=sa.text("0")),
    sa.Column("created_at", sa.Text(), nullable=False),
    sa.Column("updated_at", sa.Text(), nullable=False),
)


class AppClassificationRulesRepository:
    """Classification rules store backed by ``app_classification_rules`` table.

    Args:
        session_factory: Async session maker bound to the application engine.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._session_factory = session_factory

    # ── Queries ────────────────────────────────────────────────────────

    async def get_all(self, user_id: int) -> list[dict[str, Any]]:
        """Return all rules for *user_id*, ordered by priority DESC, then created_at ASC.

        Args:
            user_id: User identifier.

        Returns:
            List of rule dicts, or empty list if no rules exist.
        """
        stmt = (
            sa.select(app_classification_rules)
            .where(app_classification_rules.c.user_id == user_id)
            .order_by(
                app_classification_rules.c.priority.desc(),
                app_classification_rules.c.created_at.asc(),
            )
        )

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            rows = result.fetchall()

        return [_row_to_dict(row) for row in rows]

    # ── Commands ───────────────────────────────────────────────────────

    async def add(self, user_id: int, rule: dict[str, Any]) -> dict[str, Any]:
        """Insert a new classification rule and return it with generated fields.

        Args:
            user_id: User identifier.
            rule: Dict with keys:
                - ``process_name`` (required): e.g. ``"notion.exe"``
                - ``window_title_pattern`` (optional): SQL LIKE pattern
                - ``category`` (required): one of the known categories
                - ``priority`` (optional, default 0): higher = checked first

        Returns:
            The complete rule dict including generated ``id``, ``created_at``,
            and ``updated_at``.
        """
        now = datetime.now(UTC).isoformat()
        rule_id = str(uuid6.uuid7())

        values: dict[str, Any] = {
            "id": rule_id,
            "user_id": user_id,
            "process_name": rule["process_name"],
            "window_title_pattern": rule.get("window_title_pattern"),
            "category": rule["category"],
            "priority": rule.get("priority", 0),
            "created_at": now,
            "updated_at": now,
        }

        async with self._session_factory() as session, session.begin():
            await session.execute(
                app_classification_rules.insert().values(**values)
            )

        return values

    async def delete(self, rule_id: str) -> None:
        """Delete a classification rule by id.

        If *rule_id* does not exist, this is a no-op (no error raised).

        Args:
            rule_id: UUIDv7 string identifying the rule.
        """
        stmt = sa.delete(app_classification_rules).where(
            app_classification_rules.c.id == rule_id
        )

        async with self._session_factory() as session, session.begin():
            await session.execute(stmt)

    # ── Analytics ──────────────────────────────────────────────────────

    async def get_unknown_apps(
        self, user_id: int, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Return apps not yet classified by any rule with a known category.

        Queries ``activity_events`` for process names that do not appear in
        ``app_classification_rules``, grouped and ordered by occurrence count
        descending.

        Args:
            user_id: User identifier.
            limit: Maximum number of unknown apps to return (default 20).

        Returns:
            List of dicts with keys: ``process_name``, ``count``, ``last_seen``
            (ISO8601 text).
        """
        # Late import to keep module-level imports clean; activity_events is
        # defined as a module-level sa.Table in that module.
        from mindflow.infrastructure.repositories.activity import (
            activity_events,
        )

        # Subquery: process_names already classified by any known-category rule.
        classified_subq = (
            sa.select(app_classification_rules.c.process_name)
            .where(
                app_classification_rules.c.user_id == user_id,
                app_classification_rules.c.category.in_(_KNOWN_CATEGORIES),
            )
        ).subquery()

        process_name_expr = sa.func.json_extract(
            activity_events.c.data_json, "$.process_name"
        ).label("process_name")

        stmt = (
            sa.select(
                process_name_expr,
                sa.func.count().label("count"),
                sa.func.max(activity_events.c.timestamp).label("last_seen"),
            )
            .where(
                activity_events.c.user_id == user_id,
                activity_events.c.data_json.isnot(None),
                sa.func.json_extract(
                    activity_events.c.data_json, "$.process_name"
                ).not_in(sa.select(classified_subq.c.process_name)),
            )
            .group_by(sa.literal_column("process_name"))
            .order_by(sa.literal_column("count").desc())
            .limit(limit)
        )

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            rows = result.fetchall()

        return [
            {
                "process_name": row.process_name,
                "count": row.count,
                "last_seen": row.last_seen,
            }
            for row in rows
        ]


# ── Helpers ─────────────────────────────────────────────────────────────


def _row_to_dict(row: sa.Row[Any]) -> dict[str, Any]:
    """Convert an ``app_classification_rules`` row to a plain dict."""
    return {
        "id": row.id,
        "user_id": row.user_id,
        "process_name": row.process_name,
        "window_title_pattern": row.window_title_pattern,
        "category": row.category,
        "priority": row.priority,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }
