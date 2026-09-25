"""TaskContextRulesRepository for the task-context mapping layer.

Uses the ``task_context_rules`` table (migration 0027).  Each row maps either an
*activity category* (the vocabulary produced by ``app_classification_rules`` /
``AppClassifier``) or a browser *domain* to one observed task context
(``coding`` / ``writing`` / ``reading`` / ``meeting`` / ``entertainment`` /
``social`` / ``unknown``).  The rollup reads these rules and turns process and
domain time into the v4 task-context feature columns.

Domain values are reduced to a bare host on write
(:func:`mindflow.domain.task_context.normalize_domain_rule`), so a pasted URL
can never store a private path or query — the privacy boundary stays
"domain only".  Reads mirror ``AppClassificationRulesRepository``: priority
DESC, then creation order.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
import uuid6
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.domain.task_context import (
    CLASSIFICATION_CATEGORIES,
    MATCH_TYPE_CATEGORY,
    MATCH_TYPE_DOMAIN,
    MATCH_TYPES,
    TASK_CONTEXT_CATEGORIES,
    normalize_domain_rule,
)
from mindflow.infrastructure.schema import task_context_rules


class TaskContextRulesRepository:
    """Classification-to-task-context rules backed by ``task_context_rules``.

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
        """Return all rules for *user_id*, highest priority first."""
        stmt = (
            sa.select(task_context_rules)
            .where(task_context_rules.c.user_id == user_id)
            .order_by(
                task_context_rules.c.priority.desc(),
                task_context_rules.c.created_at.asc(),
            )
        )
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            rows = result.fetchall()
        return [_row_to_dict(row) for row in rows]

    # ── Commands ───────────────────────────────────────────────────────

    async def add(self, user_id: int, rule: dict[str, Any]) -> dict[str, Any]:
        """Insert one mapping rule and return the stored values.

        Raises:
            ValueError: when ``match_type``/``match_value``/``task_context`` is
                not part of the contract, or when a domain value has no usable
                host.
        """
        values = _build_rule_values(user_id, rule, datetime.now(UTC).isoformat())
        async with self._session_factory() as session, session.begin():
            await session.execute(task_context_rules.insert().values(**values))
        return values

    async def replace_all(
        self,
        user_id: int,
        rules: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Atomically replace all rules for a user with one bulk insert."""
        now = datetime.now(UTC).isoformat()
        values = [_build_rule_values(user_id, rule, now) for rule in rules]
        async with self._session_factory() as session, session.begin():
            await session.execute(
                sa.delete(task_context_rules).where(
                    task_context_rules.c.user_id == user_id
                )
            )
            if values:
                await session.execute(task_context_rules.insert(), values)
        return values

    async def delete(self, rule_id: str) -> None:
        """Delete a rule by id; a missing id is a no-op."""
        async with self._session_factory() as session, session.begin():
            await session.execute(
                sa.delete(task_context_rules).where(
                    task_context_rules.c.id == rule_id
                )
            )


# ── Helpers ─────────────────────────────────────────────────────────────


def _build_rule_values(
    user_id: int,
    rule: dict[str, Any],
    now: str,
) -> dict[str, Any]:
    match_type = str(rule.get("match_type", "")).strip().lower()
    if match_type not in MATCH_TYPES:
        msg = f"match_type must be one of {sorted(MATCH_TYPES)}"
        raise ValueError(msg)

    raw_value = rule.get("match_value")
    if match_type == MATCH_TYPE_DOMAIN:
        match_value = normalize_domain_rule(raw_value)
        if not match_value:
            msg = "domain match_value must contain a host, e.g. 'docs.python.org'"
            raise ValueError(msg)
    else:
        match_value = str(raw_value or "").strip().lower()
        if match_value not in CLASSIFICATION_CATEGORIES:
            msg = f"category match_value must be one of {sorted(CLASSIFICATION_CATEGORIES)}"
            raise ValueError(msg)

    task_context = str(rule.get("task_context", "")).strip().lower()
    if task_context not in TASK_CONTEXT_CATEGORIES:
        msg = f"task_context must be one of {sorted(TASK_CONTEXT_CATEGORIES)}"
        raise ValueError(msg)

    try:
        priority = int(rule.get("priority", 0) or 0)
    except (TypeError, ValueError):
        priority = 0

    return {
        "id": str(uuid6.uuid7()),
        "user_id": user_id,
        "match_type": match_type,
        "match_value": match_value,
        "task_context": task_context,
        "priority": priority,
        "created_at": now,
        "updated_at": now,
    }


def _row_to_dict(row: sa.Row[Any]) -> dict[str, Any]:
    """Convert a ``task_context_rules`` row to a plain dict."""
    return {
        "id": row.id,
        "user_id": row.user_id,
        "match_type": row.match_type,
        "match_value": row.match_value,
        "task_context": row.task_context,
        "priority": row.priority,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


# Re-exported so the module and the API layer share one vocabulary.
VALID_MATCH_TYPE_CATEGORY = MATCH_TYPE_CATEGORY
VALID_MATCH_TYPE_DOMAIN = MATCH_TYPE_DOMAIN
