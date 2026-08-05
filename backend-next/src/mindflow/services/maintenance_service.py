"""Maintenance service: event cleanup, workflow retention, and database backup.

Implements Wave 5 data-retention and backup policies, plus Wave 18 maintenance policies:
  - Raw activity events beyond *retention_days* are deleted in batches
    (10 000 rows per batch, with per-batch commit) to avoid long-running
    transactions and WAL file bloat.
  - Workflow/checkpoint/event cleanup: remove completed/failed/cancelled runs
    older than N days, preserving analyses and chat messages.
  - Stale-run reconciliation: mark runs stuck in "running" for >1h as "failed".
  - Orphan chat-turn reconciliation: detect user messages without assistant
    responses — logged, not deleted.
  - Budget expiry: release budget reservations past their expiry time.
  - Daily backup via ``VACUUM INTO`` creates a crash-consistent snapshot.
  - Backup failures are logged and sent as desktop notifications.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import platformdirs
import sqlalchemy as sa
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from mindflow.infrastructure.database import backup_database
from mindflow.infrastructure.notification import NotificationService
from mindflow.infrastructure.repositories.activity import activity_events
from mindflow.infrastructure.repositories.preferences import PreferencesRepository
from mindflow.infrastructure.schema import (
    chat_messages,
    intervention_checks,
    workflow_budget_reservations,
    workflow_node_events,
    workflow_runs,
)

_BATCH_SIZE: int = 10_000
"""Maximum rows deleted in a single DELETE + COMMIT cycle."""


class MaintenanceService:
    """Periodic maintenance operations for data retention and backup.

    Args:
        engine: SQLAlchemy AsyncEngine for direct table operations.
        session_factory: Session factory for transactional operations.
        notifier: Notification service for alerting on failures.
        data_dir: Optional data directory override.  Defaults to
            ``platformdirs.user_data_dir("mindflow")``.
        clock: Optional callable returning the current UTC datetime.
            Used for testability (inject a fixed clock).  Defaults to
            ``lambda: datetime.now(UTC)``.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        session_factory: async_sessionmaker[AsyncSession],
        notifier: NotificationService,
        data_dir: Path | None = None,
        clock: Callable[[], datetime] | None = None,
        preferences_repository: PreferencesRepository | None = None,
    ) -> None:
        self._engine = engine
        self._session_factory = session_factory
        self._notifier = notifier
        self._data_dir = data_dir or Path(
            platformdirs.user_data_dir("mindflow", ensure_exists=True)
        )
        self._now = clock or (lambda: datetime.now(UTC))
        # When wired, the user preference ``activity_retention_days`` is the
        # single effective activity-retention value; the ``retention_days``
        # argument (the env ``event_retention_days`` startup default) is only
        # the fallback when no preference is stored.
        self._preferences_repository = preferences_repository

    async def _activity_retention_days(self, fallback: int) -> int:
        """Resolve the effective activity-retention days.

        The stored preference ``activity_retention_days`` is authoritative;
        *fallback* (the env ``event_retention_days`` startup default) applies
        only when the preference is missing or no preference repository is
        wired.
        """
        if self._preferences_repository is None:
            return fallback
        preferences = await self._preferences_repository.get(1)
        value = preferences.get("telemetry", {}).get("activity_retention_days")
        if value is None:
            return fallback
        return int(value)

    # ── Event cleanup ────────────────────────────────────────────────

    async def cleanup_old_events(self, retention_days: int = 30) -> int:
        """Delete activity events older than the effective retention horizon.

        The effective horizon is the user preference
        ``activity_retention_days`` when present; otherwise *retention_days*
        (the env ``event_retention_days`` startup default) applies.

        Each batch deletes up to ``_BATCH_SIZE`` (10 000) rows and commits
        immediately, preventing long-running transactions.

        Args:
            retention_days: Env startup default for events older than this many
                days are removed. Must be >= 7 (validated at the config level).

        Returns:
            Total number of rows deleted.
        """
        effective = await self._activity_retention_days(retention_days)
        cutoff = (self._now() - timedelta(days=effective)).isoformat()
        total_deleted = 0

        while True:
            candidate_ids = (
                sa.select(activity_events.c.id)
                .where(activity_events.c.timestamp < cutoff)
                .order_by(activity_events.c.timestamp.asc(), activity_events.c.id.asc())
                .limit(_BATCH_SIZE)
            )
            delete_stmt = sa.delete(activity_events).where(
                activity_events.c.id.in_(candidate_ids)
            )

            async with self._session_factory() as session, session.begin():
                result = await session.scalars(
                    delete_stmt.returning(activity_events.c.id)
                )
                deleted = len(result.all())

            if deleted == 0:
                break
            total_deleted += deleted
            logger.debug(
                "Cleanup batch: deleted {} events (total {})",
                deleted,
                total_deleted,
            )

        if total_deleted > 0:
            logger.info(
                "Event cleanup complete: deleted {} events older than {} days",
                total_deleted,
                effective,
            )
        else:
            logger.debug("Event cleanup: no events to delete")

        await self._wal_checkpoint_truncate()

        return total_deleted

    async def _wal_checkpoint_truncate(self) -> None:
        """Run ``PRAGMA wal_checkpoint(TRUNCATE)`` to zero the WAL file.

        Called after ``cleanup_old_events`` to reclaim disk space freed by
        deleted rows.  Must run outside any active write transaction so that
        the TRUNCATE pass actually zeros the WAL file header.

        Raises on checkpoint failure — the scheduled caller (scheduler
        ``_run_daily_cron``) catches and logs exceptions, so this matches
        the existing error contract of ``cleanup_old_events`` itself.
        """
        async with self._engine.connect() as conn:
            await conn.execute(sa.text("PRAGMA wal_checkpoint(TRUNCATE)"))
            await conn.commit()

    # ── Intervention-check cleanup ─────────────────────────────────────

    async def cleanup_old_intervention_checks(self, retention_days: int = 30) -> int:
        """Delete ``intervention_checks`` older than the activity horizon.

        Rows are matched by ``checked_at`` against the same effective
        activity-retention cutoff used for raw ``activity_events``: the user
        preference ``activity_retention_days`` wins, otherwise the
        *retention_days* argument (env ``event_retention_days``) applies.
        A check exactly at the cutoff is preserved (strict ``<``).

        Args:
            retention_days: Env startup default retention in days.

        Returns:
            Number of intervention checks deleted.
        """
        effective = await self._activity_retention_days(retention_days)
        cutoff = (self._now() - timedelta(days=effective)).isoformat()

        async with self._session_factory() as session, session.begin():
            result = await session.execute(
                sa.delete(intervention_checks).where(
                    intervention_checks.c.checked_at < cutoff
                )
            )
            deleted = result.rowcount or 0

        if deleted > 0:
            logger.info(
                "Intervention-check cleanup: deleted {} checks older than {} days",
                deleted,
                effective,
            )
        else:
            logger.debug("Intervention-check cleanup: no checks to delete")

        return deleted

    # ── Daily backup ─────────────────────────────────────────────────

    async def run_daily_backup(self) -> bool:
        """Create a crash-consistent database backup.

        Backup is saved to ``{data_dir}/backups/mindflow-{date}.db``.

        On failure, the error is logged and a desktop notification is sent
        via the configured ``NotificationService``.

        Returns:
            True if the backup succeeded, False otherwise.
        """
        backup_dir = self._data_dir / "backups"
        today_str = self._now().strftime("%Y-%m-%d")
        dest = backup_dir / f"mindflow-{today_str}.db"

        success = await backup_database(self._engine, dest)

        if success:
            logger.info("Daily backup completed: {}", dest)
        else:
            logger.error("Daily backup FAILED: {}", dest)
            try:
                await self._notifier.send(
                    title="MindFlow 备份失败",
                    body=f"数据库备份到 {dest} 失败，请检查磁盘空间和数据库状态",
                    urgency="critical",
                )
            except Exception:
                logger.warning("Failed to send backup failure notification")

        return success

    # ── Workflow / checkpoint / event cleanup ─────────────────────────

    async def cleanup_old_workflows(self, retention_days: int = 30) -> int:
        """Delete completed/failed/cancelled workflow runs older than
        *retention_days*, plus their node events.

        **Does NOT touch** analyses (``procrastination_analyses``) or
        chat messages (``chat_messages``) — those tables live in separate
        namespaces and are preserved.

        Active and retryable runs (``pending``, ``running``) are NEVER
        deleted regardless of age.

        Args:
            retention_days: Runs older than this many days are removed.
                Must be >= 7.

        Returns:
            Total number of workflow runs deleted.
        """
        cutoff = (self._now() - timedelta(days=retention_days)).isoformat()
        terminal_statuses = ("completed", "failed", "cancelled")

        # ── Select candidate run IDs ─────────────────────────────────
        candidate_ids = sa.select(workflow_runs.c.run_id).where(
            sa.and_(
                workflow_runs.c.status.in_(terminal_statuses),
                workflow_runs.c.updated_at < cutoff,
            )
        )

        # ── Delete their node events first (no FK, logical cascade) ──
        async with self._session_factory() as session, session.begin():
            await session.execute(
                sa.delete(workflow_node_events).where(
                    workflow_node_events.c.run_id.in_(candidate_ids)
                )
            )
            await session.commit()

        # ── Then delete the workflow runs ─────────────────────────────
        total_deleted = 0
        while True:
            async with self._session_factory() as session, session.begin():
                batch = sa.select(workflow_runs.c.run_id).where(
                    sa.and_(
                        workflow_runs.c.status.in_(terminal_statuses),
                        workflow_runs.c.updated_at < cutoff,
                    )
                ).limit(_BATCH_SIZE)
                result = await session.execute(
                    sa.delete(workflow_runs).where(
                        workflow_runs.c.run_id.in_(batch)
                    )
                )
                deleted = result.rowcount
                if deleted == 0:
                    break
                total_deleted += deleted

        if total_deleted > 0:
            logger.info(
                "Workflow cleanup: deleted {} runs older than {} days",
                total_deleted,
                retention_days,
            )
        else:
            logger.debug("Workflow cleanup: no runs to delete")

        return total_deleted

    # ── Stale-run reconciliation ─────────────────────────────────────

    async def reconcile_stale_runs(self, timeout_minutes: int = 60) -> int:
        """Mark workflow runs stuck in ``"running"`` for longer than
        *timeout_minutes* as ``"failed"``.

        A run is stale when ``status='running'`` AND ``updated_at`` is
        older than ``now - timeout_minutes``.  This handles crashed
        processes that never called ``update_status("failed")``.

        Args:
            timeout_minutes: Staleness threshold in minutes (default 60).

        Returns:
            Number of runs marked as failed.
        """
        cutoff = (self._now() - timedelta(minutes=timeout_minutes)).isoformat()
        now_iso = self._now().isoformat()

        async with self._session_factory() as session, session.begin():
            result = await session.execute(
                sa.update(workflow_runs)
                .where(
                    sa.and_(
                        workflow_runs.c.status == "running",
                        workflow_runs.c.updated_at < cutoff,
                    )
                )
                .values(
                    status="failed",
                    updated_at=now_iso,
                    completed_at=now_iso,
                    retry_reason="Stale run: no update within timeout",
                )
            )
            stale_count = result.rowcount

        if stale_count > 0:
            logger.warning(
                "Stale-run reconciliation: marked {} running runs as failed "
                "(timeout={} min)",
                stale_count,
                timeout_minutes,
            )
        else:
            logger.debug("Stale-run reconciliation: no stale runs found")

        return stale_count

    # ── Orphan chat-turn reconciliation ──────────────────────────────

    async def reconcile_orphan_chat_turns(self) -> int:
        """Detect chat sessions where a user message has no matching
        assistant response — these are orphaned turns.

        An orphaned turn is a ``user`` message that is the last message
        in its session (i.e. no ``assistant`` response follows it).
        These are counted and logged — NOT deleted.

        Returns:
            Number of orphaned turns detected.
        """
        # ── Per-session: find user messages that are the session's last ──
        # Subquery: last message per session (max created_at)
        last_msg = (
            sa.select(
                chat_messages.c.session_id,
                sa.func.max(chat_messages.c.created_at).label("last_at"),
            )
            .group_by(chat_messages.c.session_id)
            .subquery("last_msg")
        )

        # Join to get the role of the last message per session
        orphaned = (
            sa.select(sa.func.count())
            .select_from(
                sa.join(
                    chat_messages,
                    last_msg,
                    sa.and_(
                        chat_messages.c.session_id == last_msg.c.session_id,
                        chat_messages.c.created_at == last_msg.c.last_at,
                    ),
                )
            )
            .where(chat_messages.c.role == "user")
        )

        async with self._session_factory() as session:
            result = await session.execute(orphaned)
            count: int = result.scalar() or 0

        if count > 0:
            logger.info(
                "Orphan chat-turn reconciliation: {} user messages "
                "without assistant responses detected",
                count,
            )
        else:
            logger.debug("Orphan chat-turn reconciliation: no orphaned turns")

        return count

    # ── Budget expiry ────────────────────────────────────────────────

    async def expire_stale_budgets(self) -> int:
        """Release budget reservations whose expiry time has passed.

        A reservation is stale when ``expires_at IS NOT NULL`` AND
        ``expires_at < now()`` AND ``released_at IS NULL`` (not already
        released).  Stale reservations are DELETED so the idempotency
        key becomes available for future claims.

        Returns:
            Number of budget reservations expired.
        """
        now_iso = self._now().isoformat()

        async with self._session_factory() as session, session.begin():
            result = await session.execute(
                sa.delete(workflow_budget_reservations).where(
                    sa.and_(
                        workflow_budget_reservations.c.expires_at.is_not(None),
                        workflow_budget_reservations.c.expires_at < now_iso,
                        workflow_budget_reservations.c.released_at.is_(None),
                    )
                )
            )
            expired = result.rowcount

        if expired > 0:
            logger.info(
                "Budget expiry: released {} stale reservations",
                expired,
            )
        else:
            logger.debug("Budget expiry: no stale reservations")

        return expired

    def __repr__(self) -> str:
        return f"<MaintenanceService data_dir={self._data_dir}>"
