"""Maintenance service: event cleanup and database backup.

Implements Wave 5 data-retention and backup policies:
  - Raw activity events beyond *retention_days* are deleted in batches
    (10 000 rows per batch, with per-batch commit) to avoid long-running
    transactions and WAL file bloat.
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
    ) -> None:
        self._engine = engine
        self._session_factory = session_factory
        self._notifier = notifier
        self._data_dir = data_dir or Path(
            platformdirs.user_data_dir("mindflow", ensure_exists=True)
        )
        self._now = clock or (lambda: datetime.now(UTC))

    # ── Event cleanup ────────────────────────────────────────────────

    async def cleanup_old_events(self, retention_days: int = 30) -> int:
        """Delete activity events older than *retention_days*, in batches.

        Each batch deletes up to ``_BATCH_SIZE`` (10 000) rows and commits
        immediately, preventing long-running transactions.

        Args:
            retention_days: Events older than this many days are removed.
                Must be >= 7 (validated at the config level).

        Returns:
            Total number of rows deleted.
        """
        cutoff = (self._now() - timedelta(days=retention_days)).isoformat()
        total_deleted = 0

        while True:
            # Select the IDs to delete (limit batch size)
            select_stmt = (
                sa.select(activity_events.c.id)
                .where(activity_events.c.timestamp < cutoff)
                .limit(_BATCH_SIZE)
            )

            async with self._session_factory() as session:
                ids_result = await session.execute(select_stmt)
                ids = [row[0] for row in ids_result.fetchall()]

                if not ids:
                    break

                delete_stmt = sa.delete(activity_events).where(
                    activity_events.c.id.in_(ids)
                )
                await session.execute(delete_stmt)
                await session.commit()

            total_deleted += len(ids)
            logger.debug(
                "Cleanup batch: deleted {} events (total {})",
                len(ids),
                total_deleted,
            )

        if total_deleted > 0:
            logger.info(
                "Event cleanup complete: deleted {} events older than {} days",
                total_deleted,
                retention_days,
            )
        else:
            logger.debug("Event cleanup: no events to delete")

        return total_deleted

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

    def __repr__(self) -> str:
        return f"<MaintenanceService data_dir={self._data_dir}>"
