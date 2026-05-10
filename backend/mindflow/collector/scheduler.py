from datetime import datetime, timezone
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler

from mindflow.config import settings
from mindflow.models.database import SessionLocal
from mindflow.models.schemas import User, ActivityLog
from mindflow.collector.tracker import get_active_window_info, is_user_idle
from mindflow.logging_config import get_logger

logger = get_logger(__name__)


class CollectorScheduler:
    def __init__(self):
        self._scheduler: Optional[BackgroundScheduler] = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def _ensure_default_user(self, db) -> int:
        user = db.query(User).first()
        if user is None:
            user = User(username="default", preferences={})
            db.add(user)
            db.commit()
            db.refresh(user)
            logger.info("Created default user (id=%d)", user.id)
        return user.id

    def _collect_tick(self):
        db = SessionLocal()
        try:
            user_id = self._ensure_default_user(db)
            idle = is_user_idle(settings.idle_threshold_seconds)
            info = get_active_window_info()

            if info is None:
                activity = ActivityLog(
                    user_id=user_id,
                    timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
                    process_name="unknown",
                    window_title="",
                    window_class="",
                    duration_seconds=settings.collect_interval_seconds,
                    is_idle=1 if idle else 0,
                )
            else:
                activity = ActivityLog(
                    user_id=user_id,
                    timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
                    process_name=info.get("process_name", "unknown"),
                    window_title=info.get("window_title", ""),
                    window_class=info.get("window_class", ""),
                    duration_seconds=settings.collect_interval_seconds,
                    is_idle=1 if idle else 0,
                )
            db.add(activity)
            db.commit()
        except Exception:
            db.rollback()
            logger.warning("Collection tick failed", exc_info=True)
        finally:
            db.close()

    def start(self):
        if self._running:
            return
        self._scheduler = BackgroundScheduler()
        self._scheduler.add_job(
            self._collect_tick,
            "interval",
            seconds=settings.collect_interval_seconds,
            id="collect_tick",
            replace_existing=True,
        )
        self._scheduler.start()
        self._running = True
        logger.info("Collector started (interval=%ds)", settings.collect_interval_seconds)

    def stop(self):
        if self._scheduler:
            self._scheduler.remove_all_jobs()
            self._scheduler.shutdown(wait=False)
            self._scheduler = None
        self._running = False
        logger.info("Collector stopped")


collector = CollectorScheduler()
