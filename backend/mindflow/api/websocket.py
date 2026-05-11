import json
import asyncio
import time
from datetime import date, datetime, timedelta

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import func

from mindflow.collector.tracker import get_active_window_info
from mindflow.collector.scheduler import collector
from mindflow.models.database import SessionLocal
from mindflow.models.schemas import ActivityLog, DailyReport
from mindflow.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


def _quick_snapshot():
    """Query today's key metrics for the WebSocket snapshot.
    Uses its own DB session to avoid leaking across the async loop."""
    db = SessionLocal()
    try:
        today = date.today()
        report = (
            db.query(DailyReport)
            .filter(DailyReport.date == today)
            .first()
        )

        recent_app = None
        five_min_ago = datetime.combine(today, datetime.min.time())
        now = datetime.now()
        if (now - five_min_ago).total_seconds() > 300:
            five_min_ago = now - timedelta(minutes=5)
        top = (
            db.query(
                ActivityLog.process_name,
                func.sum(ActivityLog.duration_seconds).label("dur"),
            )
            .filter(
                ActivityLog.timestamp >= five_min_ago,
                ActivityLog.is_idle == 0,
            )
            .group_by(ActivityLog.process_name)
            .order_by(func.sum(ActivityLog.duration_seconds).desc())
            .first()
        )
        if top:
            recent_app = top.process_name

        return {
            "focus_score": report.focus_score if report else None,
            "dominant_app": recent_app,
        }
    except Exception:
        return {"focus_score": None, "dominant_app": None}
    finally:
        db.close()


@router.websocket("/ws/activities")
async def websocket_activities(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected")
    try:
        while True:
            info = get_active_window_info()
            snapshot = _quick_snapshot()
            payload = {
                "type": "activity_update",
                "data": {
                    "window": info,
                    "collector_running": collector.is_running,
                    "timestamp": int(time.time()),
                    "snapshot": snapshot,
                },
            }
            await websocket.send_text(json.dumps(payload, ensure_ascii=False))
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception:
        logger.warning("WebSocket error", exc_info=True)
