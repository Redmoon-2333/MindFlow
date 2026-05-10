import json
import asyncio
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from mindflow.collector.tracker import get_active_window_info
from mindflow.collector.scheduler import collector
from mindflow.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.websocket("/ws/activities")
async def websocket_activities(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected")
    try:
        while True:
            info = get_active_window_info()
            payload = {
                "type": "activity_update",
                "data": {
                    "window": info,
                    "collector_running": collector.is_running,
                    "timestamp": int(time.time()),
                },
            }
            await websocket.send_text(json.dumps(payload, ensure_ascii=False))
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception:
        logger.warning("WebSocket error", exc_info=True)
