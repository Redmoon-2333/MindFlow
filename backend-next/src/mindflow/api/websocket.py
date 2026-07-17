"""WebSocket handler for real-time activity and focus updates.

Connection: ``/api/v1/ws``

Authentication is handled via **query parameter token** (chosen over
first-message auth; see rationale below). The token is passed as
``?token=<token>`` in the WebSocket URL.

Justification for query-param auth:
  - WebSocket ``Authorization`` headers are not sent by the browser's
    ``WebSocket(url)`` constructor — only ``Cookie`` and custom headers
    set via the ``protocols`` parameter work reliably across all origins.
  - First-message auth would require the client to send a message before
    receiving any real-time data, adding latency and complexity to the
    reconnect flow (especially during exponential backoff).
  - The token is a local-file-only secret (no network transmission), so
    query-param exposure in logs is mitigated by the fact that both the
    frontend and backend run on localhost.
  - WS ``/api/v1/ws`` shares the same token as REST, avoiding token duplication.

Message frames (§4.3 of requirements):

  All messages are JSON text frames with the structure:
  ``{"type": "<event_type>", "payload": <object>, "timestamp": "<ISO8601 UTC>"}``

  Server-to-Client:
    - activity_update: Current active window (throttled to 2s, skip if unchanged)
    - focus_change: Focus state transition
    - intervention: Intervention recommendation
    - error: Error notification
    - pong: Heartbeat response

  Client-to-Server:
    - ping: Heartbeat (client should send every 30s)
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from mindflow.infrastructure.security.token_manager import verify_token

router = APIRouter(tags=["websocket"])

# ── Connection management ──────────────────────────────────────────────────

_active_connections: dict[str, WebSocket] = {}
"""Active WebSocket connections keyed by client_id (connection id string)."""

_connection_lock = asyncio.Lock()
"""Protects _active_connections from concurrent modification."""


async def broadcast(message: dict[str, Any]) -> int:
    """Send a message to all connected WebSocket clients.

    Args:
        message: A dict with ``type``, ``payload``, and optional ``timestamp``.

    Returns:
        Number of clients the message was successfully sent to.
    """
    payload = json.dumps(_with_timestamp(message), ensure_ascii=False)
    sent = 0

    async with _connection_lock:
        disconnected: list[str] = []
        for cid, ws in _active_connections.items():
            try:
                await ws.send_text(payload)
                sent += 1
            except Exception:
                disconnected.append(cid)

        for cid in disconnected:
            _active_connections.pop(cid, None)

    return sent


async def close_all_connections() -> int:
    """Close every active WebSocket connection (app shutdown, review P2-2).

    Returns:
        Number of connections closed.
    """
    async with _connection_lock:
        closed = 0
        for ws in _active_connections.values():
            try:
                await ws.close(code=1001)  # going away
                closed += 1
            except Exception:  # noqa: BLE001 — best-effort during shutdown
                pass
        _active_connections.clear()
    return closed


_last_activity_push: float = 0.0
_last_activity_state: str | None = None


async def broadcast_activity_update(data: dict[str, Any]) -> None:
    """Broadcast an ``activity_update`` message with inline throttling.

    Enforces the §4.4 contract in the function itself (review P3): pushes at
    most once per 2 seconds, and skips entirely when the activity state
    (app + idle flag) has not changed since the last push.
    """
    global _last_activity_push, _last_activity_state
    now = time.monotonic()
    state_key = f"{data.get('app_name')}|{data.get('is_idle')}"
    if state_key == _last_activity_state and now - _last_activity_push < 2.0:
        return
    _last_activity_push = now
    _last_activity_state = state_key
    await broadcast({"type": "activity_update", "payload": data})


# ── Internal helpers ───────────────────────────────────────────────────────


def _with_timestamp(message: dict[str, Any]) -> dict[str, Any]:
    """Ensure the message has a UTC ISO8601 timestamp."""
    if "timestamp" not in message:
        message["timestamp"] = datetime.now(UTC).isoformat()
    return message


# ── WebSocket endpoint ─────────────────────────────────────────────────────


@router.websocket("/ws")
async def websocket_handler(websocket: WebSocket) -> None:
    """Handle WebSocket connections at ``/api/v1/ws``.

    Authentication: token passed as ``?token=...`` query parameter.
    Once authenticated, the client receives real-time activity updates,
    focus changes, and interventions.

    The client should send a ``ping`` message every 30 seconds to keep
    the connection alive. The server responds with ``pong``.
    """
    # ── Authentication via query parameter ──
    token = websocket.query_params.get("token", "")
    expected_token = getattr(websocket.app.state, "system_token", "")

    if not token or not verify_token(token, expected_token):
        await websocket.close(code=4001, reason="Authentication required")
        return

    # ── Accept the connection ──
    await websocket.accept()
    client_id = f"ws:{id(websocket)}"

    async with _connection_lock:
        _active_connections[client_id] = websocket

    logger.info("WebSocket client connected ({})", client_id)

    try:
        await _handle_messages(websocket, client_id)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected ({})", client_id)
    except Exception as exc:
        logger.opt(exception=True).error("WebSocket error ({}): {}", client_id, exc)
    finally:
        async with _connection_lock:
            _active_connections.pop(client_id, None)


async def _handle_messages(websocket: WebSocket, client_id: str) -> None:
    """Main message loop for an authenticated WebSocket connection.

    Handles:
      - ``ping`` messages → responds with ``pong``
      - All other messages → ignored (future extension)
    """
    async for raw in websocket.iter_text():
        try:
            data = json.loads(raw)
            msg_type = data.get("type", "")

            if msg_type == "ping":
                pong_msg = {
                    "type": "pong",
                    "payload": {},
                    "timestamp": datetime.now(UTC).isoformat(),
                }
                await websocket.send_text(json.dumps(pong_msg, ensure_ascii=False))
            # Future: handle other client messages

        except json.JSONDecodeError:
            err_msg = {
                "type": "error",
                "payload": {"code": "INVALID_JSON", "message": "无效的 JSON 格式"},
                "timestamp": datetime.now(UTC).isoformat(),
            }
            with suppress(Exception):
                await websocket.send_text(json.dumps(err_msg, ensure_ascii=False))
