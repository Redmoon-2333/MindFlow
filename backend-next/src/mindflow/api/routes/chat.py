"""Chat API routes for G004 conversational assistant.

Endpoints:
  POST /api/v1/chat          — Send a message and get an AI response
  GET  /api/v1/chat/sessions — List recent chat sessions
  GET  /api/v1/chat/{session_id}/messages — Get messages for a session

Rate limited (5/min, 60/day) via RateLimitMiddleware.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Request, Response  # noqa: B008
from loguru import logger
from pydantic import BaseModel, Field

from mindflow.api.deps import get_chat_service
from mindflow.api.schemas import ChatResponse
from mindflow.graph.chat_graph import _safe_error_metadata
from mindflow.services.chat_service import ChatAnswer, ChatService

router = APIRouter(tags=["chat"])


# ── Request / Response models ─────────────────────────────────────────


class ChatRequest(BaseModel):
    """Incoming chat message from the user.

    Attributes:
        message: The user's text message (non-empty).
        session_id: Optional session identifier. A new UUID is generated
            when omitted.
    """

    message: str = Field(..., min_length=1, description="用户消息")
    session_id: str | None = Field(None, description="会话 ID（新建会话时不传）")


# ── Routes ────────────────────────────────────────────────────────────


@router.post("/chat", response_model=ChatResponse)
async def post_chat(
    body: ChatRequest,
    request: Request,
    chat_service: ChatService = Depends(get_chat_service),  # noqa: B008
) -> ChatResponse | Response:
    """Send a message to the AI assistant.

    Creates a new session if ``session_id`` is omitted. Returns the
    assistant's response along with session and degradation metadata.
    """
    session_id = body.session_id or str(uuid.uuid4())
    user_id = 1  # Single-user desktop app

    logger.info("Chat: user={} session={} msg_len={}", user_id, session_id, len(body.message))

    async def wait_for_disconnect() -> None:
        # FastAPI has consumed the body before entering this endpoint.
        while (await request.receive())["type"] != "http.disconnect":
            pass

    chat_task = asyncio.create_task(chat_service.ask(
        user_id=user_id, session_id=session_id, message=body.message,
    ))
    disconnect_task = asyncio.create_task(wait_for_disconnect())
    try:
        done, _ = await asyncio.wait(
            {chat_task, disconnect_task}, return_when=asyncio.FIRST_COMPLETED,
        )
        if disconnect_task in done:
            disconnect_task.result()
            return Response(status_code=499)
        result: ChatAnswer = await chat_task
    except Exception as exc:
        logger.opt(exception=False).error(
            "Chat service failed unexpectedly: {}", _safe_error_metadata(exc),
        )
        from mindflow.api.errors import _internal_error

        raise _internal_error() from None
    finally:
        # Own both tasks: cancellation must release service/graph locks before exit.
        for task in (chat_task, disconnect_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(chat_task, disconnect_task, return_exceptions=True)

    return ChatResponse(
        answer=result.answer,
        session_id=result.session_id,
        tools_used=list(result.tools_used),
        evidence_cited=result.evidence_cited,
        degraded=result.degraded,
    )


@router.get("/chat/sessions")
async def get_sessions(
    chat_service: ChatService = Depends(get_chat_service),  # noqa: B008
) -> list[dict[str, Any]]:
    """List the most recent chat sessions for the current user."""
    user_id = 1

    try:
        sessions = await chat_service.list_sessions(user_id=user_id, limit=10)
    except Exception as exc:
        logger.opt(exception=False).error(
            "Failed to list chat sessions: {}", _safe_error_metadata(exc),
        )
        from mindflow.api.errors import _internal_error

        raise _internal_error() from None

    return sessions


@router.get("/chat/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    chat_service: ChatService = Depends(get_chat_service),  # noqa: B008
) -> list[dict[str, Any]]:
    """Get the current user's messages for a specific chat session."""
    user_id = 1  # Single-user desktop app; never take ownership from request input.
    try:
        messages = await chat_service.get_messages(session_id=session_id, user_id=user_id)
    except Exception as exc:
        logger.opt(exception=False).error(
            "Failed to list chat messages: {}", _safe_error_metadata(exc),
        )
        from mindflow.api.errors import _internal_error

        raise _internal_error() from None

    return messages
