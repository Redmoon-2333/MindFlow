"""One-time bootstrap authentication routes for the local desktop UI."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from mindflow.infrastructure.security.token_manager import (
    BootstrapTicketStore,
    SessionTokenStore,
)

router = APIRouter(tags=["auth"])
_COOKIE_NAME = "mindflow_session"


class BootstrapRequest(BaseModel):
    ticket: str = Field(min_length=16)


class BootstrapTicketResponse(BaseModel):
    ticket: str
    expires_in_s: int = 60


@router.post("/auth/bootstrap/ticket", response_model=BootstrapTicketResponse)
async def issue_bootstrap_ticket(request: Request) -> BootstrapTicketResponse:
    """Issue a ticket after AuthMiddleware has authenticated the launcher."""
    store = getattr(request.app.state, "bootstrap_tickets", None)
    if not isinstance(store, BootstrapTicketStore):
        raise HTTPException(status_code=503, detail="启动认证服务不可用")
    return BootstrapTicketResponse(ticket=store.issue())


@router.post("/auth/bootstrap", status_code=204)
async def exchange_bootstrap_ticket(request: Request, body: BootstrapRequest) -> Response:
    """Exchange a one-time ticket for an HttpOnly local API session cookie."""
    store = getattr(request.app.state, "bootstrap_tickets", None)
    if not isinstance(store, BootstrapTicketStore) or not store.consume(body.ticket):
        raise HTTPException(status_code=401, detail="启动票据无效、已过期或已使用")

    session_store = getattr(request.app.state, "browser_sessions", None)
    if not isinstance(session_store, SessionTokenStore):
        raise HTTPException(status_code=503, detail="浏览器会话服务不可用")

    response = Response(status_code=204)
    response.set_cookie(
        key=_COOKIE_NAME,
        value=session_store.issue(),
        httponly=True,
        samesite="strict",
        secure=False,
        path="/api",
        max_age=24 * 60 * 60,
    )
    return response
