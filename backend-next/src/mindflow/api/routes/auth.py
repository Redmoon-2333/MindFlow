"""Auth route — login and token management."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(tags=["auth"])


_HARDCODED_USER = "RedMoon2333"
_HARDCODED_PASS = "RedMoon2333"


class LoginBody(BaseModel):
    username: str
    password: str


@router.post("/auth/login")
async def login(request: Request, body: LoginBody | None = None) -> dict[str, str]:
    """Authenticate and return the system token.

    Valid credentials: username=RedMoon2333, password=RedMoon2333.
    If no body is provided, returns the token directly (local-first mode).
    """
    if (
        body is not None
        and (body.username != _HARDCODED_USER or body.password != _HARDCODED_PASS)
    ):
            raise HTTPException(status_code=401, detail="用户名或密码错误")

    token: str = getattr(request.app.state, "system_token", "")
    return {"token": token}
