"""Blocklist API routes (environment_optimization execution).

User-facing management endpoints (regular session auth):
  - GET    /api/v1/interventions/blocklist      — list blocked sites
  - POST   /api/v1/interventions/blocklist      — add / re-enable a domain
  - PATCH  /api/v1/interventions/blocklist/{domain} — enable/disable
  - DELETE /api/v1/interventions/blocklist/{domain} — remove

The browser extension polls ``GET /telemetry/browser/blocklist`` (defined
in routes/telemetry.py, authenticated by browser pairing token) to apply
declarativeNetRequest rules.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path  # noqa: B008
from pydantic import BaseModel, Field

from mindflow.api.deps import get_blocklist_repo
from mindflow.api.errors import _not_found
from mindflow.api.schemas import (
    BlockedSiteCreateRequest,
    BlockedSiteResponse,
    BlocklistCommandResponse,
    BlocklistResponse,
)
from mindflow.infrastructure.repositories.blocklist import SQLAlchemyBlocklistRepository

router = APIRouter(tags=["intervention"])


class BlockedSiteToggleRequest(BaseModel):
    """Enable/disable toggle for one blocked domain."""

    enabled: bool = Field(description="True to block, False to pause blocking")


@router.get("/interventions/blocklist", response_model=BlocklistResponse)
async def list_blocklist(
    repo: SQLAlchemyBlocklistRepository = Depends(get_blocklist_repo),  # noqa: B008
) -> BlocklistResponse:
    """List the user's blocked sites."""
    rows = await repo.list_all(user_id=1)
    items = [
        BlockedSiteResponse(
            domain=str(row["domain"]),
            enabled=bool(row["enabled"]),
            reason=row.get("reason"),
            created_at=str(row["created_at"]),
        )
        for row in rows
    ]
    return BlocklistResponse(items=items, count=len(items))


@router.post("/interventions/blocklist", response_model=BlocklistCommandResponse)
async def add_blocked_site(
    body: BlockedSiteCreateRequest,
    repo: SQLAlchemyBlocklistRepository = Depends(get_blocklist_repo),  # noqa: B008
) -> BlocklistCommandResponse:
    """Add (or re-enable) a blocked domain."""
    await repo.ensure_blocked(user_id=1, domain=body.domain, reason=body.reason)
    return BlocklistCommandResponse(domain=body.domain)


@router.patch(
    "/interventions/blocklist/{domain}", response_model=BlocklistCommandResponse
)
async def toggle_blocked_site(
    body: BlockedSiteToggleRequest,
    domain: str = Path(..., description="Blocked domain"),  # noqa: B008
    repo: SQLAlchemyBlocklistRepository = Depends(get_blocklist_repo),  # noqa: B008
) -> BlocklistCommandResponse:
    """Enable or disable blocking for one domain."""
    changed = await repo.set_enabled(user_id=1, domain=domain, enabled=body.enabled)
    if not changed:
        raise _not_found(f"拦截域名 {domain}")
    return BlocklistCommandResponse(domain=domain)


@router.delete(
    "/interventions/blocklist/{domain}", response_model=BlocklistCommandResponse
)
async def remove_blocked_site(
    domain: str = Path(..., description="Blocked domain"),  # noqa: B008
    repo: SQLAlchemyBlocklistRepository = Depends(get_blocklist_repo),  # noqa: B008
) -> BlocklistCommandResponse:
    """Permanently remove a blocked domain."""
    removed = await repo.remove(user_id=1, domain=domain)
    if not removed:
        raise _not_found(f"拦截域名 {domain}")
    return BlocklistCommandResponse(domain=domain)
