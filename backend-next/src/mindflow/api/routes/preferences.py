"""User preferences endpoints — /api/v1/preferences.

Provides:
  - GET /preferences: Read all preferences as JSON
  - PUT /preferences: Replace all preferences
  - PATCH /preferences: Merge partial updates into existing preferences

Preferences are stored in the ``user_preferences`` table as a single JSON
blob per user.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends

from mindflow.api.deps import get_preferences_repo
from mindflow.infrastructure.repositories.preferences import (
    PreferencesRepository,
)

router = APIRouter(tags=["preferences"])


@router.get("/preferences")
async def get_preferences(
    preferences_repo: PreferencesRepository = Depends(get_preferences_repo),  # noqa: B008
) -> dict[str, Any]:
    """Return the user's preferences as a JSON object.

    If no preferences have been set, returns an empty dict ``{}``.
    """
    prefs = await preferences_repo.get(user_id=1)
    return prefs


@router.put("/preferences")
async def replace_preferences(
    preferences_repo: PreferencesRepository = Depends(get_preferences_repo),  # noqa: B008
    body: dict[str, Any] = Body(default={}),  # noqa: B008
) -> dict[str, Any]:
    """Replace all user preferences with the provided JSON object."""
    await preferences_repo.set(user_id=1, preferences=body)
    return body


@router.patch("/preferences")
async def update_preferences(
    preferences_repo: PreferencesRepository = Depends(get_preferences_repo),  # noqa: B008
    body: dict[str, Any] = Body(default={}),  # noqa: B008
) -> dict[str, Any]:
    """Merge the provided JSON into existing preferences."""
    existing = await preferences_repo.get(user_id=1)
    existing.update(body)
    merged = {k: v for k, v in existing.items() if v is not None}
    await preferences_repo.set(user_id=1, preferences=merged)
    return merged
