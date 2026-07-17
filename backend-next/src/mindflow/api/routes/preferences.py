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


_MAX_PREFS_BYTES = 64 * 1024
_MAX_PREFS_DEPTH = 8


def _validate_prefs(body: dict[str, Any]) -> None:
    """Guard against oversized or deeply nested preference payloads.

    No fixed key schema — the frontend owns preference keys — but memory
    and storage abuse is bounded here (security audit M4).
    """
    import json

    from mindflow.api.errors import ProblemDetail

    encoded = json.dumps(body, ensure_ascii=False).encode()
    if len(encoded) > _MAX_PREFS_BYTES:
        raise ProblemDetail(
            type_slug="validation-error",
            title="Validation Error",
            status=422,
            detail=f"偏好设置过大（上限 {_MAX_PREFS_BYTES // 1024}KB）",
        )

    def _depth(obj: Any, level: int = 1) -> int:
        if level > _MAX_PREFS_DEPTH:
            return level
        if isinstance(obj, dict):
            return max((_depth(v, level + 1) for v in obj.values()), default=level)
        if isinstance(obj, list):
            return max((_depth(v, level + 1) for v in obj), default=level)
        return level

    if _depth(body) > _MAX_PREFS_DEPTH:
        raise ProblemDetail(
            type_slug="validation-error",
            title="Validation Error",
            status=422,
            detail=f"偏好设置嵌套过深（上限 {_MAX_PREFS_DEPTH} 层）",
        )


@router.put("/preferences")
async def replace_preferences(
    preferences_repo: PreferencesRepository = Depends(get_preferences_repo),  # noqa: B008
    body: dict[str, Any] = Body(default={}),  # noqa: B008
) -> dict[str, Any]:
    """Replace all user preferences with the provided JSON object."""
    _validate_prefs(body)
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
    _validate_prefs(merged)
    await preferences_repo.set(user_id=1, preferences=merged)
    return merged
