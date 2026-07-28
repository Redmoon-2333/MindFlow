"""App classification rule endpoints — /api/v1/app-classifications.

Provides:
  - GET    /app-classifications: List all classification rules
  - POST   /app-classifications: Add a new classification rule
  - PUT    /app-classifications: Replace all classification rules
  - DELETE /app-classifications/{rule_id}: Delete a single rule
  - GET    /app-classifications/unknown-apps: List unclassified apps from events

Classification rules determine how MindFlow categorises activity by
process name and optional window title pattern.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

# ── Pydantic model ────────────────────────────────────────────────────────
from pydantic import BaseModel, Field, field_validator

from mindflow.api.deps import get_classification_rules_repo
from mindflow.infrastructure.repositories.app_classification import (
    AppClassificationRulesRepository,
)

VALID_CATEGORIES = {
    "browser_work",
    "code",
    "communication",
    "document",
    "entertainment",
    "other",
    "social",
}


class ClassificationRuleCreate(BaseModel):
    """Schema for creating or replacing a classification rule.

    Attributes:
        process_name: Executable name, e.g. ``"bilibili.exe"``.
        window_title_pattern: Optional SQL LIKE pattern for window titles.
        category: One of the known activity categories.
        priority: Higher value = checked first (default 0, range 0-100).
    """

    process_name: str = Field(..., min_length=1, max_length=255)
    window_title_pattern: str | None = Field(default=None, max_length=255)
    category: str
    priority: int = Field(default=0, ge=0, le=100)

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        if v not in VALID_CATEGORIES:
            raise ValueError(f"Category must be one of {sorted(VALID_CATEGORIES)}")
        return v


router = APIRouter(tags=["app-classification"])


@router.get("/app-classifications")
async def list_classifications(
    repo: AppClassificationRulesRepository = Depends(get_classification_rules_repo),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return all classification rules for the current user.

    Rules are ordered by priority descending, then by creation time.
    """
    return await repo.get_all(user_id=1)


@router.post("/app-classifications", status_code=201)
async def add_classification(
    body: ClassificationRuleCreate,
    repo: AppClassificationRulesRepository = Depends(get_classification_rules_repo),  # noqa: B008
) -> dict[str, Any]:
    """Add a new classification rule.

    The generated rule (including ``id``, ``created_at``, ``updated_at``)
    is returned in the response body.
    """
    return await repo.add(user_id=1, rule=body.model_dump())


@router.put("/app-classifications")
async def replace_classifications(
    body: list[ClassificationRuleCreate],
    repo: AppClassificationRulesRepository = Depends(get_classification_rules_repo),  # noqa: B008
) -> list[dict[str, Any]]:
    """Replace all classification rules with a new set.

    All existing rules are deleted first, then the provided rules are
    inserted in order.  Returns the newly created rules with generated
    fields.
    """
    return await repo.replace_all(
        user_id=1,
        rules=[rule.model_dump() for rule in body],
    )


@router.delete("/app-classifications/{rule_id}", status_code=204)
async def delete_classification(
    rule_id: str,
    repo: AppClassificationRulesRepository = Depends(get_classification_rules_repo),  # noqa: B008
) -> None:
    """Delete a single classification rule by its id.

    Returns 204 even if the rule does not exist (idempotent).
    """
    await repo.delete(rule_id)


@router.get("/app-classifications/unknown-apps")
async def list_unknown_apps(
    repo: AppClassificationRulesRepository = Depends(get_classification_rules_repo),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return process names from recent activity that lack a classification rule.

    Useful for surfacing unclassified apps so the user can assign categories.
    """
    return await repo.get_unknown_apps(user_id=1, limit=20)
