"""Task-context assignment endpoints — /api/v1/task-context.

Turns "I don't know what this app is" into a permanent rule. The period review
can only say "未识别" for time the classifier cannot place; this is where the
user fixes that, once, and the rollup picks it up from then on.

Endpoints:
  - GET  /task-context/unclassified?days=7 — unrecognised apps/sites with time
  - POST /task-context/assign             — save a rule + recompute in background

Privacy: the unclassified list names processes and domains, so it is a
local-only view. It is never part of a period digest and never sent to an LLM.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Request
from loguru import logger
from pydantic import BeforeValidator

from mindflow.api.deps import (
    get_classification_rules_repo,
    get_task_context_rules_repo,
    get_telemetry_service,
)
from mindflow.api.errors import ProblemDetail
from mindflow.api.schemas import (
    TaskContextAssignRequest,
    TaskContextAssignResponse,
    UnclassifiedItemResponse,
    UnclassifiedItemsResponse,
)
from mindflow.domain.task_context import normalize_domain_rule
from mindflow.infrastructure.repositories.app_classification import (
    AppClassificationRulesRepository,
)
from mindflow.infrastructure.repositories.task_context import (
    TaskContextRulesRepository,
)
from mindflow.services.telemetry_service import TelemetryService
from mindflow.time_utils import (
    TimezoneLike,
    business_day_bounds_utc,
    business_today,
)

router = APIRouter(tags=["task-context"])

#: Query spelling of the review spans. ``BeforeValidator`` converts the raw
#: query string first: Pydantic's ``Literal`` compares strictly, so ``"7"``
#: would otherwise be rejected instead of accepted as ``7``.
DaysParam = Annotated[
    Literal[3, 7, 14], BeforeValidator(int), Query(description="回顾天数")
]

#: Activity category an assigned task context maps to. The mapping mirrors
#: ``domain.task_context.DEFAULT_CONTEXT_BY_CATEGORY`` in reverse: an app the
#: user calls "programming" is a ``code`` app, and the existing mapping layer
#: then turns that category back into ``coding``.
CATEGORY_FOR_CONTEXT: dict[str, str] = {
    "coding": "code",
    "writing": "document",
    "reading": "browser_work",
    "meeting": "communication",
    "entertainment": "entertainment",
    "social": "social",
}

#: Priority of an assigned rule. Above the built-in heuristics (0) so the
#: user's answer wins, below an explicit per-title rule the user created in the
#: app-classification UI.
_ASSIGNED_PRIORITY = 50

#: How far back the background recompute regenerates windows.
_RECOMPUTE_DAYS = 14


def _range_for_days(days: int, timezone: TimezoneLike) -> tuple[datetime, datetime]:
    """UTC bounds of the last *days* local business days (today included)."""
    end_date = business_today(timezone)
    start_date = end_date - timedelta(days=days - 1)
    start_utc = business_day_bounds_utc(start_date, timezone)[0]
    end_utc = business_day_bounds_utc(end_date, timezone)[1]
    return start_utc, end_utc


def _validation_error(detail: str) -> ProblemDetail:
    return ProblemDetail(
        type_slug="validation-error",
        title="Validation Error",
        status=422,
        detail=detail,
    )


@router.get("/task-context/unclassified")
async def list_unclassified(
    request: Request,
    days: DaysParam = 7,
    telemetry_service: TelemetryService = Depends(get_telemetry_service),  # noqa: B008
) -> UnclassifiedItemsResponse:
    """Apps and sites the classifier could not place in the last *days* days."""
    settings = getattr(request.app.state, "settings", None)
    timezone: TimezoneLike = getattr(settings, "timezone", "local")
    start_utc, end_utc = _range_for_days(days, timezone)

    items = await telemetry_service.list_unclassified(1, start_utc, end_utc, limit=10)
    return UnclassifiedItemsResponse(
        items=[
            UnclassifiedItemResponse(
                kind=item.kind,
                key=item.key,
                display=item.display,
                minutes=item.minutes,
            )
            for item in items
        ]
    )


async def _recompute_daily(
    telemetry_service: TelemetryService,
    *,
    start_date: date,
    end_date: date,
    timezone: TimezoneLike,
) -> None:
    """Rebuild feature windows one *local day* at a time, off the request path.

    A day at a time keeps each rebuild a bounded query and one transaction, and
    matches how the rollup itself buckets windows. One bad day must not abandon
    the rest, so a failure is logged and the loop continues: the rule is already
    saved and the next scheduled rollup would produce the same windows anyway.
    """
    day = start_date
    while day <= end_date:
        block_start, block_end = business_day_bounds_utc(day, timezone)
        try:
            await telemetry_service.rebuild_feature_windows(
                block_start, block_end, user_id=1, apply=True
            )
        except Exception as exc:  # noqa: BLE001 — one block must not stop the rest
            logger.opt(exception=True).warning(
                "Task-context recompute failed for {}: {}", day, exc
            )
        day += timedelta(days=1)


def _start_recompute(
    request: Request, telemetry_service: TelemetryService, timezone: TimezoneLike
) -> None:
    """Schedule the 14-local-day rebuild; keep a reference so it is not collected."""
    end_date = business_today(timezone)
    task = asyncio.create_task(
        _recompute_daily(
            telemetry_service,
            start_date=end_date - timedelta(days=_RECOMPUTE_DAYS - 1),
            end_date=end_date,
            timezone=timezone,
        )
    )
    tasks: set[asyncio.Task[None]] = getattr(
        request.app.state, "task_context_recompute_tasks", None
    ) or set()
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    request.app.state.task_context_recompute_tasks = tasks


@router.post("/task-context/assign", status_code=201)
async def assign_task_context(
    body: TaskContextAssignRequest,
    request: Request,
    classification_repo: AppClassificationRulesRepository = Depends(  # noqa: B008
        get_classification_rules_repo
    ),
    task_context_repo: TaskContextRulesRepository = Depends(  # noqa: B008
        get_task_context_rules_repo
    ),
    telemetry_service: TelemetryService = Depends(get_telemetry_service),  # noqa: B008
) -> TaskContextAssignResponse:
    """Save a permanent rule for *body.key* and queue a 14-day recompute."""
    settings = getattr(request.app.state, "settings", None)
    timezone: TimezoneLike = getattr(settings, "timezone", "local")

    if body.kind == "app":
        await _assign_app(classification_repo, body.key, body.context)
    else:
        await _assign_site(task_context_repo, body.key, body.context)

    _start_recompute(request, telemetry_service, timezone)
    return TaskContextAssignResponse()


async def _assign_app(
    repo: AppClassificationRulesRepository, process_name: str, context: str
) -> None:
    """Replace the user's untitled rule for this process, then add the new one.

    A title-patterned rule is left alone: it is a narrower answer the user
    already gave for one window title, and this assignment is about the app as
    a whole.
    """
    await _delete_untitled_app_rules(repo, process_name)
    await repo.add(
        user_id=1,
        rule={
            "process_name": process_name,
            "window_title_pattern": None,
            "category": CATEGORY_FOR_CONTEXT[context],
            "priority": _ASSIGNED_PRIORITY,
        },
    )


async def _delete_untitled_app_rules(
    repo: AppClassificationRulesRepository, process_name: str
) -> None:
    """Drop existing process-only rules for *process_name* (case-insensitive)."""
    rules = await repo.get_all(1)
    wanted = process_name.strip().lower()
    for rule in rules:
        if rule.get("window_title_pattern"):
            continue
        if str(rule.get("process_name", "")).strip().lower() != wanted:
            continue
        await repo.delete(str(rule["id"]))


async def _assign_site(
    repo: TaskContextRulesRepository, raw_domain: str, context: str
) -> None:
    """Store a domain → task-context rule, replacing the same domain's old one."""
    domain = normalize_domain_rule(raw_domain)
    if not domain:
        raise _validation_error("网站地址需要包含域名，例如 docs.python.org")

    rules = await repo.get_all(1)
    for rule in rules:
        if str(rule.get("match_type", "")).strip().lower() != "domain":
            continue
        if normalize_domain_rule(rule.get("match_value")) != domain:
            continue
        await repo.delete(str(rule["id"]))

    await repo.add(
        user_id=1,
        rule={
            "match_type": "domain",
            "match_value": domain,
            "task_context": context,
            "priority": _ASSIGNED_PRIORITY,
        },
    )


__all__ = [
    "CATEGORY_FOR_CONTEXT",
    "router",
]
