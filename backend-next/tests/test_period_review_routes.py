"""Tests for ``POST /api/v1/reports/period-review`` and the chat tool.

The route is tested with a stubbed review service so the assertions stay about
the HTTP contract (status codes, the 3/7/14 day literal, the response shape)
rather than about ML. The chat tool is tested for its look-back mapping and for
returning parseable JSON.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.deps import get_period_review_service
from mindflow.api.errors import register_exception_handlers
from mindflow.api.routes.reports import router as reports_router
from mindflow.domain.period_review import (
    build_period_digest,
    template_narrative,
)
from mindflow.graph.tools import PeriodReviewTool, period_review_days
from mindflow.services.period_review_service import PeriodReviewResult
from mindflow.time_utils import resolve_timezone


def _digest(days: int = 7) -> Any:
    today = date(2026, 11, 16)
    return build_period_digest(
        [],
        [],
        (),
        start_date=today.fromordinal(today.toordinal() - (days - 1)),
        end_date=today,
        tz=resolve_timezone("Asia/Shanghai"),
        model_version="base_v4_20261111_r2",
        is_base_model=True,
        model_available=True,
    )


class _StubReviewService:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, bool]] = []

    async def review(self, user_id: int, days: int, use_ai: bool) -> PeriodReviewResult:
        self.calls.append((user_id, days, use_ai))
        digest = _digest(days)
        return PeriodReviewResult(
            digest=digest,
            llm_payload=digest.model_dump_json(),
            ai_requested=use_ai,
            narrative=template_narrative(digest),
            generated_at="2026-11-16T00:00:00+00:00",
        )

    async def build_digest(self, user_id: int, days: int) -> Any:
        self.calls.append((user_id, days, False))
        return _digest(days)


def _client(service: Any) -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(reports_router, prefix="/api/v1")
    app.state.collector_service = None
    app.state.migration_applied = True
    app.dependency_overrides[get_period_review_service] = lambda: service
    return TestClient(app)


def test_period_review_returns_200_and_the_full_envelope() -> None:
    service = _StubReviewService()
    response = _client(service).post(
        "/api/v1/reports/period-review", json={"days": 7, "use_ai": True}
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "digest",
        "llm_payload",
        "ai_requested",
        "narrative",
        "generated_at",
    }
    assert body["digest"]["period"]["days"] == 7
    assert body["digest"]["data_state"] == "insufficient"
    assert body["ai_requested"] is True
    assert body["narrative"]["source"] == "template"
    assert json.loads(body["llm_payload"])["period"]["days"] == 7
    assert service.calls == [(1, 7, True)]


def test_period_review_defaults_to_seven_days_and_ai() -> None:
    service = _StubReviewService()
    response = _client(service).post("/api/v1/reports/period-review", json={})
    assert response.status_code == 200
    assert service.calls == [(1, 7, True)]


@pytest.mark.parametrize("days", [3, 7, 14])
def test_period_review_accepts_the_three_spans(days: int) -> None:
    service = _StubReviewService()
    response = _client(service).post(
        "/api/v1/reports/period-review", json={"days": days, "use_ai": False}
    )
    assert response.status_code == 200
    assert service.calls == [(1, days, False)]


@pytest.mark.parametrize("days", [5, 0, 30, -1])
def test_period_review_rejects_other_spans_with_422(days: int) -> None:
    service = _StubReviewService()
    response = _client(service).post(
        "/api/v1/reports/period-review", json={"days": days}
    )
    assert response.status_code == 422
    assert service.calls == []


def test_period_review_rejects_unknown_body_fields() -> None:
    response = _client(_StubReviewService()).post(
        "/api/v1/reports/period-review", json={"days": 7, "window_titles": []}
    )
    assert response.status_code == 422


# ══════════════════════════════════════════════════════════════════════════════
# Chat tool
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("days_back", "expected"),
    [(1, 3), (3, 3), (4, 7), (7, 7), (8, 14), (14, 14), (30, 14)],
)
def test_days_back_maps_onto_allowed_spans(days_back: int, expected: int) -> None:
    assert period_review_days(days_back) == expected


async def test_chat_tool_returns_parseable_digest_json() -> None:
    from mindflow.graph.tools import ToolContext

    service = _StubReviewService()
    adapter = PeriodReviewTool(period_review_service=service, timezone="Asia/Shanghai")
    adapter.context = ToolContext(user_id=1)
    output = await adapter.execute(days_back=7)

    assert output.error is None
    payload = json.loads(output.digest_json)
    assert payload["period"]["days"] == 7
    assert output.data_state == payload["data_state"]
    assert service.calls == [(1, 7, False)]


async def test_chat_tool_requires_a_user() -> None:
    adapter = PeriodReviewTool(period_review_service=_StubReviewService())
    output = await adapter.execute(days_back=7)
    assert output.error == "user_id not set"


async def test_chat_tool_reports_service_errors() -> None:
    from mindflow.graph.tools import ToolContext

    class _Broken:
        async def build_digest(self, user_id: int, days: int) -> Any:
            raise RuntimeError("model exploded")

    adapter = PeriodReviewTool(period_review_service=_Broken())
    adapter.context = ToolContext(user_id=1)
    output = await adapter.execute(days_back=7)
    assert output.error is not None
    assert "model exploded" in output.error
