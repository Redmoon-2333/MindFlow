"""Regression tests: /focus/trend avg_score must be the mean of included sessions.

The trend route only depends on ``focus_repository`` and ``business_today``,
so a minimal app fixture suffices — no activity events or analysis service.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import mindflow.api.routes.focus as focus_module
from mindflow.api.errors import register_exception_handlers
from mindflow.api.routes.focus import router as focus_router
from mindflow.infrastructure.repositories.focus import (
    SQLAlchemyFocusSessionRepository,
    focus_sessions,
)


def _utc(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=UTC)


@pytest.fixture
async def trend_app(engine, session_factory) -> FastAPI:
    """Minimal app exposing the real /focus/trend route with a real repo."""
    async with engine.begin() as conn:
        await conn.run_sync(focus_sessions.metadata.create_all)

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(focus_router, prefix="/api/v1")
    app.state.focus_repository = SQLAlchemyFocusSessionRepository(session_factory)
    app.state.settings = None
    return app


class TestFocusTrendAvgScore:
    """avg_score in /focus/trend must be the mean of the included sessions."""

    @staticmethod
    async def _seed_scores(
        trend_app: FastAPI, by_date_scores: dict[str, list[float | None]]
    ) -> None:
        focus_repo = trend_app.state.focus_repository
        for date_str, scores in by_date_scores.items():
            rows = [
                {
                    "date": date_str,
                    "start_time": _utc(f"{date_str}T08:{10 + i:02d}:00").isoformat(),
                    "end_time": _utc(f"{date_str}T08:{10 + i:02d}:30").isoformat(),
                    "session_type": "focus",
                    "dominant_app": "Code.exe",
                    "focus_score": score,
                    "switch_count": 0,
                }
                for i, score in enumerate(scores)
            ]
            await focus_repo.save_sessions(1, rows)

    def _pin_today(self, trend_app: FastAPI, monkeypatch, day_str: str) -> None:
        monkeypatch.setattr(
            focus_module,
            "business_today",
            MagicMock(return_value=date.fromisoformat(day_str)),
        )
        trend_app.state.settings = SimpleNamespace(timezone="Asia/Shanghai")

    async def test_trend_avg_score_is_mean_of_included_sessions(
        self, trend_app, monkeypatch
    ):
        """Unequal scores must yield the per-day arithmetic mean of included sessions."""
        self._pin_today(trend_app, monkeypatch, "2026-07-26")
        await self._seed_scores(trend_app, {
            "2026-07-25": [80.0, 60.0],
            "2026-07-26": [90.0],
        })

        resp = TestClient(trend_app).get("/api/v1/focus/trend?days=7")
        assert resp.status_code == 200
        daily = {d["date"]: d for d in resp.json()["daily"]}

        assert daily["2026-07-25"]["session_count"] == 2
        assert daily["2026-07-25"]["avg_score"] == 70.0
        assert daily["2026-07-26"]["session_count"] == 1
        assert daily["2026-07-26"]["avg_score"] == 90.0

    async def test_trend_avg_score_missing_and_zero_scores(
        self, trend_app, monkeypatch
    ):
        """Sessions with missing/zero focus_score count as 0 in the mean."""
        self._pin_today(trend_app, monkeypatch, "2026-07-26")
        await self._seed_scores(trend_app, {
            "2026-07-25": [None, 90.0],
            "2026-07-26": [0.0, 90.0],
        })

        resp = TestClient(trend_app).get("/api/v1/focus/trend?days=7")
        daily = {d["date"]: d for d in resp.json()["daily"]}

        assert daily["2026-07-25"]["session_count"] == 2
        assert daily["2026-07-25"]["avg_score"] == 45.0
        assert daily["2026-07-26"]["session_count"] == 2
        assert daily["2026-07-26"]["avg_score"] == 45.0

    def test_trend_empty_keeps_api_contract(self, trend_app, monkeypatch):
        """No sessions → empty daily array and zero totals (existing contract)."""
        self._pin_today(trend_app, monkeypatch, "2026-07-26")
        resp = TestClient(trend_app).get("/api/v1/focus/trend?days=7")
        assert resp.status_code == 200
        data = resp.json()
        assert data["daily"] == []
        assert data["total_sessions"] == 0
        assert data["days"] == 7
