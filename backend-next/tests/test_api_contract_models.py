"""Public API contract tests for frontend-facing routes.

Also verifies API schema model shapes (field types, required fields) for
chat, panel, and attribution responses.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from mindflow.api.routes.attribution import router as attribution_router
from mindflow.api.routes.chat import router as chat_router
from mindflow.api.routes.collector import router as collector_router
from mindflow.api.routes.intervention import router as intervention_router
from mindflow.api.routes.panel import router as panel_router
from mindflow.api.routes.telemetry import router as telemetry_router
from mindflow.api.schemas import (
    ChatResponse,
    FocusPredictionResponse,
    PanelResponse,
    PanelTranscriptEntry,
)
from mindflow.domain.prediction import FocusPredictionStatus


def _app() -> FastAPI:
    app = FastAPI()
    for router in (collector_router, chat_router, panel_router, intervention_router):
        app.include_router(router, prefix="/api/v1")
    return app


def _response_schema(path: str, method: str = "get") -> dict[str, object]:
    schema = _app().openapi()
    return schema["paths"][path][method]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]


def test_frontend_routes_publish_named_response_models() -> None:
    expected = {
        ("/api/v1/collector", "get"): "CollectorStatusResponse",
        ("/api/v1/chat", "post"): "ChatResponse",
        ("/api/v1/panel/today", "post"): "PanelResponse",
        ("/api/v1/panel", "get"): "PanelResponse",
        ("/api/v1/intervention/trigger", "post"): "InterventionTriggerResponse",
        ("/api/v1/intervention/history", "get"): "InterventionHistoryResponse",
    }
    for (path, method), model_name in expected.items():
        assert _response_schema(path, method) == {
            "$ref": f"#/components/schemas/{model_name}"
        }


def test_intervention_commands_use_json_request_bodies() -> None:
    schema = _app().openapi()["paths"]
    trigger = schema["/api/v1/intervention/trigger"]["post"]
    respond = schema["/api/v1/intervention/{intervention_id}/response"]["post"]

    assert "requestBody" in trigger
    assert "requestBody" in respond
    trigger_properties = trigger["requestBody"]["content"]["application/json"]["schema"]
    respond_properties = respond["requestBody"]["content"]["application/json"]["schema"]
    assert trigger_properties == {"$ref": "#/components/schemas/InterventionTriggerRequest"}
    assert respond_properties == {"$ref": "#/components/schemas/InterventionResponseRequest"}


def test_collector_status_exposes_running_boolean() -> None:
    class Collector:
        status = "running"

    app = _app()
    app.state.collector_service = Collector()
    with TestClient(app) as client:
        response = client.get("/api/v1/collector")
    assert response.status_code == 200
    assert response.json() == {"status": "running", "running": True, "message": None}


# ═══════════════════════════════════════════════════════════════════════════════
# API contract shape assertions
# ═══════════════════════════════════════════════════════════════════════════════


class TestChatResponseContract:
    """ChatResponse Pydantic model field types."""

    def test_chat_response_field_types(self) -> None:
        fields = ChatResponse.model_fields
        assert "answer" in fields
        assert fields["answer"].annotation is str
        assert "session_id" in fields
        assert fields["session_id"].annotation is str
        assert "tools_used" in fields
        # tools_used: list[str]
        assert "evidence_cited" in fields
        assert fields["evidence_cited"].annotation is bool
        assert fields["evidence_cited"].default is False
        assert "degraded" in fields
        assert fields["degraded"].annotation is bool
        assert fields["degraded"].default is False

    def test_chat_response_required_fields(self) -> None:
        """answer and session_id are required (no defaults)."""
        fields = ChatResponse.model_fields
        assert fields["answer"].is_required()
        assert fields["session_id"].is_required()

    def test_chat_response_openapi_schema(self) -> None:
        """OpenAPI schema includes all expected ChatResponse fields."""
        schema = _app_with_chat().openapi()
        props = schema["components"]["schemas"]["ChatResponse"]["properties"]
        assert set(props.keys()) == {
            "answer",
            "session_id",
            "tools_used",
            "evidence_cited",
            "degraded",
        }
        assert props["answer"]["type"] == "string"
        assert props["session_id"]["type"] == "string"
        assert props["tools_used"]["type"] == "array"
        assert props["tools_used"]["items"]["type"] == "string"
        assert props["evidence_cited"]["type"] == "boolean"
        assert props["degraded"]["type"] == "boolean"


class TestPanelResponseContract:
    """PanelResponse Pydantic model field types."""

    def test_panel_response_field_types(self) -> None:
        fields = PanelResponse.model_fields
        assert "types" in fields
        assert "confidence" in fields
        assert "technique" in fields
        assert "rationale" in fields
        assert fields["rationale"].annotation is str
        assert "dissent" in fields
        assert "transcript" in fields
        assert "escalated" in fields
        assert fields["escalated"].annotation is bool
        assert "call_count" in fields
        assert fields["call_count"].annotation is int
        assert "degraded" in fields
        assert fields["degraded"].annotation is bool
        assert "meta" in fields

    def test_panel_transcript_entry_field_types(self) -> None:
        fields = PanelTranscriptEntry.model_fields
        assert "role" in fields
        assert fields["role"].annotation is str
        assert "content" in fields
        assert fields["content"].annotation is str
        assert "round" in fields
        assert fields["round"].annotation is int

    def test_panel_response_openapi_schema(self) -> None:
        """OpenAPI schema includes all expected PanelResponse fields."""
        schema = _app_with_panel().openapi()
        props = schema["components"]["schemas"]["PanelResponse"]["properties"]
        assert set(props.keys()) == {
            "types",
            "confidence",
            "technique",
            "rationale",
            "dissent",
            "transcript",
            "escalated",
            "call_count",
            "degraded",
            "meta",
            "source",
            "cached",
            "insufficient_data",
            "uncertainty",
            "evidence_gaps",
        }
        assert props["types"]["type"] == "array"
        assert props["confidence"]["type"] == "object"
        assert props["technique"]["anyOf"][0]["type"] == "string"
        assert props["rationale"]["type"] == "string"
        assert props["dissent"]["type"] == "array"
        assert props["transcript"]["type"] == "array"
        assert props["escalated"]["type"] == "boolean"
        assert props["call_count"]["type"] == "integer"
        assert props["degraded"]["type"] == "boolean"
        assert props["meta"]["$ref"] == "#/components/schemas/PanelMeta"

    def test_panel_meta_field_types(self) -> None:
        """PanelMeta has degraded: bool."""
        from mindflow.api.schemas import PanelMeta

        fields = PanelMeta.model_fields
        assert "degraded" in fields
        assert fields["degraded"].annotation is bool


class TestAttributionResponseContract:
    """Attribution response shape (not a named Pydantic model but hand-built)."""

    def test_attribution_response_openapi_has_200(self) -> None:
        """The route exists and produces a 200 response."""
        app = _app_with_attribution()
        schema = app.openapi()
        path = schema["paths"]["/api/v1/analytics/attribution"]["post"]
        assert "200" in path["responses"]

    def test_attribution_route_exists(self) -> None:
        """Attribution route path and method are registered in the public OpenAPI contract.

        Uses ``app.openapi()`` (stable public API) instead of iterating
        ``app.router.routes`` directly, because FastAPI ≥0.139 wraps included
        routers in ``_IncludedRouter`` objects whose ``.path`` is the empty
        string — the individual route paths are no longer visible at the top
        level of ``app.router.routes``.
        """
        schema = _app_with_attribution().openapi()
        path_entry = schema["paths"].get("/api/v1/analytics/attribution")
        assert path_entry is not None, (
            "Path /api/v1/analytics/attribution not found in OpenAPI schema"
        )
        assert "post" in path_entry, (
            "POST method not registered for /api/v1/analytics/attribution"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers for contract tests
# ═══════════════════════════════════════════════════════════════════════════════


def _app_with_chat() -> FastAPI:
    app = FastAPI()
    app.include_router(chat_router, prefix="/api/v1")
    return app


def _app_with_panel() -> FastAPI:
    app = FastAPI()
    app.include_router(panel_router, prefix="/api/v1")
    return app


def _app_with_attribution() -> FastAPI:
    app = FastAPI()
    app.include_router(attribution_router, prefix="/api/v1")
    return app


def _app_with_telemetry() -> FastAPI:
    app = FastAPI()
    app.include_router(telemetry_router, prefix="/api/v1")
    return app


# ═══════════════════════════════════════════════════════════════════════════════
# FocusPredictionResponse contract
# ═══════════════════════════════════════════════════════════════════════════════


class TestFocusPredictionResponseContract:
    """FocusPredictionResponse — canonical GET /telemetry/focus-prediction contract.

    The response exposes exactly ``focus_probability`` (present-and-null when
    unavailable), ``status`` (one of the six domain statuses), ``mode``, and
    ``reason``. It must never add ``prediction`` or ``source``.
    """

    ALL_STATUSES: tuple[FocusPredictionStatus, ...] = (
        "ready",
        "no_model",
        "no_data",
        "stale",
        "schema_mismatch",
        "inference_error",
    )

    def test_focus_prediction_response_field_types(self) -> None:
        fields = FocusPredictionResponse.model_fields
        assert set(fields) == {"focus_probability", "status", "mode", "reason"}
        assert fields["focus_probability"].annotation == float | None
        assert fields["status"].annotation == FocusPredictionStatus
        assert fields["mode"].annotation is str
        assert fields["reason"].annotation is str

    def test_focus_prediction_response_required_fields(self) -> None:
        fields = FocusPredictionResponse.model_fields
        for name in ("focus_probability", "status", "mode", "reason"):
            assert fields[name].is_required(), f"{name} must always be present"

    @pytest.mark.parametrize("status", ALL_STATUSES)
    def test_focus_prediction_response_accepts_all_six_statuses(
        self, status: FocusPredictionStatus
    ) -> None:
        payload = FocusPredictionResponse(
            focus_probability=None,
            status=status,
            mode="ready" if status in ("ready", "no_data", "stale") else "rule_engine_only",
            reason=f"reason for {status}",
        )
        assert payload.status == status

    def test_focus_prediction_response_null_probability_is_preserved(self) -> None:
        payload = FocusPredictionResponse(
            focus_probability=None,
            status="no_model",
            mode="rule_engine_only",
            reason="未加载 ML 模型",
        )
        assert payload.model_dump(mode="json")["focus_probability"] is None

    def test_focus_prediction_response_numeric_probability_serializes(self) -> None:
        payload = FocusPredictionResponse(
            focus_probability=0.73,
            status="ready",
            mode="ready",
            reason="",
        )
        assert payload.model_dump(mode="json")["focus_probability"] == 0.73

    @pytest.mark.parametrize("status", ["bogus", "ready-ish", ""])
    def test_focus_prediction_response_rejects_invalid_status(self, status: str) -> None:
        # Raw untrusted input crosses the parse boundary and must be rejected.
        with pytest.raises(ValidationError):
            FocusPredictionResponse.model_validate(
                {
                    "focus_probability": None,
                    "status": status,
                    "mode": "ready",
                    "reason": "x",
                }
            )

    @pytest.mark.parametrize("probability", [1.5, -0.1, float("nan"), float("inf")])
    def test_focus_prediction_response_rejects_invalid_probability(
        self, probability: float
    ) -> None:
        with pytest.raises(ValidationError):
            FocusPredictionResponse(
                focus_probability=probability,
                status="ready",
                mode="ready",
                reason="x",
            )

    def test_focus_prediction_route_publishes_named_response_model(self) -> None:
        schema = _app_with_telemetry().openapi()
        assert schema["paths"]["/api/v1/telemetry/focus-prediction"]["get"][
            "responses"
        ]["200"]["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/FocusPredictionResponse"
        }

    def test_focus_prediction_openapi_schema(self) -> None:
        schema = _app_with_telemetry().openapi()
        component = schema["components"]["schemas"]["FocusPredictionResponse"]
        props = component["properties"]
        assert set(props) == {"focus_probability", "status", "mode", "reason"}
        # focus_probability is a nullable number bounded to [0, 1].
        probability = props["focus_probability"]["anyOf"]
        assert {part["type"] for part in probability} == {"number", "null"}
        assert probability[0]["minimum"] == 0.0
        assert probability[0]["maximum"] == 1.0
        assert set(props["status"]["enum"]) == set(self.ALL_STATUSES)
        assert props["mode"]["type"] == "string"
        assert props["reason"]["type"] == "string"
        assert set(component["required"]) == {
            "focus_probability",
            "status",
            "mode",
            "reason",
        }
