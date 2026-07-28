"""Public API contract tests for frontend-facing routes."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.routes.chat import router as chat_router
from mindflow.api.routes.collector import router as collector_router
from mindflow.api.routes.intervention import router as intervention_router
from mindflow.api.routes.panel import router as panel_router


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
