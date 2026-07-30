"""OpenAPI contract tests for data-visibility endpoints.

Characterisation tests: prove that the current route return shapes match
what the frontend expects (canonical keys preserved).

Schema tests: after response_model annotations are added, the OpenAPI
document must reference concrete schema names (not ``Record<string,
never>``) for every affected operation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# ── Canonical key expectations ────────────────────────────────────────
# These are the EXACT top-level keys the backend services return today.
# If a service changes a key, these tests must fail FIRST.

DAILY_REPORT_KEYS = {
    "user_id",
    "date",
    "total_focus_min",
    "total_distraction_min",
    "focus_score",
    "top_apps",
    "switch_frequency",
    "pattern_summary",
}

WEEKLY_REPORT_KEYS = {
    "week_start",
    "week_end",
    "daily_reports",
    "averages",
    "trend",
    "week_number",
    "intervention_effectiveness",
}

WEEKLY_AVERAGES_KEYS = {
    "avg_focus_min",
    "avg_distraction_min",
    "avg_focus_score",
    "avg_switch_frequency",
}

WEEKLY_TREND_KEYS = {
    "focus_min_delta_pct",
    "focus_score_delta",
    "direction",
}

PATTERNS_KEYS = {
    "high_switch_periods",
    "trigger_apps",
    "heatmap",
    "total_sessions",
    "distraction_ratio",
}

BASELINE_KEYS = {
    "user_id",
    "created_at",
    "updated_at",
    "total_days",
    "total_samples",
    "features",
}

BEHAVIORAL_PROFILE_KEYS = {
    "peak_focus_hours",
    "top_apps",
    "avg_focus_block_min",
    "distraction_triggers",
    "total_events_analysed",
    "profile_date",
}

MODEL_STATUS_KEYS = {
    "loaded",
    "ready",
    "mode",
    "v2_mode",
    "message",
}

ATTRIBUTION_RESPONSE_KEYS = {
    "assessment",
    "source",
    "cached",
    "meta",
}

ATTRIBUTION_ASSESSMENT_KEYS = {
    "procrastination_types",
    "type_confidence",
    "cbt_technique",
    "response_text",
}

# ── OpenAPI helper ────────────────────────────────────────────────────


def _get_openapi_schema() -> dict[str, Any]:
    """Export the current FastAPI OpenAPI document."""
    from mindflow.app import create_app

    return create_app().openapi()


def _get_operation_response_schema(
    openapi: dict[str, Any],
    path: str,
    method: str,
    status: int = 200,
) -> dict[str, Any] | None:
    """Resolve the response schema for a given operation."""
    path_item = openapi.get("paths", {}).get(path, {})
    operation = path_item.get(method, {})
    response = operation.get("responses", {}).get(str(status), {})
    content = response.get("content", {}).get("application/json", {})
    return content.get("schema")


def _is_ref(schema: dict[str, Any] | None) -> bool:
    """Return True when the schema is a concrete $ref (not opaque)."""
    return schema is not None and "$ref" in schema


def _ref_name(schema: dict[str, Any]) -> str:
    """Extract the component name from a ``$ref`` pointer."""
    return schema["$ref"].rsplit("/", 1)[-1]


# ── Schema tests (these MUST pass after response_model is added) ─────


class TestOpenAPIEndpointSchemasAreConcrete:
    """Each affected operation must point to a named schema, not opaque."""

    @pytest.fixture(scope="class")
    def openapi(self) -> dict[str, Any]:
        return _get_openapi_schema()

    def _assert_concrete(
        self,
        openapi: dict[str, Any],
        path: str,
        method: str,
        expected_schema_prefix: str | None = None,
    ) -> str:
        schema = _get_operation_response_schema(openapi, path, method)
        assert schema is not None, f"{method.upper()} {path}: no response schema found"
        assert _is_ref(schema), (
            f"{method.upper()} {path}: expected a $ref, got {schema}"
        )
        name = _ref_name(schema)
        if expected_schema_prefix:
            assert name == expected_schema_prefix, (
                f"{method.upper()} {path}: expected {expected_schema_prefix}, got {name}"
            )
        return name

    def test_reports_daily_schema(self, openapi):
        self._assert_concrete(openapi, "/api/v1/reports/daily", "get", "DailyReportResponse")

    def test_reports_weekly_schema(self, openapi):
        self._assert_concrete(openapi, "/api/v1/reports/weekly", "get", "WeeklyReportResponse")

    def test_analytics_patterns_schema(self, openapi):
        self._assert_concrete(openapi, "/api/v1/analytics/patterns", "get", "PatternsResponse")

    def test_analytics_baseline_schema(self, openapi):
        self._assert_concrete(openapi, "/api/v1/analytics/baseline", "get", "BaselineResponse")

    def test_analytics_profile_schema(self, openapi):
        self._assert_concrete(openapi, "/api/v1/analytics/profile", "get", "BehavioralProfileResponse")

    def test_analytics_model_status_schema(self, openapi):
        self._assert_concrete(openapi, "/api/v1/analytics/model-status", "get", "ModelStatusResponse")

    def test_analytics_attribution_schema(self, openapi):
        self._assert_concrete(openapi, "/api/v1/analytics/attribution", "post", "AttributionResponse")

    # Already-concrete endpoints (must stay concrete)
    def test_ai_runs_schema(self, openapi):
        self._assert_concrete(openapi, "/api/v1/ai/runs", "get", "DiagnosticsListResponse")

    def test_ai_run_detail_schema(self, openapi):
        self._assert_concrete(openapi, "/api/v1/ai/runs/{run_id}", "get", "RunDetail")


class TestSchemaComponentKeys:
    """Verify each schema component includes the canonical field names."""

    @pytest.fixture(scope="class")
    def openapi(self) -> dict[str, Any]:
        return _get_openapi_schema()

    @pytest.fixture(scope="class")
    def schemas(self, openapi) -> dict[str, Any]:
        return openapi.get("components", {}).get("schemas", {})

    def test_daily_report_keys(self, schemas):
        props = set(schemas["DailyReportResponse"]["properties"].keys())
        assert DAILY_REPORT_KEYS <= props, f"Missing: {DAILY_REPORT_KEYS - props}"

    def test_weekly_report_keys(self, schemas):
        props = set(schemas["WeeklyReportResponse"]["properties"].keys())
        assert WEEKLY_REPORT_KEYS <= props, f"Missing: {WEEKLY_REPORT_KEYS - props}"

    def test_patterns_keys(self, schemas):
        props = set(schemas["PatternsResponse"]["properties"].keys())
        assert PATTERNS_KEYS <= props, f"Missing: {PATTERNS_KEYS - props}"

    def test_baseline_keys(self, schemas):
        props = set(schemas["BaselineResponse"]["properties"].keys())
        assert BASELINE_KEYS <= props, f"Missing: {BASELINE_KEYS - props}"

    def test_behavioral_profile_keys(self, schemas):
        props = set(schemas["BehavioralProfileResponse"]["properties"].keys())
        assert BEHAVIORAL_PROFILE_KEYS <= props, f"Missing: {BEHAVIORAL_PROFILE_KEYS - props}"

    def test_model_status_keys(self, schemas):
        props = set(schemas["ModelStatusResponse"]["properties"].keys())
        assert MODEL_STATUS_KEYS <= props, f"Missing: {MODEL_STATUS_KEYS - props}"

    def test_attribution_response_keys(self, schemas):
        props = set(schemas["AttributionResponse"]["properties"].keys())
        assert ATTRIBUTION_RESPONSE_KEYS <= props, f"Missing: {ATTRIBUTION_RESPONSE_KEYS - props}"

    def test_attribution_assessment_keys(self, schemas):
        props = set(schemas["AttributionAssessment"]["properties"].keys())
        assert ATTRIBUTION_ASSESSMENT_KEYS <= props, f"Missing: {ATTRIBUTION_ASSESSMENT_KEYS - props}"

    # Diagnostics already have schemas — verify they exist
    def test_diagnostics_list_schema_exists(self, schemas):
        assert "DiagnosticsListResponse" in schemas

    def test_run_detail_schema_exists(self, schemas):
        assert "RunDetail" in schemas

    def test_node_event_summary_schema_exists(self, schemas):
        assert "NodeEventSummary" in schemas
