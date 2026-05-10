import pytest
from fastapi.testclient import TestClient

from mindflow.main import app
from mindflow.models.database import SessionLocal, init_db
from mindflow.models.schemas import User

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def _setup_user():
    """Ensure a default user exists before any API test in this module."""
    init_db()
    db = SessionLocal()
    try:
        user = db.query(User).first()
        if user is None:
            user = User(username="test_user", preferences={})
            db.add(user)
            db.commit()
            db.refresh(user)
    finally:
        db.close()


def test_get_status():
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    assert "collector_running" in body["data"]
    assert isinstance(body["data"]["settings"]["collect_interval_seconds"], int)


def test_collector_start_stop():
    start_resp = client.post("/api/v1/collector/start")
    assert start_resp.status_code == 200
    assert start_resp.json()["data"]["collector_running"] is True

    stop_resp = client.post("/api/v1/collector/stop")
    assert stop_resp.status_code == 200
    assert stop_resp.json()["data"]["collector_running"] is False


def test_activities_current():
    response = client.get("/api/v1/activities/current")
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    assert "window" in body["data"]


def test_activities_today():
    response = client.get("/api/v1/activities/today")
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    assert "focus_score" in body["data"]
    assert "top_apps" in body["data"]


def test_focus_today():
    response = client.get("/api/v1/focus/today")
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    assert "focus_score" in body["data"]


def test_focus_trend():
    response = client.get("/api/v1/focus/trend?days=3")
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    assert len(body["data"]["trend"]) == 3


def test_focus_trend_default_days():
    response = client.get("/api/v1/focus/trend")
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["days"] == 7


def test_preferences_get():
    response = client.get("/api/v1/preferences")
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    assert "username" in body["data"]


def test_preferences_put():
    response = client.put(
        "/api/v1/preferences",
        json={"collect_interval_seconds": 10, "intervention_intensity": "strict"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    prefs = body["data"]["preferences"]
    assert prefs.get("collect_interval_seconds") == 10
    assert prefs.get("intervention_intensity") == "strict"


def test_analytics_patterns():
    response = client.get("/api/v1/analytics/patterns")
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    assert "sessions" in body["data"]


def test_reports_weekly():
    response = client.get("/api/v1/reports/weekly")
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    assert "reports" in body["data"]


def test_api_response_format():
    response = client.get("/api/v1/status")
    body = response.json()
    assert "code" in body
    assert "message" in body
    assert "data" in body
    assert "timestamp" in body
    assert body["code"] == 0
    assert body["message"] == "success"
