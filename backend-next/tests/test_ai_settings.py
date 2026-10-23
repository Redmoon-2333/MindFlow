"""Credential persistence, precedence, secrecy and authentication."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.routes.ai_settings import router
from mindflow.config import Settings


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Explicitly bind the credential file to this test's temporary directory.
    import mindflow.api.routes.ai_settings as ai_settings

    monkeypatch.setattr(ai_settings, "_env_path", lambda: tmp_path / ".env")
    app = FastAPI()
    app.state.settings = Settings()
    app.include_router(router, prefix="/api/v1")
    return TestClient(app)


def test_save_is_secret_free_and_loads_after_restart(client: TestClient, tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "MINDFLOW_LOG_LEVEL='DEBUG'\nDEEPSEEK_API_KEY='old-key-example'\n", encoding="utf-8"
    )
    secret = "sk-test-only-123456789"
    response = client.put("/api/v1/settings/ai", json={"api_key": secret})
    assert response.status_code == 200
    assert secret not in response.text
    assert response.json()["saved"] is True
    assert response.json()["restart_required"] is True
    content = path.read_text(encoding="utf-8")
    assert "MINDFLOW_LOG_LEVEL='DEBUG'" in content
    assert "old-key-example" not in content
    reloaded = Settings(_env_file=path)  # type: ignore[call-arg]
    assert reloaded.llm.l1_target().api_key == secret
    assert secret not in client.get("/api/v1/settings/ai").text
    deleted = client.delete("/api/v1/settings/ai")
    assert deleted.status_code == 200
    assert deleted.json()["saved"] is False
    assert "MINDFLOW_LOG_LEVEL='DEBUG'" in path.read_text(encoding="utf-8")


def test_environment_override_is_explicit(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-environment-test-123")
    response = client.put("/api/v1/settings/ai", json={"api_key": "sk-saved-test-456"})
    assert response.json()["environment_override"] is True
    assert "sk-saved" not in response.text


def test_invalid_secret_does_not_echo_or_write(client: TestClient, tmp_path: Path) -> None:
    path = tmp_path / ".env"
    response = client.put("/api/v1/settings/ai", json={"api_key": "private\nunsafe-value"})
    assert response.status_code == 400
    assert "private" not in response.text
    assert not path.exists()


def test_routes_require_local_auth(tmp_path: Path) -> None:
    from mindflow.app import create_app

    app = create_app(Settings(data_dir=tmp_path), configure_logging=False)
    client = TestClient(app)
    for method in ("GET", "PUT", "DELETE"):
        response = client.request(
            method,
            "/api/v1/settings/ai",
            json={"api_key": "sk-test-only-123"} if method == "PUT" else None,
        )
        assert response.status_code == 401


@pytest.mark.parametrize("model", ["ecnu-max", "ecnu-plus"])
def test_ecnu_save_restart_and_switch_keep_credentials_separate(
    client: TestClient,
    tmp_path: Path,
    model: str,
) -> None:
    campus_key = "sk-campus-test-12345"
    direct_key = "sk-direct-test-12345"
    client.put("/api/v1/settings/ai", json={"api_key": direct_key})
    response = client.put(
        "/api/v1/settings/ai",
        json={
            "service": "ecnu",
            "ecnu_model": model,
            "api_key": campus_key,
        },
    )
    assert response.status_code == 200
    assert response.json()["provider"] == "ChatECNU"
    assert response.json()["model"] == model
    assert response.json()["restart_required"] is True
    assert campus_key not in response.text
    path = tmp_path / ".env"
    reloaded = Settings(_env_file=path)  # type: ignore[call-arg]
    target = reloaded.llm.l1_target()
    assert reloaded.llm.ecnu_compat_enabled is False
    assert target.api_key == campus_key
    assert target.base_url == "https://chat.ecnu.edu.cn/open/api/v1"
    assert target.model == model
    client.app.state.settings = reloaded  # type: ignore[union-attr]
    assert client.get("/api/v1/settings/ai").json()["restart_required"] is False
    response = client.put("/api/v1/settings/ai", json={"api_key": direct_key})
    assert response.json()["restart_required"] is True
    assert response.json()["active_provider"] == "ChatECNU"
    switched = Settings(_env_file=path)  # type: ignore[call-arg]
    assert switched.llm.l1_target().api_key == direct_key
    assert switched.llm.ecnu_api_key == campus_key
    client.delete("/api/v1/settings/ai")
    after_delete = Settings(_env_file=path)  # type: ignore[call-arg]
    assert after_delete.llm.deepseek_api_key is None
    assert after_delete.llm.ecnu_api_key == campus_key


def test_ecnu_environment_selection_and_key_override_file(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINDFLOW_LLM__SERVICE", "ecnu")
    monkeypatch.setenv("MINDFLOW_LLM__ECNU_API_KEY", "sk-campus-env-test")
    response = client.put("/api/v1/settings/ai", json={"api_key": "sk-direct-file-test"})
    assert response.json()["environment_override"] is True
    assert response.json()["service"] == "ecnu"
    target = Settings(_env_file=tmp_path / ".env").llm.l1_target()  # type: ignore[call-arg]
    assert target.api_key == "sk-campus-env-test"
    assert target.provider == "ecnu"
