from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mindflow.infrastructure.intervention_popup import (
    ACTION_RESPONSES,
    post_response,
)
from mindflow.infrastructure.notification import _TkinterInteractivePopup


class _FakeResponse:
    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _ReadyProcess:
    def __init__(self) -> None:
        self.terminated = False
        self.pid = 1234
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        self.terminated = True


class _ExitedProcess(_ReadyProcess):
    def poll(self) -> int | None:
        self.returncode = 1
        return 1


def test_popup_action_mapping() -> None:
    assert ACTION_RESPONSES == {
        "accept": "accepted",
        "reject": "dismissed",
        "ignore": "ignored",
        "close": "ignored",
        "timeout": "ignored",
    }


def test_post_response_sends_authorized_json() -> None:
    captured: dict[str, Any] = {}

    def opener(request: Any, timeout: float) -> _FakeResponse:
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse()

    assert post_response(
        api_url="http://127.0.0.1:8765/api/v1/intervention/abc/response",
        auth_token="secret-token",
        response="dismissed",
        latency_s=2.75,
        opener=opener,
    )
    assert captured["url"].endswith("/intervention/abc/response")
    assert captured["headers"]["Authorization"] == "Bearer secret-token"
    assert captured["headers"]["Content-type"] == "application/json"
    assert captured["body"] == {"response": "dismissed", "latency_s": 2.75}
    assert captured["timeout"] == 5.0


@pytest.mark.asyncio
async def test_popup_reports_success_only_after_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    process = _ReadyProcess()

    def popen(args: list[str], **kwargs: Any) -> _ReadyProcess:
        payload_path = Path(args[-2])
        ready_path = Path(args[-1])
        captured["args"] = args
        captured["kwargs"] = kwargs
        captured["payload"] = json.loads(payload_path.read_text(encoding="utf-8"))
        ready_path.write_text("ready", encoding="utf-8")
        return process

    monkeypatch.setattr("mindflow.infrastructure.notification.subprocess.Popen", popen)
    monkeypatch.setattr(
        "mindflow.infrastructure.notification.tempfile.mkdtemp",
        lambda **_: str(tmp_path),
    )

    notifier = _TkinterInteractivePopup()
    result = await notifier.send(
        title="休息一下",
        body="已经连续专注一段时间了。",
        intervention_id="abc-123",
        auth_token="secret-token",
    )

    assert result is True
    assert captured["payload"]["title"] == "休息一下"
    assert captured["payload"]["body"] == "已经连续专注一段时间了。"
    assert captured["payload"]["api_url"].endswith("/intervention/abc-123/response")
    assert "auth_token" not in captured["payload"]
    assert captured["kwargs"]["env"]["MINDFLOW_POPUP_TOKEN"] == "secret-token"
    assert captured["args"][1].endswith("intervention_popup.py")
    assert process.terminated is False


@pytest.mark.asyncio
async def test_popup_returns_false_when_child_exits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    process = _ExitedProcess()
    monkeypatch.setattr(
        "mindflow.infrastructure.notification.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(
        "mindflow.infrastructure.notification.tempfile.mkdtemp",
        lambda **_: str(tmp_path),
    )

    notifier = _TkinterInteractivePopup()
    result = await notifier.send(
        title="提醒",
        body="内容",
        intervention_id="abc-123",
        auth_token="secret-token",
    )

    assert result is False


@pytest.mark.asyncio
async def test_popup_timeout_terminates_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    process = _ReadyProcess()
    monkeypatch.setattr(
        "mindflow.infrastructure.notification.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(
        "mindflow.infrastructure.notification.tempfile.mkdtemp",
        lambda **_: str(tmp_path),
    )
    monkeypatch.setattr(_TkinterInteractivePopup, "_READY_TIMEOUT_S", 0.01)

    notifier = _TkinterInteractivePopup()
    result = await notifier.send(
        title="提醒",
        body="内容",
        intervention_id="abc-123",
        auth_token="secret-token",
    )

    assert result is False
    assert process.terminated is True
