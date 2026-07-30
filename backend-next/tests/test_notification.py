"""Tests for Windows intervention desktop notifications."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from mindflow.infrastructure import notification


class _FakeProcess:
    def __init__(self, returncode: int | None) -> None:
        self.pid = 4321
        self.returncode = returncode
        self.terminated = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True


class _FakeRoot:
    def __init__(self) -> None:
        self.destroyed = False

    def destroy(self) -> None:
        self.destroyed = True


class _FakeUrlResponse:
    def __enter__(self) -> _FakeUrlResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _popup_payload() -> dict[str, object]:
    return {
        "title": "专注提醒",
        "body": "先完成当前任务的下一小步。",
        "intervention_id": "intervention-123",
        "api_url": (
            "http://127.0.0.1:8765/api/v1/intervention/"
            "intervention-123/response"
        ),
        "timeout_s": 120,
    }


def _use_popup_temp_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    popup_dir = tmp_path / "popup"

    def fake_mkdtemp(*, prefix: str) -> str:
        assert prefix == "mindflow_intervention_"
        popup_dir.mkdir()
        return str(popup_dir)

    monkeypatch.setattr(notification.tempfile, "mkdtemp", fake_mkdtemp)
    return popup_dir


async def test_intervention_popup_launches_pythonw_with_json_payload_and_ready_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    popup_dir = _use_popup_temp_dir(monkeypatch, tmp_path)
    python_dir = tmp_path / "python"
    python_dir.mkdir()
    python_executable = python_dir / "python.exe"
    pythonw_executable = python_dir / "pythonw.exe"
    pythonw_executable.touch()
    monkeypatch.setattr(notification.sys, "executable", str(python_executable))

    captured: dict[str, Any] = {}

    def fake_popen(command: list[str], **kwargs: object) -> _FakeProcess:
        captured["command"] = command
        captured["kwargs"] = kwargs
        payload_path = Path(command[2])
        ready_path = Path(command[3])
        captured["payload"] = json.loads(payload_path.read_text(encoding="utf-8"))
        ready_path.write_text("ready", encoding="utf-8")
        return _FakeProcess(returncode=None)

    monkeypatch.setattr(notification.subprocess, "Popen", fake_popen)

    notifier = notification._TkinterInteractivePopup(
        api_base_url="http://127.0.0.1:9876"
    )
    result = await notifier.send(
        "专注提醒",
        "先完成当前任务的下一小步。",
        intervention_id="intervention-123",
        auth_token="system-token",
    )

    assert result is True
    assert captured["command"][0] == str(pythonw_executable)
    assert captured["command"][1].endswith("intervention_popup.py")
    expected_payload = _popup_payload()
    expected_payload["api_url"] = (
        "http://127.0.0.1:9876/api/v1/intervention/"
        "intervention-123/response"
    )
    assert captured["payload"] == expected_payload
    assert captured["kwargs"]["env"]["MINDFLOW_POPUP_TOKEN"] == "system-token"
    assert captured["kwargs"]["creationflags"] == notification.subprocess.CREATE_NO_WINDOW
    assert not popup_dir.exists()


async def test_intervention_popup_returns_false_when_child_exits_before_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    popup_dir = _use_popup_temp_dir(monkeypatch, tmp_path)
    process = _FakeProcess(returncode=1)
    monkeypatch.setattr(
        notification.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )

    notifier = notification._TkinterInteractivePopup()
    result = await notifier.send(
        "标题",
        "正文",
        intervention_id="intervention-123",
        auth_token="system-token",
    )

    assert result is False
    assert not popup_dir.exists()


async def test_intervention_popup_returns_false_and_terminates_on_ready_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    popup_dir = _use_popup_temp_dir(monkeypatch, tmp_path)
    process = _FakeProcess(returncode=None)
    monkeypatch.setattr(
        notification.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(notification._TkinterInteractivePopup, "_READY_TIMEOUT_S", 0.0)

    notifier = notification._TkinterInteractivePopup()
    result = await notifier.send(
        "标题",
        "正文",
        intervention_id="intervention-123",
        auth_token="system-token",
    )

    assert result is False
    assert process.terminated is True
    assert not popup_dir.exists()


async def test_windows_notifier_marks_plain_fallback_as_degraded() -> None:
    interactive = AsyncMock()
    interactive.send.return_value = False
    backend = AsyncMock()
    backend.send.return_value = True
    notifier = notification.WindowsNotifier.__new__(notification.WindowsNotifier)
    notifier._interactive = interactive
    notifier._backends = [backend]

    result = await notifier.send(
        "标题",
        "正文",
        intervention_id="intervention-123",
        auth_token="system-token",
    )

    assert result is False
    backend.send.assert_awaited_once_with("标题", "正文", "normal")


@pytest.mark.parametrize(
    ("button_text", "expected_response"),
    [
        ("接受", "accepted"),
        ("拒绝", "dismissed"),
        ("暂时忽略", "ignored"),
    ],
)
def test_chinese_popup_action_posts_mapped_response_with_actual_latency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    button_text: str,
    expected_response: str,
) -> None:
    popup_module = importlib.import_module(
        "mindflow.infrastructure.intervention_popup"
    )
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, *, timeout: float) -> _FakeUrlResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeUrlResponse()

    times = iter([10.0, 12.75])
    monkeypatch.setattr(popup_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(popup_module.time, "monotonic", lambda: next(times))
    monkeypatch.setenv("MINDFLOW_POPUP_TOKEN", "system-token")

    popup = popup_module.InterventionPopup(_popup_payload(), tmp_path / "ready")
    root = _FakeRoot()
    popup._root = root
    popup._on_action(button_text)

    request = captured["request"]
    assert request.full_url == _popup_payload()["api_url"]
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Bearer system-token"
    assert request.get_header("Content-type") == "application/json"
    assert json.loads(request.data.decode("utf-8")) == {
        "response": expected_response,
        "latency_s": 2.75,
    }
    assert captured["timeout"] == 5.0
    assert root.destroyed is True


@pytest.mark.parametrize("handler_name", ["_on_close", "_on_timeout"])
def test_popup_close_and_timeout_post_ignored(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    handler_name: str,
) -> None:
    popup_module = importlib.import_module(
        "mindflow.infrastructure.intervention_popup"
    )
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, *, timeout: float) -> _FakeUrlResponse:
        captured["request"] = request
        return _FakeUrlResponse()

    times = iter([20.0, 23.5])
    monkeypatch.setattr(popup_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(popup_module.time, "monotonic", lambda: next(times))
    monkeypatch.setenv("MINDFLOW_POPUP_TOKEN", "system-token")

    popup = popup_module.InterventionPopup(_popup_payload(), tmp_path / "ready")
    popup._root = _FakeRoot()
    getattr(popup, handler_name)()

    request_body = json.loads(captured["request"].data.decode("utf-8"))
    assert request_body == {"response": "ignored", "latency_s": 3.5}
