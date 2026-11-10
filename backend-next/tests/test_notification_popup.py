from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from mindflow.infrastructure.intervention_popup import (
    _BUTTON_RESPONSES,
    ACTION_RESPONSES,
    _cjk_wrap_text,
    _countdown_fraction,
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
        "dismiss": "dismissed",
        "ignore": "ignored",
        "close": "ignored",
        "timeout": "ignored",
    }


def test_popup_button_mapping() -> None:
    assert _BUTTON_RESPONSES == {
        "好，试试看": "accepted",
        "先忽略": "ignored",
        "不适合我": "dismissed",
    }


@pytest.mark.parametrize(
    ("elapsed", "timeout", "expected"),
    [
        (0.0, 30.0, 1.0),
        (15.0, 30.0, 0.5),
        (29.9, 30.0, pytest.approx(1 - 29.9 / 30)),
        (30.0, 30.0, 0.0),
        (45.0, 30.0, 0.0),
        (-1.0, 30.0, 1.0),
        (0.0, 0.0, 0.0),
        (5.0, 0.0, 0.0),
        (5.0, -10.0, 0.0),
    ],
)
def test_countdown_fraction_boundaries(
    elapsed: float, timeout: float, expected: float
) -> None:
    assert _countdown_fraction(elapsed, timeout) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # The space between CJK and a digit is the only break opportunity in
        # this phrase, so Tk breaks there and leaves a ragged line.
        ("设定 5 分钟", "设定 5 分钟"),
        # ASCII-only text has no such adjacency and must be untouched.
        ("Hello world", "Hello world"),
        ("CPU usage 80%", "CPU usage 80%"),
        # CJK-to-CJK spaces are legitimate break points; leave them alone.
        ("先完成 一小步", "先完成 一小步"),
        ("检测到似乎有些分心，先完成一小步",
         "检测到似乎有些分心，先完成一小步"),
        # Mixed: only the CJK/alphanumeric adjacencies change.
        (
            "用系统计时器设定 5 分钟，先完成 一小步",
            "用系统计时器设定 5 分钟，先完成 一小步",
        ),
        ("累计 12 次", "累计 12 次"),
        ("耗时 3.5 小时", "耗时 3.5 小时"),
        # No spaces at all.
        ("该回到正题了", "该回到正题了"),
        ("", ""),
    ],
)
def test_cjk_wrap_text_joins_cjk_and_alphanumerics(source: str, expected: str) -> None:
    result = _cjk_wrap_text(source)

    assert result == expected
    # Only spaces may change: turning one into NBSP must never alter any other
    # character, so the two strings are equal once NBSP is folded back.
    assert result.replace(" ", " ") == source
    # And no breakable ASCII space may remain where Tk would prefer to break.
    assert not re.search(
        r"[一-鿿] [A-Za-z0-9]|[A-Za-z0-9] [一-鿿]", result
    )


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
    assert captured["body"] == {
        "response": "dismissed",
        "latency_s": 2.75,
        "source": "human",
    }
    assert captured["timeout"] == 5.0


def test_post_response_can_mark_an_automatic_default() -> None:
    """A timeout default is reported as auto so it cannot beat a real answer."""
    captured: dict[str, Any] = {}

    def opener(request: Any, timeout: float) -> _FakeResponse:
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse()

    post_response(
        api_url="http://127.0.0.1:8765/api/v1/intervention/abc/response",
        auth_token="secret-token",
        response="ignored",
        latency_s=30.0,
        opener=opener,
        source="auto",
    )

    assert captured["body"]["source"] == "auto"


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
