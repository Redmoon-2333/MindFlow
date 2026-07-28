"""Tests for the desktop bootstrap URL helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from mindflow.bootstrap import request_bootstrap_url
from mindflow.config import Settings


@pytest.mark.asyncio
async def test_request_bootstrap_url_uses_protected_ticket_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token_path = tmp_path / "token"
    token_path.write_text("system-token\n", encoding="utf-8")
    settings = Settings(
        host="127.0.0.1",
        port=8765,
        data_dir=tmp_path,
    )
    captured: dict[str, object] = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"ticket": "ticket-value"}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def post(self, url: str, *, headers: dict[str, str]):
            captured.update(url=url, headers=headers)
            return Response()

    monkeypatch.setattr("mindflow.bootstrap.httpx.AsyncClient", lambda **_: Client())

    result = await request_bootstrap_url(settings)

    assert result == "http://127.0.0.1:8765/#bootstrap=ticket-value"
    assert captured == {
        "url": "http://127.0.0.1:8765/api/v1/auth/bootstrap/ticket",
        "headers": {"Authorization": "Bearer system-token"},
    }


@pytest.mark.asyncio
async def test_request_bootstrap_url_never_sends_root_token_to_configured_hostname(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "token").write_text("system-token\n", encoding="utf-8")
    settings = Settings(host="example.test", port=8765, data_dir=tmp_path)
    captured_url = ""

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"ticket": "ticket-value"}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def post(self, url: str, *, headers: dict[str, str]):
            nonlocal captured_url
            captured_url = url
            assert headers == {"Authorization": "Bearer system-token"}
            return Response()

    monkeypatch.setattr("mindflow.bootstrap.httpx.AsyncClient", lambda **_: Client())

    await request_bootstrap_url(settings)

    assert captured_url.startswith("http://127.0.0.1:8765/")
