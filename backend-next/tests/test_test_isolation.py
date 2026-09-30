"""The default test harness must not use production credentials or networks."""

import os
import socket
from pathlib import Path

import httpx
import pytest

from mindflow.config import LLMSettings, get_settings


def test_runtime_credentials_and_data_are_isolated(tmp_path: Path) -> None:
    assert not any(
        name.startswith("MINDFLOW_")
        and name not in {"MINDFLOW_E2E_BASE_URL", "MINDFLOW_E2E_TOKEN_FILE"}
        for name in os.environ
    )
    assert LLMSettings().deepseek_credential is None
    assert get_settings().data_dir == tmp_path / "mindflow"


def test_unmocked_sync_http_is_blocked() -> None:
    with (
        httpx.Client(trust_env=False) as client,
        pytest.raises(AssertionError, match="Unmocked HTTP blocked"),
    ):
        client.get("https://example.invalid/")


async def test_unmocked_async_http_is_blocked() -> None:
    async with httpx.AsyncClient(trust_env=False) as client:
        with pytest.raises(AssertionError, match="Unmocked HTTP blocked"):
            await client.get("http://127.0.0.1:11434/")


async def test_in_memory_http_mocks_remain_available() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True}))
    async with httpx.AsyncClient(transport=transport) as client:
        assert (await client.get("https://example.invalid/")).json() == {"ok": True}


@pytest.mark.parametrize("method", ["connect", "connect_ex"])
def test_external_sockets_are_blocked(method: str) -> None:
    with (
        socket.socket() as sock,
        pytest.raises(AssertionError, match="External socket blocked"),
    ):
        getattr(sock, method)(("192.0.2.1", 443))


def test_socket_pairs_remain_available() -> None:
    first, second = socket.socketpair()
    with first, second:
        first.sendall(b"ok")
        assert second.recv(2) == b"ok"
