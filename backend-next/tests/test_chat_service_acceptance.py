"""Service-boundary deadline, cancellation, logging and history isolation."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage
from loguru import logger

from mindflow.api.errors import register_exception_handlers
from mindflow.api.routes.chat import router
from mindflow.graph.chat_graph import ChatGraph
from mindflow.infrastructure.repositories.chat import ChatRepository
from mindflow.infrastructure.security.crisis_detector import CrisisLevel
from mindflow.services.chat_service import ChatService


@pytest.fixture
def service(session_factory, create_tables):
    repo = ChatRepository(session_factory)
    detector = MagicMock()
    detector.scan.return_value = (CrisisLevel.NONE, None)
    gateway = SimpleNamespace(_api_key="", _base_url="")
    return ChatService(
        session_factory, detector, gateway, MagicMock(), None, MagicMock(),
        MagicMock(), chat_repo=repo,
    )


@pytest.fixture
def deadlines(monkeypatch):
    original = asyncio.timeout
    captured = []
    created = asyncio.Event()

    def timeout(delay):
        timer = original(delay)
        captured.append((delay, timer))
        created.set()
        return timer

    monkeypatch.setattr(asyncio, "timeout", timeout)
    return captured, created


def install_model(service, invoke):
    model = MagicMock()
    model.ainvoke = AsyncMock(side_effect=invoke)
    service._chat_graph = ChatGraph(service._chat_repo, service._crisis_detector, model=model)
    return model


def start_chat_asgi(service):
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router, prefix="/api/v1")
    app.state.chat_service = service
    app.state.migration_applied = True
    incoming = asyncio.Queue()
    incoming.put_nowait({
        "type": "http.request",
        "body": json.dumps({"message": "synthetic", "session_id": "shared"}).encode(),
        "more_body": False,
    })
    receives, sent = [], []

    async def receive():
        receives.append(asyncio.current_task())
        return await incoming.get()

    async def send(message):
        sent.append(message)

    task = asyncio.create_task(app({
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": "/api/v1/chat",
        "raw_path": b"/api/v1/chat", "query_string": b"", "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 12345), "server": ("test", 80),
    }, receive, send))
    return task, incoming, receives, sent


@pytest.mark.parametrize("disconnect", [True, False])
async def test_route_disconnect_and_outer_cancel_wait_for_provider_cleanup(service, disconnect):
    entered, cancelling = asyncio.Event(), asyncio.Event()
    allow_cleanup, cleaned = asyncio.Event(), asyncio.Event()
    provider_tasks = []

    async def invoke(messages):
        provider_tasks.append(asyncio.current_task())
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelling.set()
            await allow_cleanup.wait()
            cleaned.set()

    install_model(service, invoke)
    task, incoming, receives, sent = start_chat_asgi(service)
    try:
        await asyncio.wait_for(entered.wait(), 2)
        if disconnect:
            incoming.put_nowait({"type": "http.disconnect"})
        else:
            task.cancel()
        await asyncio.wait_for(cancelling.wait(), 2)
        assert not task.done(), "route must await provider cleanup, not abandon the child"
        assert not cleaned.is_set()
        allow_cleanup.set()
        if disconnect:
            await asyncio.wait_for(task, 2)
            assert not any(m.get("status") == 200 for m in sent)
        else:
            with pytest.raises(asyncio.CancelledError):
                await task
        assert cleaned.is_set()
        assert all(t.done() for t in provider_tasks + receives)
        assert not service._session_locks[(1, "shared")].locked()
        assert not service._chat_graph._session_locks[(1, "shared")].locked()
        rows = await service.get_messages("shared", user_id=1)
        assert rows and all(row["role"] == "user" for row in rows)
    finally:
        allow_cleanup.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("provider_fails", [False, True])
async def test_route_completion_cleans_up_disconnect_watcher(service, provider_fails):
    entered, release = asyncio.Event(), asyncio.Event()

    async def invoke(messages):
        entered.set()
        await release.wait()
        if provider_fails:
            raise RuntimeError("synthetic provider failure")
        return AIMessage(content="completed")

    install_model(service, invoke)
    task, _, receives, sent = start_chat_asgi(service)
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert len(receives) == 2, "one body receive and one disconnect watcher"
        watcher = receives[-1]
        assert watcher is not task and not watcher.done()
        release.set()
        await asyncio.wait_for(task, 2)
        assert watcher.done()
        assert any(m.get("status") == 200 for m in sent)
        response = json.loads(b"".join(m.get("body", b"") for m in sent))
        assert response["degraded"] is provider_fails
        rows = await service.get_messages("shared", user_id=1)
        assert sorted(row["role"] for row in rows) == ["assistant", "user"]
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_queue_expiry_never_enters_graph(service, deadlines):
    captured, created = deadlines
    graph = SimpleNamespace(ask=AsyncMock())
    service._chat_graph = graph
    lock = service._get_session_lock(1, "shared")
    await lock.acquire()
    task = asyncio.create_task(service.ask(1, "shared", "queued"))
    try:
        await created.wait()
        assert captured[0][0] == 600
        captured[0][1].reschedule(asyncio.get_running_loop().time() - 1)
        result = await task
        assert result.degraded and not result.evidence_cited
        graph.ask.assert_not_awaited()
        assert lock.locked(), "timed-out waiter must not release another task's lock"
    finally:
        lock.release()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert not lock.locked()


async def test_queue_and_real_graph_share_outer_deadline(service, deadlines):
    captured, created = deadlines
    cancelled = asyncio.Event()
    remaining_deadline = None

    async def invoke(messages):
        assert captured[0][1].when() == remaining_deadline
        assert len(captured) == 2, "service budget remains active around the graph budget"
        captured[0][1].reschedule(asyncio.get_running_loop().time() - 1)
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    model = install_model(service, invoke)
    lock = service._get_session_lock(1, "shared")
    await lock.acquire()
    task = asyncio.create_task(service.ask(1, "shared", "queued"))
    try:
        await created.wait()
        assert captured[0][0] == 600
        remaining_deadline = asyncio.get_running_loop().time() + 60
        captured[0][1].reschedule(remaining_deadline)
    finally:
        lock.release()
    result = await task
    assert result.degraded and cancelled.is_set()
    model.ainvoke.assert_awaited_once()
    assert not lock.locked()
    assert not service._chat_graph._session_locks[(1, "shared")].locked()
    assert all(row["role"] == "user" for row in await service.get_messages("shared", user_id=1))
    model.ainvoke.side_effect = None
    model.ainvoke.return_value = AIMessage(content="recovered")
    assert (await service.ask(1, "shared", "retry")).answer == "recovered"


@pytest.mark.parametrize("stage", ["queue", "model"])
async def test_service_external_cancellation_propagates(service, deadlines, stage):
    _, created = deadlines
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def invoke(messages):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    model = install_model(service, invoke)
    lock = service._get_session_lock(1, "shared")
    if stage == "queue":
        await lock.acquire()
    task = asyncio.create_task(service.ask(1, "shared", "question"))
    try:
        await (created.wait() if stage == "queue" else entered.wait())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        if stage == "queue":
            model.ainvoke.assert_not_awaited()
            assert lock.locked()
        else:
            assert cancelled.is_set()
            assert not lock.locked()
            assert not service._chat_graph._session_locks[(1, "shared")].locked()
    finally:
        if stage == "queue":
            lock.release()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_same_session_different_users_do_not_block_each_other(service):
    entered, release = asyncio.Event(), asyncio.Event()

    async def invoke(messages):
        if messages[-1].content == "user-one":
            entered.set()
            await release.wait()
        return AIMessage(content="done")

    install_model(service, invoke)
    first = asyncio.create_task(service.ask(1, "shared", "user-one"))
    try:
        await entered.wait()
        second = await asyncio.wait_for(service.ask(2, "shared", "user-two"), timeout=2)
        assert second.answer == "done"
        assert not first.done()
    finally:
        release.set()
        await first
    assert service._get_session_lock(1, "shared") is not service._get_session_lock(2, "shared")


async def test_service_graph_exception_does_not_log_private_response(service):
    class ProviderError(RuntimeError):
        status_code = 503

    error = ProviderError("SYNTHETIC_OPAQUE_KEY SYNTHETIC_PRIVATE_REASONING")
    error.__cause__ = RuntimeError("SYNTHETIC_PRIVATE_REASONING")
    service._chat_graph = SimpleNamespace(ask=AsyncMock(side_effect=error))
    records, rendered = [], []

    def sink(message):
        records.append(message.record)
        rendered.append(str(message))

    handler = logger.add(sink)
    try:
        result = await service.ask(1, "shared", "question")
    finally:
        logger.remove(handler)
    assert result.degraded
    assert records and all(record["exception"] is None for record in records)
    text = "".join(rendered)
    assert "type=ProviderError status=503" in text
    assert "SYNTHETIC_OPAQUE_KEY" not in text
    assert "SYNTHETIC_PRIVATE_REASONING" not in text


async def test_history_api_real_service_and_db_enforce_owner_before_limit(service):
    repo = service._chat_repo
    await repo.append("shared", "user", "OWN_HISTORY", user_id=1)
    for i in range(25):
        await repo.append("shared", "user", f"FOREIGN_HISTORY_{i}", user_id=2)
    await repo.append("foreign-only", "assistant", "FOREIGN_ONLY", user_id=2)
    assert [row["content"] for row in await service.get_messages("shared", user_id=1, limit=1)] == [
        "OWN_HISTORY",
    ]
    assert len(await service.get_messages("shared", user_id=2)) == 20
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.state.chat_service = service
    app.state.migration_applied = True
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        shared = await client.get("/api/v1/chat/shared/messages?user_id=2")
        foreign = await client.get("/api/v1/chat/foreign-only/messages")
        missing = await client.get("/api/v1/chat/missing/messages")
    assert shared.status_code == foreign.status_code == missing.status_code == 200
    assert [row["content"] for row in shared.json()] == ["OWN_HISTORY"]
    assert foreign.json() == missing.json() == []
    assert "FOREIGN" not in shared.text + foreign.text


async def test_history_api_cannot_read_a_foreign_session(service):
    await service._chat_repo.append("foreign", "assistant", "FOREIGN_SECRET", user_id=2)
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.state.chat_service = service
    app.state.migration_applied = True
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/chat/foreign/messages")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.parametrize(("method", "path", "service_method"), [
    ("POST", "/api/v1/chat", "ask"),
    ("GET", "/api/v1/chat/sessions", "list_sessions"),
    ("GET", "/api/v1/chat/shared/messages", "get_messages"),
])
async def test_route_unexpected_errors_do_not_log_private_response(
    service, monkeypatch, method, path, service_method,
):
    class ProviderError(RuntimeError):
        status_code = 503

    error = ProviderError("SYNTHETIC_OPAQUE_KEY SYNTHETIC_PRIVATE_REASONING")
    error.__cause__ = RuntimeError("SYNTHETIC_PRIVATE_REASONING")
    monkeypatch.setattr(service, service_method, AsyncMock(side_effect=error))
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router, prefix="/api/v1")
    app.state.chat_service = service
    app.state.migration_applied = True
    rendered, records = [], []

    def sink(message):
        rendered.append(str(message))
        records.append(message.record)

    handler = logger.add(sink)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.request(
                method, path, **({"json": {"message": "synthetic"}} if method == "POST" else {}),
            )
    finally:
        logger.remove(handler)
    assert response.status_code == 500
    assert records and all(record["exception"] is None for record in records)
    text = "".join(rendered) + response.text
    assert "type=ProviderError status=503" in text
    assert "SYNTHETIC_OPAQUE_KEY" not in text
    assert "SYNTHETIC_PRIVATE_REASONING" not in text
