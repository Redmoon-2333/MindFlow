from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import AsyncMock

import pytest

import mindflow.main as main_module
from mindflow.main import Watchdog


class FakeServer:
    outcomes: Iterator[Exception | None]
    instances: list[FakeServer] = []

    def __init__(self, config: object) -> None:
        self.config = config
        self.should_exit = False
        self.instances.append(self)

    async def serve(self) -> None:
        outcome = next(self.outcomes)
        if outcome is not None:
            raise outcome


@pytest.fixture(autouse=True)
def reset_fake_server() -> None:
    FakeServer.instances = []


def configure_watchdog_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    outcomes: list[Exception | None],
) -> AsyncMock:
    FakeServer.outcomes = iter(outcomes)
    sleep = AsyncMock()
    monkeypatch.setattr(main_module, "Server", FakeServer)
    monkeypatch.setattr(main_module, "Config", lambda **kwargs: kwargs)
    monkeypatch.setattr(main_module, "create_app", lambda settings: object())
    monkeypatch.setattr(main_module, "get_settings", lambda: object())
    return sleep


async def test_watchdog_does_not_restart_after_clean_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep = configure_watchdog_dependencies(monkeypatch, [None])

    watchdog = Watchdog(max_restarts=3, sleep=sleep)
    await watchdog.run_forever()

    assert len(FakeServer.instances) == 1
    sleep.assert_not_awaited()
    assert watchdog._crash_times == []


async def test_watchdog_records_crash_and_restarts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep = configure_watchdog_dependencies(monkeypatch, [RuntimeError("boom"), None])
    watchdog = Watchdog(max_restarts=3, clock=lambda: 100.0, sleep=sleep)
    await watchdog.run_forever()

    assert len(FakeServer.instances) == 2
    sleep.assert_awaited_once_with(1.0)
    assert watchdog._crash_times == [100.0]


async def test_watchdog_stops_after_maximum_restarts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep = configure_watchdog_dependencies(
        monkeypatch,
        [RuntimeError("one"), RuntimeError("two"), RuntimeError("three")],
    )
    clock_values = iter([100.0, 101.0, 102.0])
    watchdog = Watchdog(
        max_restarts=2,
        clock=lambda: next(clock_values),
        sleep=sleep,
    )
    await watchdog.run_forever()

    assert len(FakeServer.instances) == 3
    assert [call.args[0] for call in sleep.await_args_list] == [1.0, 2.0]
    assert watchdog._crash_times == [100.0, 101.0]


def test_watchdog_prunes_restarts_outside_rolling_window() -> None:
    watchdog = Watchdog(max_restarts=2, window_s=60.0, clock=lambda: 100.0)
    watchdog._crash_times = [20.0, 50.0]

    assert watchdog._should_restart()
    assert watchdog._crash_times == [50.0, 100.0]
    assert not watchdog._should_restart()


def test_watchdog_stop_requests_active_server_exit() -> None:
    watchdog = Watchdog()
    server = FakeServer({})
    watchdog._server = server

    watchdog.stop()

    assert watchdog._is_stopping
    assert server.should_exit


def test_watchdog_backoff_is_linear_and_capped() -> None:
    watchdog = Watchdog()

    assert watchdog._backoff_delay() == 0.5
    watchdog._crash_times = [1.0]
    assert watchdog._backoff_delay() == 1.0
    watchdog._crash_times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert watchdog._backoff_delay() == 5.0
