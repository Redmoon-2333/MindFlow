"""Application entry point — uvicorn.Server programming launch + watchdog.

Per §5.1 of the architecture design:
  - Starts uvicorn programmatically (not via CLI)
  - A watchdog coroutine monitors the server and restarts on crash
  - Maximum 3 restarts per hour (crash-loop protection per NF-R1)
  - Graceful shutdown on SIGINT/SIGTERM

Usage:
    python -m mindflow.main
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import time
from collections.abc import Awaitable, Callable

from loguru import logger
from uvicorn import Config, Server

from mindflow.app import create_app
from mindflow.config import get_settings

_MAX_RESTARTS_PER_HOUR = 3
"""Maximum number of server restarts within a rolling 1-hour window (NF-R1)."""


class Watchdog:
    """Monitors the uvicorn server and restarts on crash (NF-R1).

    Args:
        host: Bind address.
        port: Bind port.
        max_restarts: Maximum restarts in the rolling window.
        window_s: Rolling window in seconds (default 3600 for 1 hour).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        max_restarts: int = _MAX_RESTARTS_PER_HOUR,
        window_s: float = 3600.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._host = host
        self._port = port
        self._max_restarts = max_restarts
        self._window_s = window_s
        self._clock = clock
        self._sleep = sleep
        self._crash_times: list[float] = []
        self._server: Server | None = None
        self._is_stopping = False

    async def run_forever(self) -> None:
        """Run the server with watchdog supervision."""
        logger.info(
            "Starting MindFlow watchdog (max {} restarts/hour)",
            self._max_restarts,
        )

        while not self._is_stopping:
            did_crash = False
            try:
                app = create_app(get_settings())
                config = Config(
                    app=app,
                    host=self._host,
                    port=self._port,
                    log_level="info",
                    # Keep local API request metadata out of persistent logs.
                    # Browser and WebSocket auth use an HttpOnly session cookie;
                    # the launcher root token is never placed in request URLs.
                    access_log=False,
                )
                self._server = Server(config)

                logger.info("uvicorn server starting on {}:{}", self._host, self._port)
                await self._server.serve()
            except Exception as exc:
                did_crash = True
                logger.opt(exception=True).error("Server crashed: {}", exc)
            else:
                logger.info("Server stopped cleanly")
            finally:
                self._server = None

            if not did_crash or self._is_stopping:
                break

            if not self._should_restart():
                logger.info("Maximum restart count reached - watchdog stopping")
                break

            wait = self._backoff_delay()
            logger.info("Restarting in {:.0f}s (attempt #{})", wait, len(self._crash_times))
            await self._sleep(wait)

    def stop(self) -> None:
        """Request a graceful stop of the active server and watchdog loop."""
        self._is_stopping = True
        if self._server is not None:
            self._server.should_exit = True

    def _should_restart(self) -> bool:
        """Register a restart unless the rolling-window limit is exhausted."""
        now = self._clock()
        self._crash_times = [t for t in self._crash_times if now - t < self._window_s]
        if len(self._crash_times) >= self._max_restarts:
            return False
        self._crash_times.append(now)
        return True

    def _backoff_delay(self) -> float:
        """Return a delay before restart, with linear backoff."""
        count = len(self._crash_times)
        if count == 0:
            return 0.5
        return min(1.0 * count, 5.0)


async def main() -> None:
    """Main entry point — runs the watchdog loop."""
    settings = get_settings()

    watchdog = Watchdog(
        host=settings.host,
        port=settings.port,
    )

    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Received shutdown signal")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, AttributeError):
            loop.add_signal_handler(sig, _signal_handler)

    watchdog_task = asyncio.create_task(watchdog.run_forever())
    stop_task = asyncio.create_task(stop_event.wait())

    done, _pending = await asyncio.wait(
        {watchdog_task, stop_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    if stop_task in done and not watchdog_task.done():
        watchdog.stop()
        await watchdog_task

    if not stop_task.done():
        stop_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stop_task

    logger.info("MindFlow stopped")


if __name__ == "__main__":
    asyncio.run(main())
