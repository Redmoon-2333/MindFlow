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
    ) -> None:
        self._host = host
        self._port = port
        self._max_restarts = max_restarts
        self._window_s = window_s
        self._crash_times: list[float] = []

    async def run_forever(self) -> None:
        """Run the server with watchdog supervision."""
        logger.info(
            "Starting MindFlow watchdog (max {} restarts/hour)",
            self._max_restarts,
        )

        while True:
            app = create_app(get_settings())
            config = Config(
                app=app,
                host=self._host,
                port=self._port,
                log_level="info",
                # WS auth uses ?token= query param; uvicorn access logs would
                # record the full request line including the token (review P1-2).
                access_log=False,
            )
            server = Server(config)

            logger.info("uvicorn server starting on {}:{}", self._host, self._port)

            try:
                await server.serve()
            except Exception as exc:
                logger.opt(exception=True).error("Server crashed: {}", exc)
            else:
                logger.info("Server stopped cleanly")

            if not self._should_restart():
                logger.info("Max restarts reached or manual exit — watchdog stopping")
                break

            wait = self._backoff_delay()
            logger.info("Restarting in {:.0f}s (attempt #{})", wait, len(self._crash_times))
            await asyncio.sleep(wait)

    def _should_restart(self) -> bool:
        """Check if the server should be restarted (crash-loop detection)."""
        now = time.time()
        self._crash_times = [t for t in self._crash_times if now - t < self._window_s]

        return not len(self._crash_times) >= self._max_restarts

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
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _signal_handler)

    watchdog_task = asyncio.create_task(watchdog.run_forever())

    await asyncio.wait(
        {watchdog_task, asyncio.create_task(stop_event.wait())},
        return_when=asyncio.FIRST_COMPLETED,
    )

    watchdog_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await watchdog_task

    logger.info("MindFlow stopped")


if __name__ == "__main__":
    asyncio.run(main())
