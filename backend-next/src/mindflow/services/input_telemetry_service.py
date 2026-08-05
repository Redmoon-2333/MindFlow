"""Lifecycle manager for the Windows aggregate input watcher."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import multiprocessing
import os
import queue
from datetime import datetime
from typing import Any

from loguru import logger

from mindflow.infrastructure.collectors.input_watcher import run_raw_input_watcher
from mindflow.infrastructure.repositories.activity import SQLAlchemyActivityRepository
from mindflow.infrastructure.repositories.telemetry import TelemetryRepository


class InputTelemetryService:
    def __init__(
        self,
        telemetry_repository: TelemetryRepository,
        activity_repository: SQLAlchemyActivityRepository,
        user_id: int = 1,
        bucket_seconds: int = 30,
    ) -> None:
        self._telemetry_repository = telemetry_repository
        self._activity_repository = activity_repository
        self._user_id = user_id
        self._bucket_seconds = bucket_seconds
        self._process: Any = None
        self._queue: Any = None
        self._stop_event: Any = None
        self._drain_task: asyncio.Task[None] | None = None
        self._status = "stopped" if os.name == "nt" else "unavailable"

    @property
    def status(self) -> str:
        return self._status

    async def start(self) -> None:
        if os.name != "nt":
            self._status = "unavailable"
            return
        if self._process is not None and self._process.is_alive():
            return
        context = multiprocessing.get_context("spawn")
        self._queue = context.Queue()
        self._stop_event = context.Event()
        self._process = context.Process(
            target=run_raw_input_watcher,
            args=(self._queue, self._stop_event, self._bucket_seconds),
            name="mindflow-input-watcher",
            daemon=True,
        )
        self._process.start()
        self._drain_task = asyncio.create_task(self._drain_loop())
        self._status = "starting"

    async def stop(self) -> None:
        if self._process is None:
            return
        self._status = "stopping"
        self._stop_event.set()
        await asyncio.to_thread(self._process.join, self._bucket_seconds + 5)
        if self._process.is_alive():
            self._process.terminate()
            await asyncio.to_thread(self._process.join, 5)
        if self._drain_task is not None:
            self._drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._drain_task
        self._process = None
        self._drain_task = None
        self._status = "stopped"

    async def _drain_loop(self) -> None:
        while self._process is not None:
            try:
                message = await asyncio.to_thread(self._queue.get, True, 1.0)
            except queue.Empty:
                if not self._process.is_alive():
                    self._status = "degraded"
                    break
                continue
            if message.get("status") == "running":
                self._status = "running"
                continue
            if "error" in message:
                self._status = "degraded"
                logger.warning("Input watcher degraded: {}", message["error"])
                continue
            await self._persist_bucket(message)

    async def _persist_bucket(self, message: dict[str, Any]) -> None:
        event = await self._activity_repository.last_event(self._user_id)
        if event is None:
            context_key = "unknown"
        else:
            # Privacy: the key derives from process_name alone — window_title
            # never contributes to the stored context_key.
            context_key = (
                f"{event.data.process_name.lower()}:"
                f"{hashlib.sha256(event.data.process_name.encode()).hexdigest()[:16]}"
            )
        await self._telemetry_repository.save_interaction_bucket(
            user_id=self._user_id,
            window_start_utc=datetime.fromisoformat(message["window_start_utc"]),
            duration_s=float(message["duration_s"]),
            context_key=context_key,
            keypress_count=int(message["keypress_count"]),
            mouse_click_count=int(message["mouse_click_count"]),
            scroll_delta=int(message["scroll_delta"]),
            mouse_distance_px=float(message["mouse_distance_px"]),
            input_active_s=float(message["input_active_s"]),
            interaction_burst_count=int(message["interaction_burst_count"]),
        )
