"""Disposable offline server for exhaustive, high-volume functional acceptance."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import cast
from unittest.mock import patch

import httpx
import uvicorn

import mindflow.app as app_module
from mindflow import config
from mindflow.api.middleware.ratelimit import RateLimitMiddleware
from mindflow.infrastructure.notification import LogOnlyNotifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8866)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    if Path(tempfile.gettempdir()).resolve() not in data_dir.parents:
        parser.error("Acceptance data must stay below the system temporary directory")
    for name in tuple(os.environ):
        if name.startswith("MINDFLOW_") or name in {
            "API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY",
            "LANGCHAIN_API_KEY", "LANGSMITH_API_KEY",
            "LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING",
        }:
            os.environ.pop(name)

    def deny_http(*_args: object, **_kwargs: object) -> httpx.Response:
        raise RuntimeError("External HTTP is disabled during offline acceptance")

    async def deny_async_http(*_args: object, **_kwargs: object) -> httpx.Response:
        return deny_http()
    data_dir.mkdir(parents=True, exist_ok=True)
    config._cached_data_dir = data_dir
    settings = config.Settings(
        data_dir=data_dir,
        models_dir=data_dir / "models",
        port=args.port,
        collect_interval_s=1,
        run_collectors=False,
        run_scheduler=False,
        llm=config.LLMSettings(api_key=None, deepseek_api_key=None, ollama_enabled=False),
    )
    config.SETTINGS = settings
    app = app_module.create_app(settings, configure_logging=False)
    # Only this disposable app instance changes; production defaults stay intact.
    for middleware in app.user_middleware:
        if cast(object, middleware.cls) is RateLimitMiddleware:
            middleware.kwargs.update(
                global_capacity=100_000.0,
                global_refill_rate=100_000.0,
                endpoint_limits={},
            )
    with (
        patch("httpx.HTTPTransport.handle_request", deny_http),
        patch("httpx.AsyncHTTPTransport.handle_async_request", deny_async_http),
        patch("mindflow.app.create_notifier", return_value=LogOnlyNotifier()),
    ):
        uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
