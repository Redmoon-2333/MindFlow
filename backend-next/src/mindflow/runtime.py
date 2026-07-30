"""Application runtime service aggregate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mindflow.config import Settings
from mindflow.ports import ScheduledJobRunsPort


@dataclass(slots=True)
class RuntimeServices:
    settings: Settings
    engine: Any
    session_factory: Any
    scheduled_job_runs_repository: ScheduledJobRunsPort
    scheduler: Any | None = None
    collector_service: Any | None = None
    input_telemetry_service: Any | None = None
    panel_service: Any | None = None
    chat_service: Any | None = None
    chat_graph: Any | None = None
    llm_service: Any | None = None
    prediction_service: Any | None = None
    evidence_builder: Any | None = None
    provider_registry: Any | None = None
    workflow_port: Any | None = None
    checkpointer: Any | None = None
