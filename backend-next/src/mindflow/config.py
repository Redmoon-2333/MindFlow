"""Application configuration via Pydantic BaseSettings.

Configuration source priority (highest to lowest):
  1. Environment variables
  2. .env file (in platformdirs user data dir)
  3. Default values

All datetime values are timezone-aware UTC throughout the application.
"""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import platformdirs
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogSettings(BaseSettings):
    """Structured logging configuration via loguru."""

    level: str = Field(default="DEBUG", description="Log level: DEBUG|INFO|WARNING|ERROR|CRITICAL")
    json_format: bool = Field(default=False, description="Emit JSON-structured logs (production)")
    rotation: str = Field(default="10 MB", description="Log file rotation threshold")
    retention: str = Field(default="30 days", description="Log file retention period")
    compression: str = Field(default="gz", description="Log file compression format")


class LLMSettings(BaseSettings):
    """LLM API configuration for attribution pipeline (Wave 6).

    Three-tier degradation chain (Architecture §3.3):
      L1: DeepSeek / OpenAI-compatible API (api_key + base_url + model)
      L2: Ollama local (ollama_enabled + ollama_base_url + ollama_model)
      L3: RuleEngine (always available, zero config)
    """

    timeout_s: int = Field(default=30, ge=1, le=300, description="LLM request timeout in seconds")
    max_retries: int = Field(default=1, ge=0, le=10, description="LLM retry budget")
    api_key: str | None = Field(default=None, description="LLM API key (e.g. DeepSeek)")
    base_url: str | None = Field(default=None, description="LLM API base URL")
    model: str | None = Field(default=None, description="LLM model identifier")
    ollama_enabled: bool = Field(default=False, description="Enable Ollama local fallback (L2)")
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama API base URL"
    )
    ollama_model: str = Field(default="qwen3:8b", description="Ollama model name")


_cached_data_dir: Path | None = None


def _get_data_dir() -> Path:
    """Return platform-appropriate user data directory (cached)."""
    global _cached_data_dir
    if _cached_data_dir is None:
        _cached_data_dir = Path(platformdirs.user_data_dir("mindflow", ensure_exists=True))
    return _cached_data_dir


class Settings(BaseSettings):
    """Application-wide settings.

    Priority: env vars > .env file > defaults.
    The .env file is searched in platformdirs user data directory.
    """

    model_config = SettingsConfigDict(
        env_prefix="MINDFLOW_",
        env_file=None,
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )

    # --- Runtime paths ---
    data_dir: Path = Field(
        default_factory=_get_data_dir,
        description="Application data directory; relative paths are anchored to platform data",
    )
    models_dir: Path = Field(
        default=Path("models"),
        description="ML model directory; relative paths are anchored to data_dir",
    )

    @property
    def backup_dir(self) -> Path:
        """Directory used for database backups."""
        return self.data_dir / "backups"

    @property
    def token_path(self) -> Path:
        """Path to the local API authentication token."""
        return self.data_dir / "token"

    # --- Database ---
    db_url: str = Field(
        default="sqlite+aiosqlite:///{data_dir}/mindflow.db",
        description="SQLAlchemy async database URL",
    )

    # --- Server ---
    host: str = Field(default="127.0.0.1", description="Bind address")
    port: int = Field(default=8765, description="Bind port")
    timezone: str = Field(
        default="local",
        description="Local business timezone: 'local' or an IANA timezone name",
    )

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, value: str) -> str:
        if value == "local":
            return value
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            msg = f"Unknown timezone: {value}"
            raise ValueError(msg) from exc
        return value

    # --- Runtime roles ---
    run_scheduler: bool = Field(default=True, description="Run scheduled background jobs")
    run_collectors: bool = Field(default=True, description="Run activity and input collectors")

    # --- Collector ---
    collect_interval_s: int = Field(
        default=5, ge=1, le=60, description="Collector tick interval in seconds"
    )
    heartbeat_pulsetime_s: int = Field(
        default=10, ge=1, le=300, description="Heartbeat merge window in seconds"
    )

    # --- Data Retention ---
    event_retention_days: int = Field(default=30, description="Raw event retention in days (7-90)")
    workflow_retention_days: int = Field(
        default=30, description="Workflow run retention in days (7-90). Completed/failed/cancelled runs "
        "older than this are cleaned up. Analyses and chat messages are preserved."
    )
    stale_run_timeout_minutes: int = Field(
        default=60, description="Minutes before a run stuck in 'running' status is marked as 'failed'"
    )

    @field_validator("event_retention_days")
    @classmethod
    def _validate_retention(cls, v: int) -> int:
        if not 7 <= v <= 90:
            msg = f"event_retention_days must be between 7 and 90, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("workflow_retention_days")
    @classmethod
    def _validate_workflow_retention(cls, v: int) -> int:
        if not 7 <= v <= 90:
            msg = f"workflow_retention_days must be between 7 and 90, got {v}"
            raise ValueError(msg)
        return v

    # --- Logging ---
    log: LogSettings = Field(default_factory=LogSettings)

    # --- Chat ---
    max_history_rounds: int = Field(
        default=10, ge=1, le=100, description="Max conversation rounds kept verbatim in chat"
    )

    # --- Auto-intervention ---
    auto_intervention_min_confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Min confidence to trigger auto-intervention"
    )
    auto_intervention_panel_confidence: float = Field(
        default=0.75, ge=0.0, le=1.0, description="Confidence threshold for panel escalation"
    )
    intervention_start_hour: int = Field(
        default=8, ge=0, le=23, description="Auto-intervention window start hour (local 24h)"
    )
    intervention_end_hour: int = Field(
        default=23, ge=0, le=24,
        description="Auto-intervention window end hour (exclusive, local 24h)",
    )

    # --- Intervention throttle ---
    throttle_daily_limit: int = Field(
        default=3, ge=1, le=20, description="Max interventions per user per day"
    )
    throttle_type_limit: int = Field(
        default=2, ge=1, le=10, description="Max interventions of same type per day"
    )
    throttle_cooldown_hours: float = Field(
        default=2.0, ge=0.5, le=24.0, description="Min hours between interventions"
    )
    throttle_ignore_rate_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Ignore rate above which fatigue kicks in"
    )
    throttle_fatigue_daily_limit: int = Field(
        default=1, ge=1, le=10, description="Reduced daily cap when fatigued"
    )
    throttle_annoying_threshold: int = Field(
        default=3, ge=1, le=20, description="Annoying feedback count that reduces type limit"
    )

    # --- Human Review Interrupt (Todo 10) ---
    human_review_enabled: bool = Field(
        default=False,
        description="Enable human review interrupt on low-confidence/high-disagreement verdicts",
    )
    human_review_confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Confidence below which human review is triggered",
    )
    human_review_disagreement_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Disagreement strength above which human review is triggered (1.0 - agreement_strength)",
    )

    # --- Graph orchestration (ADR-005 — default to legacy paths until cutover) ---
    graph_version: int = Field(default=1, description="LangGraph version (1=legacy, 2=new)")
    checkpointing_enabled: bool = Field(
        default=False, description="Enable LangGraph checkpoint persistence"
    )
    new_analysis_graph: bool = Field(
        default=False, description="Use v2 analysis graph (default: legacy v1)"
    )
    new_chat_graph: bool = Field(
        default=False, description="Use v2 chat graph (default: legacy v1)"
    )
    shadow_mode_chat: bool = Field(
        default=False, description="Run both legacy and new chat paths, compare, return legacy output"
    )

    # --- LLM placeholder ---
    llm: LLMSettings = Field(default_factory=LLMSettings)

    @model_validator(mode="after")
    def _resolve_runtime_paths(self) -> Settings:
        """Anchor runtime paths to the platform data directory, never cwd."""
        if not self.data_dir.is_absolute():
            self.data_dir = _get_data_dir() / self.data_dir
        self.data_dir = self.data_dir.expanduser()

        if not self.models_dir.is_absolute():
            self.models_dir = self.data_dir / self.models_dir
        self.models_dir = self.models_dir.expanduser()

        if "{data_dir}" in self.db_url:
            self.db_url = self.db_url.format(data_dir=self.data_dir.as_posix())
        return self


SETTINGS: Settings | None = None


def get_settings() -> Settings:
    """Return cached application settings (global singleton).

    The .env file is loaded from the platform data directory (platformdirs).
    Environment variables with MINDFLOW_ prefix override .env values,
    which in turn override default values.
    """
    global SETTINGS

    if SETTINGS is not None:
        return SETTINGS

    data_dir = _get_data_dir()
    env_path = data_dir / ".env"

    SETTINGS = Settings(_env_file=env_path) if env_path.exists() else Settings()  # type: ignore[call-arg]

    return SETTINGS
