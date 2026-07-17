"""Application configuration via Pydantic BaseSettings.

Configuration source priority (highest to lowest):
  1. Environment variables
  2. .env file (in platformdirs user data dir)
  3. Default values

All datetime values are timezone-aware UTC throughout the application.
"""

from __future__ import annotations

from pathlib import Path

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
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # --- Database ---
    db_url: str = Field(
        default="sqlite+aiosqlite:///{data_dir}/mindflow.db",
        description="SQLAlchemy async database URL",
    )

    # --- Server ---
    host: str = Field(default="127.0.0.1", description="Bind address")
    port: int = Field(default=8765, description="Bind port")

    # --- Collector ---
    collect_interval_s: int = Field(
        default=5, ge=1, le=60, description="Collector tick interval in seconds"
    )
    heartbeat_pulsetime_s: int = Field(
        default=10, ge=1, le=300, description="Heartbeat merge window in seconds"
    )

    # --- Data Retention ---
    event_retention_days: int = Field(default=30, description="Raw event retention in days (7-90)")

    @field_validator("event_retention_days")
    @classmethod
    def _validate_retention(cls, v: int) -> int:
        if not 7 <= v <= 90:
            msg = f"event_retention_days must be between 7 and 90, got {v}"
            raise ValueError(msg)
        return v

    # --- Logging ---
    log: LogSettings = Field(default_factory=LogSettings)

    # --- LLM placeholder ---
    llm: LLMSettings = Field(default_factory=LLMSettings)

    @model_validator(mode="after")
    def _resolve_db_url(self) -> Settings:
        """Resolve {data_dir} placeholder in db_url."""
        if "{data_dir}" in self.db_url:
            self.db_url = self.db_url.format(data_dir=_get_data_dir())
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
